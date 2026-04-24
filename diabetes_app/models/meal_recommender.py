"""
models/meal_recommender.py
==========================
Personalized meal recommendation engine.
Fetches plans from SQLite and calculates nutrition from food entries.
"""

import sqlite3
import json


class MealRecommender:
    """
    Provides personalized meal plans and nutrition calculation.

    Personalization factors:
      - Diabetes type (type1 / type2)
      - Glucose level (normal / high / low)
      - Region preference (south_indian / north_indian / pan_indian)
      - Dietary preference (vegetarian / non-vegetarian)
      - Age and BMI (portion size adjustments)
    """

    SERVING_MAP = {
        'cup': 240, 'bowl': 200, 'plate': 300, 'glass': 240,
        'piece': 80, 'pieces': 80, 'slice': 60, 'handful': 30,
        'tbsp': 15, 'tablespoon': 15, 'tsp': 5, 'teaspoon': 5,
        'serving': 100, 'portion': 100, 'small': 60, 'medium': 100, 'large': 150,
    }

    def __init__(self, db_path: str):
        self.db_path = db_path

    # ── public ────────────────────────────────────────────────────────────────
    def get_plans(self, diabetes_type: str, meal_time: str,
                  glucose_level: float, profile: dict = None) -> dict:
        """
        Fetch meal plans from DB filtered by diabetes type, meal time,
        glucose range, and user profile preferences.
        """
        glc_range = self._glucose_range(glucose_level)
        profile   = profile or {}
        region    = profile.get('region', 'south_indian')
        diet_pref = profile.get('dietary_pref', 'vegetarian')

        if meal_time == 'all':
            result = {}
            for t in ['breakfast', 'lunch', 'dinner', 'snacks']:
                plans = self._fetch_plans(diabetes_type, t, glc_range)
                result[t] = self._enrich_plans(plans, profile)
        else:
            plans = self._fetch_plans(diabetes_type, meal_time, glc_range)
            result = {meal_time: self._enrich_plans(plans, profile)}

        avoid = self._foods_to_avoid(diabetes_type)
        return {
            "meal_plan":      result,
            "foods_to_avoid": avoid,
            "diabetes_type":  diabetes_type,
            "glucose_range":  glc_range,
            "region":         region,
        }

    def calculate_nutrition(self, items: list) -> tuple:
        """
        Calculate total nutrition for a list of (food_name, grams) tuples.

        Returns: (totals dict, breakdown list)
        """
        totals = {k: 0.0 for k in ['cal', 'carb', 'pro', 'fat', 'fib', 'glc']}
        rows   = []

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row

        for food_key, grams in items:
            fd = conn.execute(
                "SELECT * FROM foods WHERE name=?", (food_key,)
            ).fetchone()
            if not fd:
                continue
            fd = dict(fd)
            s  = grams / 100.0
            row = {
                'food':  food_key.replace('_', ' ').title(),
                'grams': grams,
                'cal':   round(fd['cal_100g'] * s, 1),
                'carb':  round(fd['carb_100g'] * s, 1),
                'pro':   round(fd['protein_100g'] * s, 1),
                'fat':   round(fd['fat_100g'] * s, 1),
                'fib':   round(fd['fiber_100g'] * s, 1),
                'glc':   round(fd['glucose_impact'] * s, 1),
                'gi':    fd['gi'],
                'gi_value': fd['gi_value'],
            }
            for k in totals:
                totals[k] += row[k]
            rows.append(row)

        conn.close()
        for k in totals:
            totals[k] = round(totals[k], 1)
        return totals, rows

    def get_food_by_gi(self, gi_level: str, region: str = None, limit: int = 15) -> list:
        """Return food names filtered by GI level and optional region."""
        conn = sqlite3.connect(self.db_path)
        if region:
            rows = conn.execute(
                "SELECT name FROM foods WHERE gi=? AND region LIKE ? "
                "ORDER BY gi_value ASC LIMIT ?",
                (gi_level, f'%{region}%', limit)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT name FROM foods WHERE gi=? ORDER BY gi_value ASC LIMIT ?",
                (gi_level, limit)
            ).fetchall()
        conn.close()
        return [r[0].replace('_', ' ').title() for r in rows]

    # ── private ───────────────────────────────────────────────────────────────
    def _fetch_plans(self, dtype: str, meal_time: str, glc_range: str) -> list:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """SELECT * FROM meal_plans
               WHERE diabetes_type=? AND meal_time=? AND glucose_range=?
               ORDER BY gi_rating ASC""",
            (dtype, meal_time, glc_range)
        ).fetchall()
        if not rows:
            rows = conn.execute(
                "SELECT * FROM meal_plans WHERE diabetes_type=? AND meal_time=?",
                (dtype, meal_time)
            ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def _enrich_plans(self, plans: list, profile: dict) -> list:
        """
        Add personalization notes to each plan and return enriched dicts.

        FIX SUMMARY:
          - 'meal_name' is now always present AND aliased as 'name'
            (frontend safety: m.meal_name || m.name both work)
          - gi_rating is normalised to lowercase before comparison
            (backend sends 'low'/'medium'/'high', not 'Low'/'Medium'/'High')
          - gi_rating None guard prevents AttributeError on .lower()
          - All numeric fields have explicit float() cast + fallback
          - foods JSON parse wrapped in try/except
        """
        result  = []
        glucose = float(profile.get('glucose_level') or 120)
        age     = float(profile.get('age') or 40)

        GI_ICONS = {"low": "🟢", "medium": "🟡", "high": "🔴"}

        for p in plans[:4]:
            # ── Safely parse foods JSON ───────────────────────────────
            try:
                foods_list = json.loads(p.get('foods') or '[]')
                if not isinstance(foods_list, list):
                    foods_list = []
            except (json.JSONDecodeError, TypeError):
                foods_list = []

            # ── Normalise GI to lowercase — DB stores lowercase already
            # but guard against None or unexpected casing ──────────────
            gi_raw    = p.get('gi_rating') or 'low'
            gi_rating = gi_raw.lower().strip()          # always lowercase
            gi_icon   = GI_ICONS.get(gi_rating, "⚪")

            # ── Personalisation notes ─────────────────────────────────
            notes = p.get('reason') or ''
            if glucose > 180:
                notes += " (Prioritized for high glucose management)"
            if age > 65:
                notes += " (Suitable for seniors — easily digestible)"

            # ── meal_name: guaranteed present, with 'name' alias ──────
            # ROOT CAUSE FIX: old JS used m.name (undefined) instead of
            # m.meal_name.  We now send BOTH keys so either works.
            meal_name = p.get('meal_name') or 'Unnamed Meal'

            result.append({
                # Primary key (correct)
                "meal_name":      meal_name,
                # Alias for backward compatibility with old JS (m.name)
                "name":           meal_name,

                "foods":          foods_list,
                "foods_display":  ', '.join(
                    f.replace('_', ' ').title() for f in foods_list
                ),

                # Numeric fields — explicit float() + fallback to 0
                "total_calories": float(p.get('total_cal')  or 0),
                "total_carbs":    float(p.get('total_carb') or 0),
                "total_glucose":  float(p.get('total_glc')  or 0),

                # GI — always lowercase string
                "gi_rating":      gi_rating,
                "gi_icon":        gi_icon,

                "reason":         notes,
            })
        return result

    def _foods_to_avoid(self, dtype: str) -> list:
        conn  = sqlite3.connect(self.db_path)
        cond  = "suitable_t2=0" if dtype == 'type2' else "suitable_t1=0"
        rows  = conn.execute(
            f"SELECT name FROM foods WHERE gi='high' OR {cond} LIMIT 10"
        ).fetchall()
        conn.close()
        return [r[0].replace('_', ' ').title() for r in rows]

    @staticmethod
    def _glucose_range(glucose: float) -> str:
        if glucose is None:     return 'normal'
        if glucose >= 180:      return 'high'
        if glucose < 80:        return 'low'
        return 'normal'
