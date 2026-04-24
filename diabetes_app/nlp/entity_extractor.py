"""
nlp/entity_extractor.py
========================
Extracts structured entities from free-text user messages:
  - Food names   (fuzzy match against DB aliases)
  - Portions     (grams / cups / pieces / servings)
  - Glucose val  (regex: "glucose is 220", "sugar 150")
  - Diabetes type (type1 / type2 keywords)
  - Age, BMI, HbA1c  (self-assessment queries)
  - Meal items   (for nutrition calculator)
"""

import re
import sqlite3
from difflib import SequenceMatcher


class EntityExtractor:
    """
    Extracts all relevant entities from a user message.

    Usage:
        extractor = EntityExtractor(db_path='database/diabetes_ai.db')
        entities  = extractor.extract("3 idlis and 1 cup sambar")
        food      = extractor.find_food("ragi dosa")
        items     = extractor.parse_meal("2 chapati + dal tadka + 100g bhindi")
    """

    SERVING_MAP = {
        'cup': 240, 'cups': 240, 'bowl': 200, 'bowls': 200,
        'plate': 300, 'plates': 300, 'glass': 240, 'glasses': 240,
        'piece': 80, 'pieces': 80, 'serving': 100, 'servings': 100,
        'portion': 100, 'portions': 100, 'handful': 30,
        'tbsp': 15, 'tablespoon': 15, 'tablespoons': 15,
        'tsp': 5, 'teaspoon': 5, 'teaspoons': 5,
        'small': 60, 'medium': 100, 'large': 150,
        'slice': 60, 'slices': 60,
    }

    NUMBER_WORDS = {
        'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
        'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
        'half': 0.5, 'quarter': 0.25, 'a': 1, 'an': 1,
    }

    def __init__(self, db_path: str):
        self.db_path  = db_path
        self._aliases = None   # built lazily

    # ── public ────────────────────────────────────────────────────────────────
    def extract(self, text: str) -> dict:
        """
        Extract all entities from text.

        Returns dict with:
          food, portion, glucose, diabetes_type, age, bmi, hba1c,
          meal_items (list of (food_key, grams)), numbers
        """
        ml = text.lower()
        return {
            'food':          self.find_food(ml),
            'portion':       self.find_number(ml),
            'glucose':       self.find_glucose_value(ml),
            'diabetes_type': self.find_diabetes_type(ml),
            'age':           self._find_age(ml),
            'bmi':           self._find_labelled_number('bmi', ml),
            'hba1c':         self._find_labelled_number('hba1c', ml),
            'meal_items':    self.parse_meal(text),
        }

    def find_food(self, text: str) -> str | None:
        """
        Find the best matching food name from text.
        Returns the food DB key (e.g. 'ragi_dosa') or None.
        """
        aliases = self._get_aliases()
        t       = text.lower()

        # Longest exact match first
        for alias, key in sorted(aliases.items(), key=lambda x: -len(x[0])):
            if alias in t:
                return key

        # Fuzzy fallback
        best_r, best_key = 0.0, None
        for alias, key in aliases.items():
            r = SequenceMatcher(None, alias, t).ratio()
            if r > best_r and r > 0.72:
                best_r, best_key = r, key
        return best_key

    def find_number(self, text: str) -> float:
        """Extract the first numeric value (portion grams) from text."""
        for pat in [r'(\d+(?:\.\d+)?)\s*(?:g|gram|grams|ml)',
                    r'(\d+(?:\.\d+)?)']:
            m = re.search(pat, text.lower())
            if m:
                return float(m.group(1))
        return 100.0

    def find_glucose_value(self, text: str) -> int | None:
        """Extract blood glucose value from text."""
        patterns = [
            r'(?:glucose|sugar|blood sugar|blood glucose)[^\d]*(\d+)',
            r'(\d{2,3})\s*(?:mg/dl|mg)',
        ]
        for pat in patterns:
            m = re.search(pat, text.lower())
            if m:
                val = int(m.group(1))
                if 30 <= val <= 600:   # plausible glucose range
                    return val
        return None

    def find_diabetes_type(self, text: str) -> str | None:
        """Detect diabetes type from text."""
        t = text.lower()
        if any(x in t for x in ['type 1', 'type1', 't1d', 'type one', 'insulin dependent']):
            return 'type1'
        if any(x in t for x in ['type 2', 'type2', 't2d', 'type two', 'non insulin']):
            return 'type2'
        return None

    def parse_meal(self, text: str) -> list:
        """
        Parse natural-language meal description into (food_key, grams) tuples.

        Examples:
          "3 idli + 1 cup sambar"  →  [('idli', 240), ('sambar', 240)]
          "2 chapati and dal"      →  [('chapati', 160), ('dal_tadka', 100)]
          "100g bhindi masala"     →  [('bhindi_masala', 100)]
        """
        results = []
        parts   = re.split(r'[+,]|\band\b|\bwith\b|\bplus\b', text.lower())

        for part in parts:
            part = part.strip()
            if not part:
                continue

            grams = self._parse_grams(part)
            food  = self.find_food(part)
            if food:
                results.append((food, round(grams, 1)))

        return results

    # ── private ───────────────────────────────────────────────────────────────
    def _get_aliases(self) -> dict:
        """Build food alias dict from DB (cached after first call)."""
        if self._aliases is not None:
            return self._aliases

        conn  = sqlite3.connect(self.db_path)
        rows  = conn.execute("SELECT name, name_local FROM foods").fetchall()
        conn.close()

        aliases = {}
        for name, local in rows:
            aliases[name.replace('_', ' ')] = name
            aliases[name] = name
            if local:
                aliases[local.lower()] = name
            words = name.split('_')
            if len(words) == 1 and len(words[0]) > 4:
                aliases[words[0]] = name

        self._aliases = aliases
        return aliases

    def _parse_grams(self, part: str) -> float:
        """Parse a portion string into grams."""
        # Explicit grams: "100g", "150 grams"
        gm = re.search(r'(\d+(?:\.\d+)?)\s*g(?:ram)?s?', part)
        if gm:
            return float(gm.group(1))

        # Serving size words: "1 cup", "2 pieces"
        for sv, sg in self.SERVING_MAP.items():
            if sv in part:
                nm = re.search(r'(\d+(?:\.\d+)?)\s*' + sv, part)
                if nm:
                    return sg * float(nm.group(1))
                return float(sg)     # "cup sambar" → 1 cup = 240g

        # Plain number: "3 idli", "two dosa"
        for word, val in self.NUMBER_WORDS.items():
            if re.search(r'\b' + word + r'\b', part):
                return 80.0 * val   # default ~80g per piece

        nm = re.search(r'(\d+(?:\.\d+)?)', part)
        if nm:
            return 80.0 * float(nm.group(1))

        return 100.0    # default 100g

    def _find_labelled_number(self, label: str, text: str) -> float | None:
        """Extract labelled number: 'bmi 28.5' or 'hba1c 6.2'."""
        m = re.search(rf'{label}[^\d]*(\d+(?:\.\d+)?)', text)
        return float(m.group(1)) if m else None

    def _find_age(self, text: str) -> int | None:
        """Extract age: '45 years old', 'age 50'."""
        m = re.search(r'(\d+)\s*years?\s*old', text)
        if m:
            return int(m.group(1))
        m = re.search(r'age[^\d]*(\d+)', text)
        return int(m.group(1)) if m else None
