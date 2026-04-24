"""
nlp/response_builder.py
========================
Generates chatbot responses using:
  1. RAG  — TF-IDF cosine similarity over SQLite Q&A database
  2. Intent dispatch — structured responses for known intents
  3. Entity-aware — food lookups, meal calculations, self-assessment
  4. Profile-aware — personalized by diabetes type, glucose level, region
"""

import sqlite3
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ResponseBuilder:
    """
    Builds natural language responses for the chatbot.

    Pipeline (in priority order):
      A. Nutrition Calculator  — if meal items detected
      B. Emergency Glucose     — if high/low glucose detected
      C. Self Assessment       — if numbers + risk query
      D. RAG Retrieval         — if strong Q&A match found (>0.35)
      E. Intent Dispatch       — structured handlers per intent
      F. Food Lookup           — if food entity found
      G. Default fallback
    """

    def __init__(self, db_path, risk_model, meal_rec, entity_ext):
        self.db_path    = db_path
        self.risk_model = risk_model
        self.meal_rec   = meal_rec
        self.entity_ext = entity_ext

        # Build RAG index from DB
        self._qa_questions = []
        self._qa_answers   = []
        self._qa_intents   = []
        self._rag_tfidf    = None
        self._rag_matrix   = None
        self._build_rag_index()

    # ── public ────────────────────────────────────────────────────────────────
    def build(self, message: str, intent: str, entities: dict,
              history: list, profile: dict) -> str:
        """
        Generate a response given message, intent, entities, and user profile.
        """
        ml      = message.lower()
        dtype   = entities.get('diabetes_type') or profile.get('diabetes_type', 'type2')
        glucose = (entities.get('glucose') or
                   float(profile.get('glucose_level') or 120))

        # ── A. Nutrition calculator ──────────────────────────────────────────
        items = entities.get('meal_items', [])
        calc_triggers = ['calculate', 'calc ', 'my meal', 'total nutrition',
                         'how many cal', 'what did i eat']
        is_calc = (any(t in ml for t in calc_triggers) or
                   ('+' in ml and items) or
                   (items and len(items) > 1) or
                   bool(re.search(r'\d+\s*(?:idli|chapati|roti|dosa|cup|bowl)', ml)))

        if is_calc and items:
            return self._resp_nutrition_calc(items)

        # ── B. Emergency glucose ─────────────────────────────────────────────
        glc_val = entities.get('glucose')
        if glc_val and glc_val >= 200:
            return self._resp_high_glucose(glc_val)
        if any(w in ml for w in ['low blood sugar', 'hypoglycemia', 'dizzy',
                                  'shaking low', 'faint', 'below 70', 'sugar low']):
            return self._resp_low_glucose()

        # ── C. Self-assessment ───────────────────────────────────────────────
        has_numbers = any(entities.get(k) for k in ['age', 'bmi', 'glucose', 'hba1c'])
        is_self = any(w in ml for w in ['am i diabetic', 'check my risk',
                                         'years old', 'is my sugar'])
        if has_numbers and is_self:
            return self._resp_self_assess(entities)

        # ── D. RAG retrieval ─────────────────────────────────────────────────
        rag_results = self._rag_retrieve(message, top_k=3)
        best_match  = rag_results[0] if rag_results else None
        if best_match and best_match[2] > 0.38:
            answer = best_match[1]
            return self._personalize_rag(answer, profile, entities)

        # ── E. Food lookup (high priority if food + nutrition keyword) ───────
        food = entities.get('food')
        if food and any(w in ml for w in ['nutrition', 'calori', 'carb', 'protein', 'fat',
                                           'fiber', 'gi', 'glycemic', 'good for',
                                           'ok for', 'safe for', 'suitable']):
            return self._resp_food_card(food, entities.get('portion', 100), profile)

        # ── F. Intent dispatch ───────────────────────────────────────────────
        resp = self._dispatch_intent(intent, ml, dtype, glucose, profile, entities)
        if resp:
            return resp

        # ── G. Food lookup fallback ──────────────────────────────────────────
        if food:
            return self._resp_food_card(food, entities.get('portion', 100), profile)

        # ── H. Weak RAG fallback ─────────────────────────────────────────────
        if best_match and best_match[2] > 0.15:
            return (best_match[1] +
                    f"\n\n_(Matched from knowledge base — confidence: "
                    f"{round(best_match[2]*100)}%)_")

        return self._resp_fallback()

    # ── RAG ───────────────────────────────────────────────────────────────────
    def _build_rag_index(self):
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            "SELECT intent, question, answer FROM chatbot_qa"
        ).fetchall()
        conn.close()

        self._qa_intents   = [r[0] for r in rows]
        self._qa_questions = [r[1] for r in rows]
        self._qa_answers   = [r[2] for r in rows]

        if self._qa_questions:
            self._rag_tfidf  = TfidfVectorizer(ngram_range=(1, 3), sublinear_tf=True)
            self._rag_matrix = self._rag_tfidf.fit_transform(self._qa_questions)
            print(f"   ✅ RAG index: {len(self._qa_questions)} Q&A pairs")

    def _rag_retrieve(self, query: str, top_k: int = 3) -> list:
        if self._rag_matrix is None:
            return []
        vec  = self._rag_tfidf.transform([query])
        sims = cosine_similarity(vec, self._rag_matrix).flatten()
        idxs = np.argsort(sims)[::-1][:top_k]
        return [(self._qa_questions[i], self._qa_answers[i],
                 float(sims[i])) for i in idxs]

    def _personalize_rag(self, answer: str, profile: dict, entities: dict) -> str:
        dtype   = profile.get('diabetes_type', '')
        glucose = float(profile.get('glucose_level') or 120)
        note    = ""
        if dtype == 'type2' and 'rice' in answer.lower():
            note = "\n\n💡 _As a Type 2 diabetic, prioritize ragi or brown rice over white rice._"
        elif glucose > 160 and ('breakfast' in answer.lower() or 'morning' in answer.lower()):
            note = f"\n\n💡 _Your glucose ({glucose} mg/dL) is elevated — choose ragi porridge or bitter gourd juice today._"
        return answer + note

    # ── Response builders ─────────────────────────────────────────────────────
    def _resp_nutrition_calc(self, items: list) -> str:
        totals, rows = self.meal_rec.calculate_nutrition(items)
        if not rows:
            return "🤔 I couldn't find those foods in my database. Try the Food Analyzer tab or check the food name spelling."

        lines = '\n'.join(
            f"  • {r['food']} ({r['grams']}g): "
            f"{r['cal']} kcal | Carbs {r['carb']}g | Glucose {r['glc']}g | GI {r['gi'].upper()}"
            for r in rows
        )
        glc_warn = (
            "🚨 Very high glucose load — split this meal or swap to ragi/dal."
            if totals['glc'] > 80 else
            "⚠️ Moderate — pair with extra vegetables and walk 15 min after."
            if totals['glc'] > 45 else
            "✅ Glucose load is manageable for diabetics."
        )
        return (f"🧮 **Meal Nutrition ({len(items)} food{'s' if len(items)>1 else ''}):**\n\n"
                f"{lines}\n\n"
                f"**── TOTAL ──**\n"
                f"• Calories: **{totals['cal']} kcal**\n"
                f"• Carbs: **{totals['carb']}g** | Protein: **{totals['pro']}g** | "
                f"Fat: **{totals['fat']}g** | Fiber: **{totals['fib']}g**\n"
                f"• Glucose Impact: **{totals['glc']}g** {glc_warn}")

    def _resp_high_glucose(self, glucose_val: int) -> str:
        safe_foods = self._db_query(
            "SELECT name FROM foods WHERE gi='low' AND glucose_impact<15 ORDER BY glucose_impact LIMIT 6"
        )
        safe_names = ', '.join(r[0].replace('_', ' ').title() for r in safe_foods)
        level = "Dangerously high" if glucose_val > 250 else "High"
        return (f"🚨 **{level} Blood Glucose ({glucose_val} mg/dL):**\n\n"
                f"**Immediate steps:**\n"
                f"• Drink 2–3 glasses of plain water NOW\n"
                f"• Do NOT eat high-carb foods\n"
                f"• Walk 20 minutes if able\n"
                f"• Check if you missed medication/insulin\n"
                f"• If >300 mg/dL → call your doctor immediately\n\n"
                f"✅ **Safe foods right now (lowest glucose impact):**\n{safe_names}\n\n"
                f"🚫 **Avoid:** Rice, Roti, Bread, Sweets, Fruits, Juice\n\n"
                f"⚠️ This is supportive guidance only. Always consult your doctor.")

    def _resp_low_glucose(self) -> str:
        return ("🆘 **Hypoglycemia (Low Blood Sugar) — Act Immediately:**\n\n"
                "**15-15 Rule:**\n"
                "1. Take **15g fast carbs RIGHT NOW:**\n"
                "   • 4–5 glucose tablets, OR\n"
                "   • Half cup fruit juice / regular soft drink, OR\n"
                "   • 3 tsp sugar dissolved in water, OR\n"
                "   • 2 tbsp raisins\n\n"
                "2. Wait **15 minutes** → recheck glucose\n"
                "3. If still <70 mg/dL → repeat step 1\n"
                "4. Once stable → eat small snack (roti + dal)\n\n"
                "🚨 If person is unconscious: **DO NOT give food/drink.**\n"
                "Call emergency services immediately.")

    def _resp_self_assess(self, entities: dict) -> str:
        risk = 0; flags = []
        g = entities.get('glucose')
        h = entities.get('hba1c')
        b = entities.get('bmi')
        a = entities.get('age')
        if g:
            if g >= 126:   risk += 30; flags.append(f"High glucose {g} mg/dL")
            elif g >= 100: risk += 15; flags.append(f"Pre-diabetic glucose {g} mg/dL")
        if h:
            if h >= 6.5:   risk += 30; flags.append(f"HbA1c {h}% — diabetic range")
            elif h >= 5.7: risk += 15; flags.append(f"HbA1c {h}% — pre-diabetic")
        if b:
            if b >= 30:    risk += 15; flags.append(f"Obese BMI {b}")
            elif b >= 25:  risk += 8;  flags.append(f"Overweight BMI {b}")
        if a and a >= 45:  risk += 10; flags.append(f"Age {a} ≥45")

        if not flags:
            return ("🩺 Please share your values:\n"
                    "'I am **45 years old**, **BMI 28**, **glucose 140**, **HbA1c 6.1** — am I diabetic?'\n\n"
                    "Or use the **Risk Assessment** tab for full ML analysis.")

        level = "🔴 **High**" if risk >= 50 else "🟡 **Moderate**" if risk >= 25 else "🟢 **Low**"
        flag_text = '\n'.join(f"  ⚠️ {f}" for f in flags)
        return (f"{level} **Risk ({risk}/100) — Quick Assessment:**\n\n"
                f"{flag_text}\n\n"
                f"💡 For full ML prediction (**{self.risk_model.accuracy['ensemble']}% accuracy**), "
                f"use the **Risk Assessment** tab.\n⚕️ Always confirm with your doctor.")

    def _resp_food_card(self, food_key: str, portion: float, profile: dict) -> str:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        fd   = conn.execute("SELECT * FROM foods WHERE name=?", (food_key,)).fetchone()
        conn.close()
        if not fd:
            return f"I don't have data for '{food_key}'. Try the Food Analyzer tab."
        fd    = dict(fd)
        s     = portion / 100
        gi_l  = {"low":  "✅ Low GI — safe for diabetics",
                 "medium":"⚠️ Medium GI — eat in moderation",
                 "high": "❌ High GI — avoid or limit"}[fd['gi']]
        local = f" ({fd['name_local']})" if fd.get('name_local') else ""
        dtype = profile.get('diabetes_type', 'type2')
        suit  = fd['suitable_t2'] if dtype == 'type2' else fd['suitable_t1']
        d_note = ("\n\n❌ **Not recommended for this diabetes type.**"
                  if suit == 0 else
                  "\n\n✅ **Suitable — control portion size.**"
                  if fd['gi'] in ('low', 'medium') else "")
        return (f"📊 **{fd['name'].replace('_',' ').title()}{local}** — {portion}g\n\n"
                f"• Calories: **{round(fd['cal_100g']*s, 1)} kcal**\n"
                f"• Carbohydrates: **{round(fd['carb_100g']*s, 1)} g**\n"
                f"• Protein: **{round(fd['protein_100g']*s, 1)} g**\n"
                f"• Fat: **{round(fd['fat_100g']*s, 1)} g**\n"
                f"• Fiber: **{round(fd['fiber_100g']*s, 1)} g**\n"
                f"• Glucose Impact: **{round(fd['glucose_impact']*s, 1)} g**\n"
                f"• GI: {gi_l} (GI score: {fd['gi_value']})\n"
                f"• Category: {fd['category'].title()} | "
                f"Region: {fd['region'].replace('_',' ').title()}"
                + (f"\n• Notes: _{fd['notes']}_" if fd.get('notes') else "")
                + d_note)

    def _dispatch_intent(self, intent: str, ml: str, dtype: str,
                          glucose: float, profile: dict, entities: dict) -> str | None:
        """Route to structured response handlers by intent."""

        if intent == 'greeting':
            fc  = self._count_table('foods')
            qc  = self._count_table('chatbot_qa')
            n   = profile.get('name', '')
            hi  = f"Welcome back, {n}! 👋" if n else "👋 **Vanakkam! DiabetesMeal AI v4.0**"
            return (f"{hi}\n\n"
                    f"🤖 ML ensemble: **{self.risk_model.accuracy['ensemble']}% accuracy** | "
                    f"{self.risk_model.ds['total']:,} patient records\n"
                    f"🍛 Database: **{fc} foods** | **{qc} Q&A** | South Indian specialist\n\n"
                    f"**Ask me:**\n"
                    f"• Food nutrition: 'ragi dosa', 'bitter gourd', 'drumstick sambar'\n"
                    f"• Meal calc: '3 idli + sambar + coconut chutney'\n"
                    f"• Diet guide: 'type 2 diet' | 'south Indian foods for diabetes'\n"
                    f"• Emergency: 'my glucose is 220 what to eat'\n"
                    f"• 'I am 45 years old BMI 28 glucose 145 am I diabetic?'")

        if intent == 'thanks':
            return "😊 Glad I could help! Ask me anything about diabetes nutrition or South Indian foods."

        if intent in ('type1_diet', 'type2_diet'):
            dt      = entities.get('diabetes_type') or ('type1' if intent == 'type1_diet' else 'type2')
            glc_val = float(profile.get('glucose_level') or 120)
            parts   = [self.meal_rec.get_plans(dt, t, glc_val, profile)
                       for t in ['breakfast', 'lunch', 'dinner']]
            lines   = [f"🩺 **{'Type 1' if dt=='type1' else 'Type 2'} Diabetes — Full Day Meal Plan:**\n"]
            for p in parts:
                for time_key, meals in p['meal_plan'].items():
                    icons = {"breakfast":"🌅","lunch":"☀️","dinner":"🌙"}
                    lines.append(f"{icons.get(time_key,'🍴')} **{time_key.title()}:**")
                    for m in meals:
                        lines.append(f"  • {m['meal_name']} — {m['reason']}")
            lines.append(f"\n🚫 **Avoid:** {', '.join(p['foods_to_avoid'][:6])}")
            return '\n'.join(lines)

        if intent in ('breakfast', 'lunch', 'dinner', 'snacks'):
            result = self.meal_rec.get_plans(dtype, intent, glucose, profile)
            plans  = result['meal_plan'].get(intent, [])
            icons  = {"breakfast":"🌅","lunch":"☀️","dinner":"🌙","snacks":"🍎"}
            lines  = [f"{icons.get(intent,'🍴')} **{intent.title()} Ideas — {'Type 1' if dtype=='type1' else 'Type 2'} Diabetes:**\n"]
            for m in plans[:4]:
                gi_icon = {"low":"🟢","medium":"🟡","high":"🔴"}.get(m['gi_rating'].lower(), "⚪")
                lines.append(f"• **{m['meal_name']}** {gi_icon}")
                lines.append(f"  Foods: {m['foods_display']}")
                lines.append(f"  ~{round(m['total_calories'])} kcal | {m['reason']}\n")
            avoid = result.get('foods_to_avoid', [])
            if avoid:
                lines.append(f"🚫 **Avoid:** {', '.join(avoid[:5])}")
            return '\n'.join(lines)

        if intent == 'low_gi':
            si    = self.meal_rec.get_food_by_gi('low', region='south_indian', limit=12)
            other = self.meal_rec.get_food_by_gi('low', limit=20)
            other = [f for f in other if f not in si][:8]
            return (f"✅ **Low GI Indian Foods (from database):**\n\n"
                    f"🌴 **South Indian:** {', '.join(si[:10])}\n\n"
                    f"🍛 **Pan-Indian:** {', '.join(other[:8])}\n\n"
                    f"⭐ **Diabetes superstars:** Bitter Gourd (GI 14), Keerai (GI 15), "
                    f"Drumstick (GI 20), Rasam (GI 30), Ragi Dosa (GI 44)")

        if intent == 'high_gi':
            foods = self._db_query("SELECT name, gi_value FROM foods WHERE gi='high' ORDER BY gi_value DESC LIMIT 12")
            names = ', '.join(r[0].replace('_', ' ').title() for r in foods)
            return (f"❌ **High GI Foods to Avoid:**\n\n{names}\n\n"
                    f"🚫 Especially: Jalebi (GI 90), Kesari (GI 80), "
                    f"Gulab Jamun (GI 86), White Rice (GI 72), Payasam (GI 75)")

        if intent == 'south_indian':
            si_low = self._db_query(
                "SELECT name FROM foods WHERE region='south_indian' AND gi='low' ORDER BY gi_value LIMIT 12")
            si_med = self._db_query(
                "SELECT name FROM foods WHERE region='south_indian' AND gi='medium' ORDER BY gi_value LIMIT 6")
            low_n  = ', '.join(r[0].replace('_', ' ').title() for r in si_low)
            med_n  = ', '.join(r[0].replace('_', ' ').title() for r in si_med)
            count  = self._count_where("foods", "region='south_indian'")
            return (f"🌴 **South Indian Foods for Diabetes ({count} in database):**\n\n"
                    f"✅ **Excellent (Low GI):** {low_n}\n\n"
                    f"⚠️ **Moderate — control portions:** {med_n}\n\n"
                    f"💡 **Top picks:** Ragi Dosa (GI 44), Bitter Gourd Fry (GI 22), "
                    f"Drumstick Sambar (GI 28), Pesarattu (GI 42), Keerai Masiyal (GI 20)")

        if intent == 'superfoods':
            rows = self._db_query(
                "SELECT name, notes FROM foods WHERE tags LIKE '%diabetes_superfood%' "
                "OR tags LIKE '%diabetic_friendly%' LIMIT 10")
            lines = '\n'.join(f"• **{r[0].replace('_',' ').title()}** — {r[1]}"
                               for r in rows if r[1])
            return f"⭐ **Top Diabetes Superfoods in Database:**\n\n{lines}"

        if intent == 'food_compare':
            if 'rice' in ml and 'roti' in ml:
                return self._compare_rice_roti()
            if 'idli' in ml and 'dosa' in ml:
                return self._compare_idli_dosa()
            return self._generic_compare(ml)

        if intent == 'dataset_stats':
            fc = self._count_table('foods')
            qc = self._count_table('chatbot_qa')
            pc = self._count_table('meal_plans')
            si = self._count_where("foods", "region='south_indian'")
            ds = self.risk_model.ds
            return (f"📊 **Database & Dataset Statistics:**\n\n"
                    f"• Patient records: **{ds['total']:,}** "
                    f"(main: {ds.get('main', '?')} + Pima: {ds.get('pima', '?')})\n"
                    f"• Diabetic patients: **{ds['diabetic']:,}** "
                    f"({round(ds['diabetic']/ds['total']*100, 1)}%)\n"
                    f"• Avg glucose (diabetic): **{ds['glc_d']} mg/dL** vs {ds['glc_nd']} non-diabetic\n"
                    f"• Foods in DB: **{fc}** | South Indian: **{si}**\n"
                    f"• Meal plans: **{pc}** | Q&A pairs: **{qc}**\n"
                    f"• ML accuracy: **{self.risk_model.accuracy['ensemble']}%**")

        if intent == 'model_accuracy':
            top5 = self.risk_model.get_top_features(5)
            fi   = '\n'.join(f"  {i+1}. {k}: {v}%" for i,(k,v) in enumerate(top5))
            acc  = self.risk_model.accuracy
            return (f"🤖 **ML Model Performance ({self.risk_model.ds['total']:,} records):**\n\n"
                    f"• Random Forest:       **{acc['random_forest']}%**\n"
                    f"• Gradient Boosting:   **{acc['gradient_boost']}%**\n"
                    f"• Logistic Regression: **{acc['logistic_reg']}%**\n"
                    f"• 🏆 Ensemble:         **{acc['ensemble']}%**\n\n"
                    f"**Top risk factors:**\n{fi}")

        return None   # no match — let caller handle fallback

    def _resp_fallback(self) -> str:
        return ("🤔 Try asking:\n"
                "• Food name: 'idli', 'ragi dosa', 'bitter gourd', 'drumstick sambar'\n"
                "• Meal calc: '3 idli + sambar + coconut chutney'\n"
                "• 'type 2 diet' | 'south Indian foods for diabetes'\n"
                "• 'breakfast ideas' | 'low GI foods'\n"
                "• Emergency: 'my glucose is 250 what to eat'\n"
                "• 'I am 50, BMI 30, HbA1c 7.2 — am I diabetic?'")

    # ── Food comparison helpers ────────────────────────────────────────────────
    def _compare_rice_roti(self) -> str:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rice  = dict(conn.execute("SELECT * FROM foods WHERE name='white_rice'").fetchone() or {})
        roti  = dict(conn.execute("SELECT * FROM foods WHERE name='roti'").fetchone() or {})
        brice = dict(conn.execute("SELECT * FROM foods WHERE name='brown_rice'").fetchone() or {})
        ragi  = dict(conn.execute("SELECT * FROM foods WHERE name='ragi_mudde'").fetchone() or {})
        conn.close()

        def v(d, k): return d.get(k, '?')
        return (f"⚖️ **Rice vs Roti for Diabetes (per 100g):**\n\n"
                f"| Nutrient | White Rice | Roti | Brown Rice | Ragi |\n"
                f"|----------|-----------|------|------------|------|\n"
                f"| Calories | {v(rice,'cal_100g')} | {v(roti,'cal_100g')} | {v(brice,'cal_100g')} | {v(ragi,'cal_100g')} |\n"
                f"| Carbs    | {v(rice,'carb_100g')}g | {v(roti,'carb_100g')}g | {v(brice,'carb_100g')}g | {v(ragi,'carb_100g')}g |\n"
                f"| Fiber    | {v(rice,'fiber_100g')}g | {v(roti,'fiber_100g')}g | {v(brice,'fiber_100g')}g | {v(ragi,'fiber_100g')}g |\n"
                f"| GI       | HIGH (72) | MED (52) | MED (50) | MED (55) |\n\n"
                f"🏆 **Best for diabetics:** Ragi > Brown Rice > Roti > White Rice\n"
                f"💡 Roti has more fiber and lower GI than white rice. Ragi is the healthiest grain.")

    def _compare_idli_dosa(self) -> str:
        conn  = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        idli  = dict(conn.execute("SELECT * FROM foods WHERE name='idli'").fetchone() or {})
        dosa  = dict(conn.execute("SELECT * FROM foods WHERE name='dosa'").fetchone() or {})
        rdosa = dict(conn.execute("SELECT * FROM foods WHERE name='ragi_dosa'").fetchone() or {})
        conn.close()

        def v(d, k): return d.get(k, '?')
        return (f"⚖️ **Idli vs Dosa (per piece / 80g):**\n\n"
                f"| | Idli (1 pc) | Dosa (1 med) | Ragi Dosa |\n"
                f"|---|---|---|---|\n"
                f"| Calories | {round(v(idli,'cal_100g')*0.4,0):.0f} | {round(v(dosa,'cal_100g')*0.8,0):.0f} | {round(v(rdosa,'cal_100g')*0.8,0):.0f} |\n"
                f"| Carbs    | {round(v(idli,'carb_100g')*0.4,1)}g | {round(v(dosa,'carb_100g')*0.8,1)}g | {round(v(rdosa,'carb_100g')*0.8,1)}g |\n"
                f"| GI       | MED (50) | MED (57) | LOW (44) |\n\n"
                f"🏆 **Best for diabetics:** Ragi Dosa > Idli > Plain Dosa\n"
                f"💡 Idli is steamed (no oil), fermented (lower GI). Ragi dosa is best — lowest GI of all.")

    def _generic_compare(self, ml: str) -> str:
        return ("⚖️ **Common Food Comparisons:**\n\n"
                "• 'rice vs roti' — ask me!\n"
                "• 'idli vs dosa' — ask me!\n"
                "• 'brown rice vs white rice' — ask me!\n\n"
                "Or type two food names with 'vs' between them.")

    # ── DB helpers ────────────────────────────────────────────────────────────
    def _db_query(self, sql: str, args: tuple = ()) -> list:
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(sql, args).fetchall()
        conn.close()
        return rows

    def _count_table(self, table: str) -> int:
        return self._db_query(f"SELECT COUNT(*) FROM {table}")[0][0]

    def _count_where(self, table: str, condition: str) -> int:
        return self._db_query(f"SELECT COUNT(*) FROM {table} WHERE {condition}")[0][0]
