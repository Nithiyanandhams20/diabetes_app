"""
nlp/intent_engine.py
====================
TF-IDF intent classifier for the diabetes chatbot.
Classifies user messages into one of 34 intent categories.

Architecture:
  - TF-IDF vectorizer (ngram 1-3, sublinear TF)
  - Cosine similarity against intent phrase corpus
  - Returns (intent_tag, confidence_score)
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ── Intent corpus: (tag, list of representative phrases) ─────────────────────
INTENT_CORPUS = [
    ("greeting",        [
        "hello hi hey namaste good morning vanakkam good evening",
        "who are you what can you do help me"
    ]),
    ("thanks",          [
        "thank you thanks great helpful nice awesome superb well done"
    ]),
    ("food_lookup",     [
        "nutrition calories carbs protein fat fiber gi glycemic of food",
        "what is in tell me about food nutritional value how much calories",
        "nutrition facts food info macros for"
    ]),
    ("meal_calc",       [
        "calculate my meal total calories how many calories i ate",
        "what is nutrition for my food total for 3 idli and sambar",
        "calculate nutrition 2 chapati dal bhindi"
    ]),
    ("glucose_mgmt",    [
        "manage blood sugar reduce glucose spike post meal control sugar",
        "how to lower blood sugar after eating glucose management"
    ]),
    ("high_glucose",    [
        "glucose is high blood sugar 200 250 300 high sugar emergency",
        "what to eat blood sugar very high glucose level high now"
    ]),
    ("low_glucose",     [
        "low blood sugar hypoglycemia dizzy shaking faint 70 mg",
        "glucose low feel weak what to eat immediately sugar dropped"
    ]),
    ("type1_diet",      [
        "type 1 diabetes diet type one food plan t1d insulin dependent",
        "what to eat type 1 diabetes meal plan type1"
    ]),
    ("type2_diet",      [
        "type 2 diabetes diet type two food plan t2d non insulin",
        "what to eat type 2 diabetes meal plan type2 sugar patient food"
    ]),
    ("type_compare",    [
        "difference between type 1 and type 2 diabetes compare",
        "type 1 vs type 2 which is worse how are they different"
    ]),
    ("low_gi",          [
        "low gi foods low glycemic index safe foods good foods recommended",
        "what foods are good for diabetics what can diabetic eat safely"
    ]),
    ("high_gi",         [
        "foods to avoid high gi dangerous bad foods avoid list",
        "worst foods for diabetes what not to eat diabetic avoid"
    ]),
    ("gi_explain",      [
        "what is glycemic index explain gi glycaemic index meaning",
        "how does gi work what does gi mean glycemic index explained"
    ]),
    ("south_indian",    [
        "south indian food diabetes tamil food kerala andhra food traditional",
        "south indian diet diabetic ragi idli dosa sambar pongal"
    ]),
    ("tamil_foods",     [
        "tamil food diabetes tamil nadu food for sugar patient traditional tamil",
        "kollu kanji ragi koozh kerala food andhra food for diabetes"
    ]),
    ("superfoods",      [
        "superfoods diabetes best foods ragi bitter gourd fenugreek drumstick",
        "jamun amla cinnamon turmeric moringa karela pavakkai diabetes"
    ]),
    ("hba1c",           [
        "hba1c level a1c hemoglobin a1c long term blood sugar what is hba1c",
        "hba1c normal range how to lower hba1c reduce a1c"
    ]),
    ("bmi",             [
        "bmi obesity overweight body mass index weight diabetes",
        "how does weight affect diabetes bmi and blood sugar"
    ]),
    ("breakfast",       [
        "breakfast ideas morning meal diabetic what to eat for breakfast",
        "morning food for diabetes healthy breakfast suggestions"
    ]),
    ("lunch",           [
        "lunch ideas midday meal afternoon food lunch suggestions diabetic",
        "what to eat for lunch diabetes best lunch"
    ]),
    ("dinner",          [
        "dinner ideas evening meal night food dinner suggestions diabetic",
        "what to eat for dinner diabetes best dinner night meal"
    ]),
    ("snacks",          [
        "snack ideas between meals healthy snacks evening mid day snack",
        "what to eat between meals diabetic snacks healthy"
    ]),
    ("exercise",        [
        "exercise diabetes walk yoga workout physical activity gym",
        "how exercise helps blood sugar walking after meals"
    ]),
    ("insulin",         [
        "insulin dose timing injection carb ratio units how much insulin",
        "insulin and food when to take insulin before meals"
    ]),
    ("hypertension",    [
        "blood pressure hypertension bp high bp diet sodium salt",
        "diabetes and high blood pressure food diet"
    ]),
    ("insulin_resist",  [
        "insulin resistance what is it how to reduce insulin sensitivity",
        "cells not responding insulin cause type 2 insulin resistant"
    ]),
    ("food_compare",    [
        "compare rice vs roti which is better chapati or rice",
        "idli vs dosa which healthier brown rice vs white rice for diabetes"
    ]),
    ("weight_loss",     [
        "lose weight diabetes diet obesity weight loss for diabetic",
        "how to reduce weight with diabetes bmi diabetes"
    ]),
    ("fruits",          [
        "fruits for diabetes can i eat fruits which fruits are safe",
        "fruit and blood sugar diabetic fruits to eat avoid mango banana"
    ]),
    ("complications",   [
        "diabetes complications kidney damage eye problems nerve damage neuropathy",
        "retinopathy nephropathy what happens if diabetes untreated"
    ]),
    ("medication",      [
        "metformin medication medicine tablet for diabetes treatment drug",
        "what medicines used diabetes can i stop medication"
    ]),
    ("water",           [
        "how much water drink diabetes hydration fluid intake daily",
        "water intake for diabetic patients dehydration diabetes"
    ]),
    ("stress",          [
        "stress blood sugar anxiety cortisol emotional eating diabetes",
        "how does stress affect blood sugar cortisol glucose"
    ]),
    ("dataset_stats",   [
        "dataset statistics patient records how many data records",
        "how many patients data size training data records"
    ]),
    ("model_accuracy",  [
        "model accuracy machine learning how accurate prediction quality",
        "how good is the ml model rf accuracy percent"
    ]),
    ("self_assess",     [
        "am i diabetic check my risk years old bmi glucose hba1c",
        "is my blood sugar dangerous should i worry about my levels"
    ]),
    ("nutrition_calc",  [
        "nutrition calculator calculate food intake how to use calculator",
        "how to calculate my meal nutrition"
    ]),
]


class IntentEngine:
    """
    TF-IDF intent classifier.

    Usage:
        engine = IntentEngine()
        intent, confidence = engine.classify("is idli good for diabetes?")
    """

    def __init__(self):
        self._phrases = []
        self._labels  = []
        self._tfidf   = None
        self._matrix  = None
        self._build()

    # ── public ────────────────────────────────────────────────────────────────
    def classify(self, text: str) -> tuple:
        """
        Classify text into an intent.

        Returns:
            (intent_tag: str, confidence: float 0-1)
        """
        vec  = self._tfidf.transform([text.lower()])
        sims = cosine_similarity(vec, self._matrix).flatten()
        idx  = int(np.argmax(sims))
        return self._labels[idx], float(sims[idx])

    def classify_top(self, text: str, n: int = 3) -> list:
        """Return top-n (intent, confidence) pairs."""
        vec  = self._tfidf.transform([text.lower()])
        sims = cosine_similarity(vec, self._matrix).flatten()
        idxs = np.argsort(sims)[::-1][:n]
        return [(self._labels[i], float(sims[i])) for i in idxs]

    def all_intents(self) -> list:
        """Return list of all registered intent tags."""
        return list(dict.fromkeys(self._labels))  # unique, preserving order

    # ── private ───────────────────────────────────────────────────────────────
    def _build(self):
        """Build TF-IDF matrix from corpus."""
        for tag, phrases in INTENT_CORPUS:
            for ph in phrases:
                for sentence in ph.split('\n'):
                    s = sentence.strip()
                    if s:
                        self._phrases.append(s)
                        self._labels.append(tag)

        self._tfidf  = TfidfVectorizer(ngram_range=(1, 3), sublinear_tf=True, analyzer='word')
        self._matrix = self._tfidf.fit_transform(self._phrases)
        print(f"   ✅ IntentEngine: {len(self.all_intents())} intents, "
              f"{len(self._phrases)} training phrases")
