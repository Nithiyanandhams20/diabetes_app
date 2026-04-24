"""
DiabetesMeal AI v4.0 — Production Flask Application
=====================================================
✅ ML Ensemble Risk Model (RF + GB + LR, 97% accuracy, 100,768 records)
✅ RAG Chatbot (TF-IDF over SQLite Q&A + cosine similarity + entity extraction)
✅ SQLite Dynamic Database (96 foods, 23 meal plans, 112+ Q&A)
✅ Personalized Meal Plans (glucose-aware, region-aware, diabetes-type-aware)
✅ Nutrition Calculator (natural language parsing)
✅ Image-based food recognition (color + texture analysis)
✅ User profiles + food logging
✅ South Indian food specialist (38+ South Indian foods)
✅ No API key required — fully self-contained
"""

from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import hashlib
import pandas as pd
import numpy as np
import sqlite3
import json
import base64
import io
import re
import os
import secrets
from PIL import Image
from difflib import SequenceMatcher

# ML imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# ── Module imports from our custom folders ────────────────────────────────────
from nlp.intent_engine  import IntentEngine
from nlp.entity_extractor import EntityExtractor
from nlp.response_builder import ResponseBuilder
from models.risk_model   import RiskModel
from models.image_model  import ImageModel
from models.meal_recommender import MealRecommender

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, 'data')
DB_PATH   = os.path.join(BASE_DIR, 'database', 'diabetes_ai.db')

# ══════════════════════════════════════════════════════════════
# DATABASE HELPERS
# ══════════════════════════════════════════════════════════════
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def query_db(sql, args=(), one=False):
    conn = get_db()
    cur  = conn.execute(sql, args)
    rv   = cur.fetchall()
    conn.close()
    return (rv[0] if rv else None) if one else rv

def exec_db(sql, args=()):
    conn = get_db()
    conn.execute(sql, args)
    conn.commit()
    conn.close()

def hash_password(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

def init_patients_table():
    conn = get_db()
    conn.execute('''CREATE TABLE IF NOT EXISTS patients (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        name       TEXT NOT NULL,
        email      TEXT UNIQUE NOT NULL,
        phone      TEXT,
        dob        TEXT,
        gender     TEXT,
        password   TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    conn.commit()
    conn.close()

init_patients_table()

# ══════════════════════════════════════════════════════════════
# INITIALISE ALL MODULES
# ══════════════════════════════════════════════════════════════
print("🚀 Initialising DiabetesMeal AI v4.0…")

# ML Risk Model
risk_model = RiskModel(data_dir=DATA_DIR)
risk_model.train()

# Image Model
image_model = ImageModel(db_path=DB_PATH)

# Meal Recommender
meal_rec = MealRecommender(db_path=DB_PATH)

# NLP Engine
entity_extractor = EntityExtractor(db_path=DB_PATH)
intent_engine    = IntentEngine()
response_builder = ResponseBuilder(
    db_path      = DB_PATH,
    risk_model   = risk_model,
    meal_rec     = meal_rec,
    entity_ext   = entity_extractor,
)

print("✅ All modules loaded")

# ══════════════════════════════════════════════════════════════
# SESSION HELPER
# ══════════════════════════════════════════════════════════════
def get_session_id():
    if 'sid' not in session:
        session['sid'] = secrets.token_hex(8)
    return session['sid']

# ══════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════
@app.route('/login')
def login_page():
    if session.get('patient_id'):
        return redirect(url_for('index'))
    return render_template('login.html')

@app.route('/auth/register', methods=['POST'])
def auth_register():
    d = request.json or {}
    name     = (d.get('name') or '').strip()
    email    = (d.get('email') or '').strip().lower()
    phone    = (d.get('phone') or '').strip()
    dob      = (d.get('dob') or '').strip()
    gender   = d.get('gender', 'female')
    password = d.get('password', '')
    if not name or not email or not password:
        return jsonify({'success': False, 'error': 'Name, email and password are required.'})
    if len(password) < 6:
        return jsonify({'success': False, 'error': 'Password must be at least 6 characters.'})
    try:
        exec_db(
            'INSERT INTO patients (name,email,phone,dob,gender,password) VALUES (?,?,?,?,?,?)',
            (name, email, phone or None, dob or None, gender, hash_password(password))
        )
    except Exception:
        return jsonify({'success': False, 'error': 'Email already registered. Please sign in.'})
    patient = query_db('SELECT * FROM patients WHERE email=?', (email,), one=True)
    session['patient_id']   = patient['id']
    session['patient_name'] = patient['name']
    session['patient_email']= patient['email']
    return jsonify({'success': True})

@app.route('/auth/login', methods=['POST'])
def auth_login():
    d = request.json or {}
    email    = (d.get('email') or '').strip().lower()
    password = d.get('password', '')
    patient = query_db('SELECT * FROM patients WHERE email=? AND password=?',
                       (email, hash_password(password)), one=True)
    if not patient:
        return jsonify({'success': False, 'error': 'Invalid email or password.'})
    session['patient_id']   = patient['id']
    session['patient_name'] = patient['name']
    session['patient_email']= patient['email']
    return jsonify({'success': True})

@app.route('/auth/logout')
def auth_logout():
    session.clear()
    return redirect(url_for('login_page'))

@app.route('/auth/me')
def auth_me():
    if not session.get('patient_id'):
        return jsonify({'logged_in': False})
    return jsonify({
        'logged_in': True,
        'name':  session.get('patient_name', ''),
        'email': session.get('patient_email', ''),
    })

@app.route('/')
def index():
    if not session.get('patient_id'):
        return redirect(url_for('login_page'))
    foods = query_db("SELECT name FROM foods ORDER BY name")
    food_names = json.dumps([r['name'] for r in foods])
    return render_template('index.html', indian_foods=food_names,
                           patient_name=session.get('patient_name',''),
                           patient_email=session.get('patient_email',''))


@app.route('/predict_diabetes', methods=['POST'])
def predict_diabetes():
    d   = request.json
    sid = get_session_id()

    features = {
        'age':          float(d.get('age', 0)),
        'bmi':          float(d.get('bmi', 0)),
        'glucose':      float(d.get('glucose', 0)),
        'hba1c':        float(d.get('hba1c', 0)),
        'hypertension': int(d.get('hypertension', 0)),
        'heart_disease':int(d.get('heart_disease', 0)),
        'smoking':      d.get('smoking', 'never'),
        'gender':       d.get('gender', 'female'),
    }

    result = risk_model.predict(features)

    # Save profile
    exec_db("""INSERT OR REPLACE INTO user_profiles
               (session_id, age, bmi, glucose_level, hba1c, hypertension,
                heart_disease, diabetes_type, updated_at)
               VALUES (?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP)""",
            (sid, features['age'], features['bmi'], features['glucose'],
             features['hba1c'], features['hypertension'], features['heart_disease'],
             result['diabetes_type']))

    return jsonify(result)


@app.route('/get_meal_plan', methods=['POST'])
def get_meal_plan():
    d     = request.json
    dtype = d.get('diabetes_type', 'type2')
    mtime = d.get('meal_time', 'all')
    glc   = float(d.get('glucose_level', 120))

    # ✅ Load JSON dataset
    with open('datasets/meal_plans.json') as f:
        data = json.load(f)

    plans = data["plans"]

    # ✅ Structure response (VERY IMPORTANT for frontend)
    result = {
        "diabetes_type": dtype,
        "glucose_range": "normal",
        "meal_plan": {
            "breakfast": [],
            "lunch": [],
            "dinner": [],
            "snacks": []
        }
    }

    # ✅ Filter plans
    for p in plans:
        if p["diabetes_type"] == dtype:

            if mtime == "all" or p["meal_time"] == mtime:

                meal = {
                    "meal_name": p["meal_name"],
                    "gi_rating": p["gi_rating"],
                    "reason": p["reason"],
                    "total_calories": p["total_cal"],
                    "foods_display": p.get("foods", ""),
                    "gi_icon": "🟢" if p["gi_rating"] == "low" else "🟡" if p["gi_rating"] == "medium" else "🔴"
                }

                result["meal_plan"][p["meal_time"]].append(meal)

    return jsonify(result)


@app.route('/analyze_food', methods=['POST'])
def analyze_food():
    d     = request.json
    sid   = get_session_id()
    fname = d.get('food_name', '').lower().replace(' ', '_')
    port  = float(d.get('portion_g', 100))
    dtype = d.get('diabetes_type', 'type2')

    food_key = entity_extractor.find_food(fname)
    if not food_key:
        return jsonify({"found": False, "food": fname.replace('_', ' ').title(),
                        "message": "Food not in database. Try the autocomplete list."})

    fd = query_db("SELECT * FROM foods WHERE name=?", (food_key,), one=True)
    if not fd:
        return jsonify({"found": False})

    fd   = dict(fd)
    port = min(port, 500)   # 🔥 limit max portion
    s    = port / 100
    gi   = fd['gi']
    suit = fd['suitable_t2'] if dtype == 'type2' else fd['suitable_t1']

    calories = fd['cal_100g'] * s
    fat      = fd['fat_100g'] * s
    carbs    = fd['carb_100g'] * s

    # 🔥 SMART DIABETIC LOGIC
    if suit == 0:
        s_text, s_color = "❌ Not Suitable", "#ef4444"
        advice = "This food is not suitable for your diabetes type."

    elif gi == 'high':
        s_text, s_color = "❌ Avoid", "#ef4444"
        advice = "High GI food — causes rapid blood sugar spike."

    elif calories > 500 or fat > 25:
        s_text, s_color = "⚠️ Not Recommended", "#ef4444"
        advice = "High calories or fat — may increase insulin resistance."

    elif carbs > 50:
        s_text, s_color = "⚠️ Moderate", "#f97316"
        advice = "High carbohydrates — control portion size."

    else:
        s_text, s_color = "✅ Recommended", "#22c55e"
        advice = "Healthy low GI food in controlled portion."

    # Log
    exec_db("""INSERT INTO food_logs
               (session_id, food_name, grams, calories, carbs, glucose_imp, meal_time)
               VALUES (?,?,?,?,?,?,?)""",
            (sid, food_key, port, round(fd['cal_100g']*s, 1),
             round(fd['carb_100g']*s, 1), round(fd['glucose_impact']*s, 1), 'unspecified'))

    return jsonify({
        "found": True,
        "food": fd['name'].replace('_', ' ').title(),
        "name_local": fd.get('name_local', ''),
        "category": fd['category'],
        "region": fd['region'],
        "portion_g": port,
        "calories":      round(fd['cal_100g'] * s, 1),
        "glucose_impact":round(fd['glucose_impact'] * s, 1),
        "carbs":         round(fd['carb_100g'] * s, 1),
        "protein":       round(fd['protein_100g'] * s, 1),
        "fat":           round(fd['fat_100g'] * s, 1),
        "fiber":         round(fd['fiber_100g'] * s, 1),
        "gi": gi, "gi_value": fd['gi_value'],
        "suitability": s_text, "suitability_color": s_color,
        "advice": advice,
        "notes": fd.get('notes', ''),
    })


@app.route('/analyze_image', methods=['POST'])
def analyze_image():
    try:
        d         = request.json
        img_b64   = d.get('image_data', '')
        dtype     = d.get('diabetes_type', 'type2')
        img_bytes = base64.b64decode(img_b64)

        result = image_model.predict(img_bytes)

        fd = query_db("SELECT * FROM foods WHERE name=?", (result['food_key'],), one=True)
        if fd:
            fd = dict(fd)
            gi = fd['gi']
            result['nutrition'] = {
                'calories':       fd['cal_100g'],
                'carbs':          fd['carb_100g'],
                'protein':        fd['protein_100g'],
                'fat':            fd['fat_100g'],
                'fiber':          fd['fiber_100g'],
                'glucose_impact': fd['glucose_impact'],
                'gi':             gi,
                'gi_value':       fd['gi_value'],
            }
            if gi == 'high':
                result['advice'] = "❌ High GI — avoid or very small portions for diabetics."
            elif gi == 'medium':
                result['advice'] = "⚠️ Medium GI — eat in moderation with vegetables and protein."
            else:
                result['advice'] = "✅ Low GI — safe for diabetes management."

        result['success'] = True
        return jsonify(result)

    except Exception as e:
        return jsonify({"success": False, "error": str(e),
                        "detected_food": "Unknown", "confidence": "low",
                        "nutrition": {"calories": 200, "glucose_impact": 35,
                                      "carbs": 25, "gi": "medium"},
                        "advice": "Could not detect food. Use the Food Analyzer tab."})


@app.route('/chat', methods=['POST'])
def chat():
    d       = request.json
    message = d.get('message', '').strip()
    history = d.get('history', [])
    if not message:
        return jsonify({"error": "Empty message"}), 400

    sid     = get_session_id()
    profile = {}
    row     = query_db("SELECT * FROM user_profiles WHERE session_id=?", (sid,), one=True)
    if row:
        profile = dict(row)

    # NLP pipeline
    entities = entity_extractor.extract(message)
    intent, confidence = intent_engine.classify(message)
    reply    = response_builder.build(message, intent, entities, history, profile)

    return jsonify({"reply": reply, "intent": intent, "confidence": round(confidence, 3)})


@app.route('/chat_structured', methods=['POST'])
def chat_structured():
    """
    Structured AI Nutrition Assistant.
    Rules: only food/diabetes topics, always returns structured JSON.
    """
    import re as _re

    d       = request.json or {}
    message = d.get('message', '').strip()
    history = d.get('history', [])
    if not message:
        return jsonify({"error": "Empty message"}), 400

    FOOD_SCOPE = _re.compile(
        r'food|eat|diet|meal|nutrition|glucose|sugar|blood sugar|diabet|'
        r'glycem|gi level|glycaemic|calori|carb|protein|fat|fiber|'
        r'idli|dosa|rice|ragi|sambar|rasam|dal|roti|chapati|biryani|'
        r'breakfast|lunch|dinner|snack|south indian|indian food|tamil|kerala|'
        r'hba1c|a1c|insulin|type 1|type 2|bmi|weight|obes|'
        r'bitter gourd|drumstick|fenugreek|methi|amla|jamun|turmeric|'
        r'mango|banana|apple|fruit|vegetable|lentil|legume|oats|millet|'
        r'portion|serving|avoid|safe|recommend|manage|control|spike|'
        r'complication|kidney|eye|nerve|medication|metformin',
        _re.IGNORECASE
    )

    if not FOOD_SCOPE.search(message):
        return jsonify({"out_of_scope": True, "structured": False,
                        "message": "I can only help with diabetes, nutrition, and food-related questions."})

    sid     = get_session_id()
    profile = {}
    row     = query_db("SELECT * FROM user_profiles WHERE session_id=?", (sid,), one=True)
    if row:
        profile = dict(row)

    food_key = entity_extractor.find_food(message)
    fd = None
    if food_key:
        fd = query_db("SELECT * FROM foods WHERE name=?", (food_key,), one=True)
        if fd:
            fd = dict(fd)

    if fd:
        gi     = fd.get('gi', 'na')
        gi_val = fd.get('gi_value')
        dtype  = profile.get('diabetes_type', 'type2')
        suit   = fd.get('suitable_t2', 1) if dtype == 'type2' else fd.get('suitable_t1', 1)
        rec    = ('avoid' if (gi == 'high' or suit == 0)
                  else 'moderate' if gi == 'medium'
                  else 'safe' if gi == 'low' else 'info')

        name_d = fd['name'].replace('_', ' ').title()
        local  = f" ({fd['name_local']})" if fd.get('name_local') else ""
        notes  = fd.get('notes') or ''
        answer = (f"{name_d} has {fd['cal_100g']} kcal per 100g, "
                  f"{fd['carb_100g']}g carbs, {fd['protein_100g']}g protein, "
                  f"{fd['fiber_100g']}g fiber. "
                  + (notes if notes else
                     f"It has a {gi} GI (score {gi_val or '?'}) and is "
                     + ("safe for diabetics in controlled portions." if rec == 'safe'
                        else "best eaten in moderation with protein and vegetables." if rec == 'moderate'
                        else "not recommended — causes rapid blood sugar spikes.")))

        gi_effect = {"low":    "Raises blood sugar slowly — generally safe.",
                     "medium": "Moderate rise — control portions carefully.",
                     "high":   "Rapid glucose spike — avoid or strictly limit."}.get(gi, "")

        portion = (f"Typical serving: ~{fd.get('serving_g',100)}g. "
                   + ("Freely consumed regularly." if rec == 'safe'
                      else "Limit to 1 small serving; pair with dal or vegetables." if rec == 'moderate'
                      else "Avoid completely or take only a tiny portion rarely."))

        region = fd.get('region','')
        tip = (f"{name_d} is a "
               f"{'South Indian' if 'south' in region else 'pan-Indian'} food. "
               + ("Include it regularly in your meal plan." if rec == 'safe'
                  else "Pair with low-GI foods to balance overall meal GI." if rec == 'moderate'
                  else "Try ragi dosa or pesarattu as a diabetes-friendly alternative."))

        return jsonify({
            "structured": True, "out_of_scope": False,
            "topic": name_d + local, "answer": answer,
            "gi": gi, "gi_value": gi_val, "gi_effect": gi_effect,
            "rec": rec, "portion": portion, "tip": tip,
            "intent": "food_lookup", "food_key": food_key,
        })

    entities = entity_extractor.extract(message)
    intent, confidence = intent_engine.classify(message)

    TOPIC_RESP = {
        'gi_explain': {"topic":"Glycemic index (GI)",
            "answer":"Glycemic Index ranks foods 0-100 by how quickly they raise blood sugar. Low GI (55 or below): slow, safe release. Medium GI (56-69): moderate rise. High GI (70+): rapid spike — most harmful for diabetics.",
            "gi":"na","gi_value":None,"rec":"info",
            "portion":"Build every meal around low-GI foods. Vegetables, lentils, and millets are mostly low GI.",
            "tip":"Combining a high-GI food with protein, fat, or fiber significantly lowers the overall meal GI."},
        'hba1c': {"topic":"HbA1c",
            "answer":"HbA1c measures average blood glucose over 3 months. Normal: below 5.7%. Pre-diabetic: 5.7-6.4%. Diabetic: 6.5%+. Target for managed diabetics: below 7%.",
            "gi":"na","gi_value":None,"rec":"info",
            "portion":"Test HbA1c every 3 months. Consistent low-GI diet + exercise can reduce it 0.5-1% in 3 months.",
            "tip":"Replacing white rice with ragi for 3 months alone measurably reduces HbA1c levels."},
        'type2_diet': {"topic":"Type 2 diabetes diet",
            "answer":"Type 2 diabetes is largely diet-driven and can be significantly improved through food choices. Reduce total carbohydrate load, choose low-GI foods, and increase protein and fiber at every meal.",
            "gi":"na","gi_value":None,"rec":"info",
            "portion":"Half plate non-starchy vegetables, quarter plate protein (dal/egg/fish), quarter plate low-GI grain (ragi/brown rice).",
            "tip":"Losing just 5-10% body weight has a dramatic effect on blood sugar control in Type 2 diabetes."},
        'type1_diet': {"topic":"Type 1 diabetes diet",
            "answer":"Type 1 requires careful carbohydrate counting and matching insulin doses to carb intake. Consistency matters more than GI alone — eating similar carb amounts daily makes dosing predictable.",
            "gi":"na","gi_value":None,"rec":"info",
            "portion":"Target 45-60g carbs per main meal. 1 chapati=20g, 1 cup rice=45g, 1 cup dal=15g, 3 idlis=24g.",
            "tip":"Always carry fast-acting glucose (glucose tablets or juice) in case of hypoglycemia."},
        'breakfast': {"topic":"Best breakfast for diabetics",
            "answer":"The ideal diabetic breakfast is high in protein and fiber, low in refined carbs, and has a low glycemic index. It prevents mid-morning glucose spikes and provides steady energy.",
            "gi":"low","gi_value":None,"rec":"safe",
            "portion":"Best: ragi dosa + sambar, oats upma with vegetables, pesarattu + chutney, 2-3 idlis + sambar.",
            "tip":"Eating protein (egg/dal/paneer) before carbs in the morning significantly reduces the breakfast glucose spike."},
        'south_indian': {"topic":"South Indian foods for diabetes",
            "answer":"South Indian cuisine is naturally well-suited for diabetics when chosen carefully. Sambar, rasam, keerai, ragi, bitter gourd, and drumstick are excellent. Choose steamed or boiled over fried.",
            "gi":"low","gi_value":None,"rec":"safe",
            "portion":"Best daily: ragi dosa, idli + sambar, rasam, any keerai dish, drumstick sambar, pesarattu, mor kuzhambu.",
            "tip":"Traditional Tamil ragi koozh and kollu rasam were designed for blood sugar management centuries before modern nutrition science."},
        'high_glucose': {"topic":"High blood glucose — what to eat",
            "answer":"When glucose exceeds 180 mg/dL, avoid all high-carb foods. Drink 2-3 glasses of water, walk 15-20 minutes, and choose very-low-GI foods until levels stabilise.",
            "gi":"low","gi_value":None,"rec":"info",
            "portion":"Safe now: bitter gourd, cucumber, rasam, plain curd, boiled egg, leafy greens. Avoid: rice, roti, bread, sweets, fruit.",
            "tip":"If glucose exceeds 300 mg/dL or you feel unwell, contact your doctor immediately."},
        'low_glucose': {"topic":"Low blood glucose (hypoglycemia)",
            "answer":"Hypoglycemia (below 70 mg/dL) is an emergency. Apply the 15-15 rule: eat 15g fast carbs, wait 15 minutes, recheck. Repeat if still low. Once stable, eat a mixed snack.",
            "gi":"high","gi_value":None,"rec":"info",
            "portion":"Fast carbs: 4-5 glucose tablets, OR half cup fruit juice, OR 3 tsp sugar in water, OR 2 tbsp raisins.",
            "tip":"Never use diet drinks or sugar-free juice for hypoglycemia — they have no real sugar and will not work."},
        'glucose_mgmt': {"topic":"Managing blood sugar after meals",
            "answer":"Post-meal spikes can be controlled through food order and activity. Eat vegetables first, then protein, then carbs. Pair all carbs with fiber and protein. Walk 15 minutes after eating.",
            "gi":"na","gi_value":None,"rec":"info",
            "portion":"Target: below 140 mg/dL at 2 hours after meals. Replace white rice with ragi. Add sambar/dal to every meal.",
            "tip":"A 15-minute post-meal walk drops blood sugar 20-40 mg/dL — one of the most effective interventions available."},
        'superfoods': {"topic":"Indian diabetes superfoods",
            "answer":"Several Indian foods have clinically proven blood sugar-lowering properties. Bitter gourd contains charantin (acts like insulin). Fenugreek seeds slow glucose absorption. Ragi polyphenols slow starch digestion. Jamun and amla are rich in antioxidants.",
            "gi":"low","gi_value":None,"rec":"safe",
            "portion":"Daily habits: 1 tsp fenugreek water (morning), bitter gourd fry 3x/week, ragi as grain, amla 2x/week.",
            "tip":"These work as consistent daily habits, not occasional additions. 4-6 weeks shows measurable glucose improvement."},
    }

    if intent in TOPIC_RESP:
        resp_data = TOPIC_RESP[intent].copy()
        resp_data.update({"structured": True, "out_of_scope": False, "intent": intent})
        return jsonify(resp_data)

    try:
        reply = response_builder.build(message, intent, entities, history, profile)
        return jsonify({"structured": False, "out_of_scope": False,
                        "reply": reply, "intent": intent})
    except Exception as e:
        return jsonify({"structured": False, "out_of_scope": False,
                        "reply": "I am not fully sure. Please try asking about a specific food or diabetes topic."})


@app.route('/nutrition_calc', methods=['POST'])
def nutrition_calc():
    d     = request.json
    text  = d.get('meal_text', '')
    items = entity_extractor.parse_meal(text)
    if not items:
        return jsonify({"error": "No recognisable foods found. Try: '3 idli + sambar + coconut chutney'"}), 400

    totals, breakdown = meal_rec.calculate_nutrition(items)
    return jsonify({"totals": totals, "breakdown": breakdown, "items_found": len(items)})


@app.route('/save_profile', methods=['POST'])
def save_profile():
    d   = request.json
    sid = get_session_id()
    bmi = None
    if d.get('weight_kg') and d.get('height_cm'):
        h   = float(d['height_cm']) / 100
        bmi = round(float(d['weight_kg']) / (h * h), 1)

    exec_db("""INSERT OR REPLACE INTO user_profiles
               (session_id, name, age, gender, weight_kg, height_cm, bmi,
                diabetes_type, glucose_level, hba1c, hypertension, heart_disease,
                activity_level, dietary_pref, region, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP)""",
            (sid, d.get('name'), d.get('age'), d.get('gender', 'female'),
             d.get('weight_kg'), d.get('height_cm'), bmi or d.get('bmi'),
             d.get('diabetes_type', 'type2'), d.get('glucose_level'),
             d.get('hba1c'), d.get('hypertension', 0), d.get('heart_disease', 0),
             d.get('activity_level', 'moderate'), d.get('dietary_pref', 'vegetarian'),
             d.get('region', 'south_indian')))

    return jsonify({"success": True, "bmi": bmi, "session_id": sid})


@app.route('/get_profile')
def get_profile():
    sid = get_session_id()
    row = query_db("SELECT * FROM user_profiles WHERE session_id=?", (sid,), one=True)
    return jsonify(dict(row) if row else {})


@app.route('/food_log')
def food_log():
    sid  = get_session_id()
    rows = query_db("""SELECT food_name, grams, calories, carbs, glucose_imp, logged_at
                       FROM food_logs WHERE session_id=?
                       ORDER BY logged_at DESC LIMIT 20""", (sid,))
    logs = [dict(r) for r in rows]
    return jsonify({
        "logs":                  logs,
        "total_calories":        round(sum(r['calories']    for r in logs), 1),
        "total_carbs":           round(sum(r['carbs']       for r in logs), 1),
        "total_glucose_impact":  round(sum(r['glucose_imp'] for r in logs), 1),
    })


@app.route('/search_foods')
def search_foods():
    q   = request.args.get('q', '').lower()
    gi  = request.args.get('gi', '')
    reg = request.args.get('region', '')
    sql  = "SELECT name,name_local,category,region,gi,gi_value,cal_100g,suitable_t2 FROM foods WHERE 1=1"
    args = []
    if q:
        sql += " AND (name LIKE ? OR name_local LIKE ?)"; args += [f'%{q}%', f'%{q}%']
    if gi:
        sql += " AND gi=?"; args.append(gi)
    if reg:
        sql += " AND region LIKE ?"; args.append(f'%{reg}%')
    sql += " ORDER BY gi_value ASC LIMIT 30"
    return jsonify([dict(r) for r in query_db(sql, args)])


@app.route('/dataset_stats')
def dataset_stats():
    fc = query_db("SELECT COUNT(*) n FROM foods", one=True)['n']
    qc = query_db("SELECT COUNT(*) n FROM chatbot_qa", one=True)['n']
    pc = query_db("SELECT COUNT(*) n FROM meal_plans", one=True)['n']
    si = query_db("SELECT COUNT(*) n FROM foods WHERE region='south_indian'", one=True)['n']
    lg = query_db("SELECT COUNT(*) n FROM foods WHERE gi='low'", one=True)['n']
    hg = query_db("SELECT COUNT(*) n FROM foods WHERE gi='high'", one=True)['n']
    return jsonify({
        "total_records":       risk_model.ds['total'],
        "diabetic_count":      risk_model.ds['diabetic'],
        "avg_glucose_diabetic":risk_model.ds['glc_d'],
        "avg_bmi_diabetic":    risk_model.ds['bmi_d'],
        "indian_foods_count":  fc,
        "south_indian_foods":  si,
        "low_gi_foods":        lg,
        "high_gi_foods":       hg,
        "meal_plans":          pc,
        "chatbot_qa":          qc,
        "model_accuracy":      risk_model.accuracy,
    })


if __name__ == '__main__':
    import os

    # Auto-build DB if missing
    if not os.path.exists(DB_PATH):
        print("🔨 Building database for first time…")
        from database.build_db import build
        build()

    fc = query_db("SELECT COUNT(*) n FROM foods", one=True)['n']
    qc = query_db("SELECT COUNT(*) n FROM chatbot_qa", one=True)['n']

    print(f"\n🩺 DiabetesMeal AI v4.0 — Ready!")
    print(f"   📊 Patients : {risk_model.ds['total']:,}")
    print(f"   🤖 Accuracy : {risk_model.accuracy['ensemble']}%")
    print(f"   🍛 Foods    : {fc}  |  💬 Q&A: {qc}")

    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
