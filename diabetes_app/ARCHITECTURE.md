# DiabetesMeal AI v4.0 — Complete System Architecture
## Final Year Project + Production Deployment Guide

---

## 1. SYSTEM ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE LAYER                          │
│  ┌──────────────┐  ┌─────────────┐  ┌────────────┐  ┌──────────┐  │
│  │ Risk Assess  │  │ Meal Plans  │  │  Food Scan │  │   Chat   │  │
│  │  (Form UI)   │  │  (Dynamic)  │  │  (Upload)  │  │  (NLP)   │  │
│  └──────┬───────┘  └──────┬──────┘  └─────┬──────┘  └────┬─────┘  │
└─────────┼────────────────┼───────────────┼───────────────┼─────────┘
          │                │               │               │
┌─────────▼────────────────▼───────────────▼───────────────▼─────────┐
│                         FLASK API LAYER (Python)                     │
│  POST /predict_diabetes  │  GET /get_meal_plan  │  POST /chat       │
│  POST /analyze_food      │  POST /analyze_image │  POST /nutrition_calc│
│  GET /search_foods       │  POST /save_profile  │  GET /food_log    │
└──────────────┬──────────────────┬──────────────────┬────────────────┘
               │                  │                  │
    ┌──────────▼──────┐  ┌────────▼──────┐  ┌───────▼────────────┐
    │  ML RISK MODEL  │  │  NLP ENGINE   │  │   DATABASE LAYER   │
    │                 │  │               │  │                    │
    │ Random Forest   │  │ TF-IDF Intent │  │  SQLite DB         │
    │ Gradient Boost  │  │ Classifier    │  │  ├── foods (96+)   │
    │ Logistic Reg    │  │               │  │  ├── meal_plans    │
    │ (Ensemble 97%)  │  │ RAG Engine    │  │  ├── chatbot_qa    │
    │                 │  │ (Cosine Sim)  │  │  ├── user_profiles │
    │ 100,768 records │  │               │  │  └── food_logs     │
    └─────────────────┘  │ Entity Extract│  │                    │
                         │               │  │  CSV Datasets      │
    ┌────────────────┐   │ Context Memory│  │  (100K patients)   │
    │  IMAGE MODEL   │   └───────────────┘  └────────────────────┘
    │                │
    │ Color Analysis │  ──► Upgrade to: MobileNetV2 / EfficientNet-B0
    │ (Current)      │       trained on Indian Food-101 dataset
    │                │
    │ CNN (Planned)  │
    └────────────────┘
```

---

## 2. TECHNOLOGY STACK

### Frontend
- HTML5 + CSS3 + Vanilla JavaScript (current — no build tools needed)
- **Upgrade option:** React.js + Tailwind CSS
- Chart.js for glucose trend visualization
- Responsive design (mobile-first)

### Backend
- **Python 3.11** + Flask 3.x
- REST API architecture
- Session-based user tracking

### ML / AI Layer
| Component | Technology | Accuracy |
|-----------|-----------|---------|
| Risk Model | RF + GB + LR Ensemble | 97% |
| NLP Intent | TF-IDF + Cosine Similarity | 34+ intents |
| RAG Chatbot | TF-IDF over SQLite Q&A | 112+ Q&A pairs |
| Image (current) | Color/texture matching | ~60% |
| Image (upgrade) | MobileNetV2 / EfficientNet | 85-90% target |

### Database
- **SQLite** (current — zero-config, runs anywhere)
- **Upgrade:** PostgreSQL (production) / Supabase (cloud)
- **Vector DB (RAG upgrade):** ChromaDB / Pinecone / FAISS

---

## 3. DATASET STRUCTURE

### 3a. Foods Dataset (SQLite — foods table)
```json
{
  "id": 1,
  "name": "ragi_dosa",
  "name_local": "ராகி தோசை",
  "category": "grain",
  "region": "south_indian",
  "cal_100g": 130,
  "carb_100g": 22,
  "protein_100g": 4,
  "fat_100g": 3,
  "fiber_100g": 3,
  "glucose_impact": 22,
  "gi": "low",
  "gi_value": 44,
  "suitable_t1": 1,
  "suitable_t2": 1,
  "serving_g": 80,
  "tags": ["breakfast", "millet", "diabetic_friendly"],
  "notes": "Finger millet crepe, excellent for diabetics",
  "color_r": 160, "color_g": 130, "color_b": 90
}
```

### 3b. Meal Plans Dataset
```json
{
  "diabetes_type": "type2",
  "meal_time": "breakfast",
  "glucose_range": "normal",
  "meal_name": "Ragi Dosa + Sambar",
  "foods": ["ragi_dosa", "sambar", "coconut_chutney"],
  "total_cal": 230,
  "total_carb": 35,
  "total_glc": 32,
  "gi_rating": "low",
  "reason": "Ragi has GI 44, lowest glucose impact breakfast"
}
```

### 3c. Chatbot Q&A Dataset
```json
{
  "intent": "food_nutrition",
  "question": "Is ragi dosa good for diabetes?",
  "answer": "Yes! Ragi dosa is the best breakfast for diabetics — GI 44 (low), high fiber, only 130 cal/100g...",
  "tags": "food,ragi,south_indian",
  "language": "en",
  "difficulty": "basic"
}
```

### 3d. Patient Risk Dataset Schema
```csv
age, bmi, blood_glucose_level, HbA1c_level, hypertension, heart_disease,
smoking_history, gender, diabetes
45, 28.5, 145, 6.2, 1, 0, never, Female, 1
```

---

## 4. IMAGE MODEL — UPGRADE GUIDE

### Current (Color-based)
- Accuracy: ~60% on common Indian foods
- Works offline, no GPU needed
- 96 food color signatures in database

### Recommended Upgrade: MobileNetV2 + Fine-tuning

```python
# Transfer Learning with MobileNetV2 for Indian food recognition
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model

def build_food_classifier(num_classes=96):
    base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
    base.trainable = False  # Freeze base initially

    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base.input, outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Training pipeline
def prepare_dataset(image_dir, img_size=(224,224), batch_size=32):
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        validation_split=0.2
    )
    train_gen = datagen.flow_from_directory(image_dir, target_size=img_size,
                    batch_size=batch_size, subset='training')
    val_gen   = datagen.flow_from_directory(image_dir, target_size=img_size,
                    batch_size=batch_size, subset='validation')
    return train_gen, val_gen

# Phase 1: Train top layers only (10 epochs)
# Phase 2: Unfreeze last 30 layers, fine-tune with low LR (10 more epochs)
```

### Recommended Datasets for Training
1. **Indian Food-101** — 101 classes, ~10K images (Kaggle)
2. **Food-101** (Stanford) — transfer learning base
3. **Custom South Indian dataset** — build by scraping + labeling 50 images per dish

### Alternative: Free APIs
- **LogMeal API** — food recognition, has Indian foods (free tier: 100 calls/day)
- **Clarifai Food Model** — general food recognition (free tier available)
- **Nutritionix API** — nutrition database (50K calls/day free)

---

## 5. CHATBOT NLP PIPELINE

```
User Input
    │
    ▼
Preprocessing
(lowercase, strip, spell-correct)
    │
    ├─► Entity Extraction ──────────────────────┐
    │   • Food names (FOOD_ALIASES dict)         │
    │   • Numbers/portions (regex)               │
    │   • Glucose values (regex)                 │
    │   • Age, BMI, HbA1c (regex)               │
    │   • Diabetes type (keyword)                │
    │                                            │
    ▼                                            ▼
TF-IDF Intent Classification          RAG Retrieval
(34 intent categories)                (cosine similarity
         │                             over 112+ Q&A)
         │                                    │
         └───────────────┬────────────────────┘
                         │
                         ▼
               Response Generator
               • Food lookup from SQLite
               • Meal plan from DB
               • Personalization (profile)
               • Context from history
                         │
                         ▼
                  Final Response
```

### Upgrade to Full RAG with Vector DB
```python
# Install: pip install chromadb sentence-transformers
import chromadb
from sentence_transformers import SentenceTransformer

class RAGChatbot:
    def __init__(self, db_path):
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.client  = chromadb.PersistentClient(path="./chroma_db")
        self.coll    = self.client.get_or_create_collection("diabetes_qa")
        self._index_qa(db_path)

    def _index_qa(self, db_path):
        import sqlite3
        conn = sqlite3.connect(db_path)
        rows = conn.execute("SELECT id, question, answer, intent FROM chatbot_qa").fetchall()
        embeddings = self.encoder.encode([r[1] for r in rows]).tolist()
        self.coll.upsert(
            ids=[str(r[0]) for r in rows],
            embeddings=embeddings,
            documents=[r[1] for r in rows],
            metadatas=[{"answer": r[2], "intent": r[3]} for r in rows]
        )
        print(f"Indexed {len(rows)} Q&A pairs")

    def retrieve(self, query, n_results=3):
        emb = self.encoder.encode([query]).tolist()
        res = self.coll.query(query_embeddings=emb, n_results=n_results)
        return [(res['documents'][0][i], res['metadatas'][0][i]['answer'])
                for i in range(len(res['documents'][0]))]
```

---

## 6. PERSONALIZATION LOGIC

```python
def get_personalized_meal(profile):
    """
    Personalization rules based on:
    - Glucose level (normal/high/low)
    - Age (young/middle/senior)
    - BMI (normal/overweight/obese)
    - Region (south/north Indian)
    - Dietary preference (veg/non-veg)
    - Diabetes type (1/2)
    """
    filters = {"diabetes_type": profile["diabetes_type"]}

    # Glucose-based adjustment
    if profile["glucose_level"] > 180:
        filters["glucose_range"] = "high"
        # Prioritize bitter gourd, keerai, rasam
    elif profile["glucose_level"] < 80:
        filters["glucose_range"] = "low"
        # Add faster-acting carbs
    else:
        filters["glucose_range"] = "normal"

    # Age-based adjustment
    if profile["age"] > 65:
        # Prefer softer foods: khichdi, upma, idli
        filters["texture"] = "soft"

    # BMI-based calorie adjustment
    if profile["bmi"] > 30:
        filters["max_cal"] = 1400  # calorie restriction
    elif profile["bmi"] < 18.5:
        filters["min_cal"] = 1800  # calorie surplus

    # Region-based food preference
    if profile["region"] == "south_indian":
        filters["preferred_region"] = "south_indian"
        # Prioritize: ragi, sambar, rasam, idli, dosa

    return query_db_with_filters(filters)
```

---

## 7. STEP-BY-STEP IMPLEMENTATION PLAN

### Phase 1 — Foundation (Weeks 1-2) ✅ DONE
- [x] Flask backend with REST API
- [x] ML risk prediction model (97% accuracy)
- [x] SQLite database with 96 foods + Q&A
- [x] Basic NLP chatbot (34 intents)
- [x] Nutrition calculator
- [x] South Indian food specialist

### Phase 2 — Enhancement (Weeks 3-4)
- [ ] Expand food database to 300+ foods
- [ ] Add 500+ Q&A pairs to chatbot_qa table
- [ ] Implement vector DB (ChromaDB) for RAG
- [ ] MobileNetV2 image model training
- [ ] User profile persistence (login system)

### Phase 3 — Production (Weeks 5-6)
- [ ] Deploy to cloud (Render / Railway — free tier)
- [ ] Add Tamil language support
- [ ] Glucose trend tracking + charts
- [ ] Daily meal logging with analytics
- [ ] Doctor report generation (PDF)

### Phase 4 — Advanced (Optional)
- [ ] Mobile app (React Native)
- [ ] Continuous glucose monitor (CGM) integration
- [ ] WhatsApp chatbot integration

---

## 8. DEPLOYMENT GUIDE

### Local Development
```bash
# Install dependencies
pip install flask pandas numpy scikit-learn Pillow

# Build database
python database/build_db.py

# Run app
python app.py
# Open: http://127.0.0.1:5000
```

### Cloud Deployment (Render — Free)
```yaml
# render.yaml
services:
  - type: web
    name: diabetes-meal-ai
    env: python
    buildCommand: "pip install -r requirements.txt && python database/build_db.py"
    startCommand: "gunicorn app:app"
    envVars:
      - key: FLASK_ENV
        value: production
```

### Requirements (Production)
```
flask>=3.0.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
Pillow>=10.0.0
gunicorn>=21.0.0
# For upgrade:
# tensorflow>=2.13.0
# chromadb>=0.4.0
# sentence-transformers>=2.2.0
```

---

## 9. EVALUATION METRICS

| Metric | Current | Target |
|--------|---------|--------|
| Risk Model Accuracy | 97% | 97%+ |
| Chatbot Intent Accuracy | ~85% | 90%+ |
| Food Image Recognition | ~60% | 85%+ (CNN) |
| Response Relevance | Good | Excellent (RAG) |
| Foods in Database | 96 | 300+ |
| Q&A Pairs | 112 | 500+ |
| South Indian Foods | 39 | 100+ |

---

## 10. HOW TO RUN

```bash
# Step 1: Install
pip install -r requirements.txt

# Step 2: Build database
python database/build_db.py

# Step 3: Run
python app.py

# Step 4: Open browser
# http://127.0.0.1:5000
```
