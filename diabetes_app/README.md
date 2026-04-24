# DiabetesMeal AI 🩺🍛
**Final Year Project — Precise Meal Prediction for Diabetes Patients**

---

## Project Overview
An AI-powered web application that:
- **Predicts diabetes risk** from patient health parameters (100,000 record dataset)
- **Recommends personalised Indian meal plans** for Type 1 & Type 2 diabetes
- **Analyzes any food item** for glucose impact, calories, carbs, protein, fat, fiber
- **Scans food photos** using AI to predict nutritional content
- **AI Chat Assistant** for real-time diabetes nutrition Q&A (Indian food focused)

---

## Datasets Used
| File | Description |
|------|-------------|
| `diabetes_prediction_dataset.csv` | 100,000 patient records with HbA1c, glucose, BMI, hypertension |
| `diabetes.csv` | 768 Pima Indian diabetes records (classic ML dataset) |
| `PP_users.csv` | Food rating users data |
| `interactions_test.csv` | Recipe interaction data |
| `List_of_Indian_Foods.txt` | 80+ Indian food items |

---

## How to Run

### Step 1 — Install Python (if not installed)
Download Python 3.9+ from https://python.org

### Step 2 — Install dependencies
Open terminal / command prompt in the `diabetes_app` folder:
```bash
pip install -r requirements.txt
```

### Step 3 — Set your Anthropic API Key
The AI Chat and Photo Scanner features require a Claude API key.
Get one free at https://console.anthropic.com

**Windows:**
```cmd
set ANTHROPIC_API_KEY=sk-ant-your-key-here
```

**Mac / Linux:**
```bash
export ANTHROPIC_API_KEY=sk-ant-your-key-here
```

### Step 4 — Run the app
```bash
python app.py
```

### Step 5 — Open in browser
Go to: **http://127.0.0.1:5000**

---

## Features

### 🩺 Risk Assessment
- Enter age, BMI, blood glucose, HbA1c, hypertension, heart disease
- Get risk score (0–100%) with contributing factors
- Automatically suggests diabetes type context

### 🍽️ Meal Plans
- Personalised for Type 1 or Type 2 diabetes
- Breakfast, Lunch, Dinner, Snacks
- Shows GI rating and calorie estimate per meal
- Lists foods to avoid

### 🔍 Food Analyzer
- Search from 80+ Indian foods with autocomplete
- Enter portion size in grams
- Get: calories, glucose impact, carbs, protein, fat, fiber, GI
- Suitability rating for your diabetes type

### 📷 Photo Scanner
- Upload food photo
- AI identifies food and estimates nutrition values
- Returns glucose impact, calories, full macros

### 💬 AI Chat
- Ask anything about diabetes nutrition
- Indian food focused knowledge
- Quick suggestion buttons for common questions

---

## Project Structure
```
diabetes_app/
├── app.py                  ← Main Flask application
├── requirements.txt        ← Python dependencies
├── templates/
│   └── index.html          ← Frontend UI
└── data/
    ├── diabetes_prediction_dataset.csv
    ├── diabetes.csv
    ├── PP_users.csv
    ├── interactions_test.csv
    ├── interactions_validation.csv
    └── List_of_Indian_Foods.txt
```

---

## Technologies Used
- **Backend**: Python, Flask
- **Data**: Pandas, NumPy, Scikit-learn
- **Frontend**: HTML5, CSS3, Vanilla JavaScript
- **AI**: Anthropic Claude API (image analysis + chat)
- **Datasets**: 100K+ patient records, Indian food database

---

*⚠️ Disclaimer: This tool is for educational and research purposes only. Always consult a qualified medical professional for diabetes management decisions.*
