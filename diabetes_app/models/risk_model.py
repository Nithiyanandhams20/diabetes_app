"""
models/risk_model.py
====================
ML Ensemble Risk Model for Diabetes Prediction.
Trains Random Forest + Gradient Boosting + Logistic Regression
on 100,768 combined patient records.
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')


class RiskModel:
    """
    Diabetes risk prediction using a weighted ensemble of:
      - Random Forest (weight 0.50)
      - Gradient Boosting (weight 0.30)
      - Logistic Regression (weight 0.20)

    Usage:
        model = RiskModel(data_dir='data/')
        model.train()
        result = model.predict({'age': 45, 'bmi': 28, 'glucose': 145, 'hba1c': 6.2, ...})
    """

    FEATURE_COLS = [
        'age', 'bmi', 'blood_glucose_level', 'HbA1c_level',
        'hypertension', 'heart_disease', 'insulin', 'pregnancies', 'dpf',
        'gender_enc', 'smoking_enc'
    ]

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.scaler      = StandardScaler()
        self.le_gender   = LabelEncoder()
        self.le_smoking  = LabelEncoder()
        self.rf  = RandomForestClassifier(n_estimators=150, max_depth=10,
                                          min_samples_leaf=5, random_state=42, n_jobs=-1)
        self.gb  = GradientBoostingClassifier(n_estimators=100, max_depth=5,
                                               learning_rate=0.05, random_state=42)
        self.lr  = LogisticRegression(max_iter=500, C=1.0, random_state=42)
        self.accuracy    = {}
        self.feat_imp    = {}
        self.ds          = {}
        self._trained    = False

    # ── public ────────────────────────────────────────────────────────────────
    def train(self):
        """Load datasets, merge, train all models."""
        df = self._load_data()
        self._compute_stats(df)
        X, y = self._prepare_features(df)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                                    random_state=42, stratify=y)
        X_trs = self.scaler.fit_transform(X_tr)
        X_tes = self.scaler.transform(X_te)

        for clf, name in [(self.rf,'random_forest'), (self.gb,'gradient_boost'), (self.lr,'logistic_reg')]:
            clf.fit(X_trs, y_tr)
            acc = round(accuracy_score(y_te, clf.predict(X_tes)) * 100, 2)
            self.accuracy[name] = acc

        self.accuracy['ensemble'] = round(
            0.5 * self.accuracy['random_forest'] / 100 +
            0.3 * self.accuracy['gradient_boost'] / 100 +
            0.2 * self.accuracy['logistic_reg'] / 100, 4) * 100
        self.accuracy['ensemble'] = round(self.accuracy['ensemble'], 2)

        self.feat_imp = dict(zip(self.FEATURE_COLS, self.rf.feature_importances_))
        self._trained = True
        print(f"   ✅ RiskModel: RF={self.accuracy['random_forest']}% | "
              f"GB={self.accuracy['gradient_boost']}% | "
              f"Ensemble={self.accuracy['ensemble']}%")

    def predict(self, features: dict) -> dict:
        """
        Predict diabetes risk from patient features.

        Args:
            features: dict with keys: age, bmi, glucose, hba1c,
                      hypertension, heart_disease, smoking, gender

        Returns:
            dict with risk_score, risk_level, diabetes_type, risk_factors,
                 color, model_accuracy, similar_patients, recommendation
        """
        if not self._trained:
            self.train()

        row = self._feature_row(features)
        xs  = self.scaler.transform(row)
        prob = (0.5 * self.rf.predict_proba(xs)[0][1] +
                0.3 * self.gb.predict_proba(xs)[0][1] +
                0.2 * self.lr.predict_proba(xs)[0][1])
        prob = round(float(prob) * 100, 1)

        risk_level, color = self._risk_label(prob)
        risk_factors      = self._build_risk_factors(features)
        similar_pct       = self._similar_patients(features)
        dtype             = 'type2' if prob >= 30 else 'type1'

        return {
            "risk_score":           prob,
            "risk_level":           risk_level,
            "diabetes_type":        dtype,
            "risk_factors":         risk_factors,
            "color":                color,
            "dataset_size":         self.ds['total'],
            "similar_patients":     self.ds.get('similar_n', 0),
            "similar_diabetic_pct": similar_pct,
            "model_accuracy":       self.accuracy['ensemble'],
            "feature_importance":   {k: round(v*100, 1) for k, v in
                                     sorted(self.feat_imp.items(), key=lambda x:-x[1])[:5]},
            "recommendation":       (
                "⚠️ High risk detected. Consult your doctor and follow a strict low-GI diet."
                if prob >= 60 else
                "⚠️ Moderate risk. Adopt low-GI South Indian diet and exercise 30 min daily."
                if prob >= 30 else
                "✅ Low risk. Maintain healthy diet and routine glucose monitoring."
            )
        }

    def get_top_features(self, n: int = 5) -> list:
        """Return top N most important risk features."""
        sorted_f = sorted(self.feat_imp.items(), key=lambda x: -x[1])
        return [(k.replace('_', ' ').title(), round(v*100, 1)) for k, v in sorted_f[:n]]

    # ── private ───────────────────────────────────────────────────────────────
    def _load_data(self):
        main_path = os.path.join(self.data_dir, 'diabetes_prediction_dataset.csv')
        pima_path = os.path.join(self.data_dir, 'diabetes.csv')
        df_main   = pd.read_csv(main_path)
        df_pima   = pd.read_csv(pima_path)

        df_pima_n = pd.DataFrame({
            'age':                 df_pima['Age'].astype(float),
            'bmi':                 df_pima['BMI'].astype(float),
            'blood_glucose_level': df_pima['Glucose'].astype(float),
            'HbA1c_level':         (df_pima['Glucose'] / 30).clip(4, 14),
            'hypertension':        (df_pima['BloodPressure'] > 90).astype(int),
            'heart_disease':       0,
            'smoking_history':     'unknown',
            'gender':              'Female',
            'diabetes':            df_pima['Outcome'],
            'insulin':             df_pima['Insulin'].astype(float),
            'pregnancies':         df_pima['Pregnancies'].astype(float),
            'dpf':                 df_pima['DiabetesPedigreeFunction'].astype(float),
        })
        df_main_n = df_main.copy()
        for col in ['insulin', 'pregnancies', 'dpf']:
            df_main_n[col] = 0.0

        df_all = pd.concat([df_main_n, df_pima_n], ignore_index=True)
        print(f"   📂 Loaded {len(df_all):,} patient records")
        self._raw = df_all      # store for similar-patient lookup
        return df_all

    def _compute_stats(self, df):
        _d  = df[df['diabetes'] == 1]
        _nd = df[df['diabetes'] == 0]
        self.ds = {
            'total':    len(df),
            'diabetic': int(df['diabetes'].sum()),
            'glc_d':    round(float(_d['blood_glucose_level'].mean()), 1),
            'glc_nd':   round(float(_nd['blood_glucose_level'].mean()), 1),
            'bmi_d':    round(float(_d['bmi'].mean()), 1),
            'hba1c_d':  round(float(_d['HbA1c_level'].mean()), 1),
            'hyp_pct':  round(float(df['hypertension'].mean()) * 100, 1),
        }

    def _prepare_features(self, df):
        df = df.copy()
        df['gender_enc']  = self.le_gender.fit_transform(df['gender'].fillna('unknown'))
        df['smoking_enc'] = self.le_smoking.fit_transform(df['smoking_history'].fillna('unknown'))
        dm = df.dropna(subset=['diabetes'])
        X  = dm[self.FEATURE_COLS].fillna(0)
        y  = dm['diabetes'].astype(int)
        return X, y

    def _feature_row(self, features):
        try:
            ge = int(self.le_gender.transform([features.get('gender', 'female')])[0])
        except Exception:
            ge = 0
        try:
            se = int(self.le_smoking.transform([features.get('smoking', 'never')])[0])
        except Exception:
            se = 0
        row = {
            'age':                 float(features.get('age', 0)),
            'bmi':                 float(features.get('bmi', 0)),
            'blood_glucose_level': float(features.get('glucose', 0)),
            'HbA1c_level':         float(features.get('hba1c', 0)),
            'hypertension':        int(features.get('hypertension', 0)),
            'heart_disease':       int(features.get('heart_disease', 0)),
            'insulin':             0.0,
            'pregnancies':         0.0,
            'dpf':                 0.0,
            'gender_enc':          ge,
            'smoking_enc':         se,
        }
        return pd.DataFrame([row]).fillna(0)

    def _risk_label(self, prob):
        if prob >= 60:  return "High Risk",     "#ef4444"
        if prob >= 30:  return "Moderate Risk", "#f97316"
        return "Low Risk", "#22c55e"

    def _build_risk_factors(self, f):
        flags = []
        glc = float(f.get('glucose', 0))
        hba = float(f.get('hba1c', 0))
        bmi = float(f.get('bmi', 0))
        age = float(f.get('age', 0))
        if glc >= 200:  flags.append(f"Very high blood glucose ({glc} mg/dL ≥200)")
        elif glc >= 126:flags.append(f"High fasting glucose ({glc} mg/dL ≥126)")
        elif glc >= 100:flags.append(f"Pre-diabetic glucose ({glc} mg/dL 100–125)")
        if hba >= 6.5:  flags.append(f"HbA1c {hba}% — diabetic range (≥6.5%)")
        elif hba >= 5.7:flags.append(f"HbA1c {hba}% — pre-diabetic (5.7–6.4%)")
        if bmi >= 30:   flags.append(f"Obese (BMI {bmi} ≥30)")
        elif bmi >= 25: flags.append(f"Overweight (BMI {bmi} 25–29.9)")
        if age >= 45:   flags.append(f"Age {int(age)} ≥45 (higher-risk group)")
        if f.get('hypertension'):   flags.append("Hypertension present")
        if f.get('heart_disease'):  flags.append("Heart disease present")
        if f.get('smoking') == 'current': flags.append("Current smoker")
        return flags

    def _similar_patients(self, f):
        if not hasattr(self, '_raw'):
            return None
        bmi = float(f.get('bmi', 0))
        age = float(f.get('age', 0))
        sim = self._raw[
            (self._raw['bmi'].between(bmi - 3, bmi + 3)) &
            (self._raw['age'].between(age - 5, age + 5))
        ]
        self.ds['similar_n'] = len(sim)
        return round(float(sim['diabetes'].mean()) * 100, 1) if len(sim) > 5 else None
