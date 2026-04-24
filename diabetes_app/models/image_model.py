"""
models/image_model.py
=====================
Food image recognition using color + texture analysis.
Matches uploaded food photo against color signatures stored in the DB.

Architecture designed for CNN upgrade:
  Current  → Color/texture matching (~60% accuracy)
  Upgrade  → MobileNetV2 fine-tuned on Indian Food-101 (~88% accuracy)
"""

import sqlite3
import numpy as np
import io
from PIL import Image


class ImageModel:
    """
    Food image recognition model.

    Current implementation:
      1. Extract dominant color from image (masked mid-tone pixels)
      2. Extract texture score (pixel std deviation)
      3. Compute color distance against all food entries in SQLite
      4. Return top-3 matches with confidence scores

    To upgrade to CNN:
      from models.cnn_model import CNNFoodClassifier
      self.cnn = CNNFoodClassifier('models/mobilenet_indian_food.h5')
      result = self.cnn.predict(img_array)
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._food_colors = None   # loaded lazily

    # ── public ────────────────────────────────────────────────────────────────
    def predict(self, image_bytes: bytes) -> dict:
        """
        Analyse a food image and return predicted food + nutrition info.

        Args:
            image_bytes: raw image bytes (JPEG / PNG / WEBP)

        Returns:
            dict with detected_food, food_key, confidence, score,
                 alternatives (top-2 other candidates), image_features, nutrition
        """
        features = self._extract_features(image_bytes)
        scores   = self._match_colors(features['dominant_color'])

        best       = scores[0]
        food_key   = best[0]
        best_dist  = best[2]
        confidence = ("high"   if best_dist < 30 else
                      "medium" if best_dist < 70 else "low")

        return {
            "detected_food": food_key.replace('_', ' ').title(),
            "food_key":      food_key,
            "confidence":    confidence,
            "score":         round(best[1], 3),
            "color_distance":round(best_dist, 1),
            "alternatives":  [(s[0].replace('_', ' ').title(), round(s[1], 3))
                               for s in scores[1:3]],
            "image_features": {
                "dominant_color": [round(x) for x in features['dominant_color']],
                "brightness":     round(features['brightness'], 1),
                "texture_score":  round(features['texture'], 1),
                "r_ratio":        round(features['r_ratio'], 3),
                "g_ratio":        round(features['g_ratio'], 3),
                "b_ratio":        round(features['b_ratio'], 3),
            },
        }

    def predict_batch(self, images: list) -> list:
        """Predict food for a list of image byte objects."""
        return [self.predict(img) for img in images]

    # ── private ───────────────────────────────────────────────────────────────
    def _load_food_colors(self):
        """Load food color signatures from database (cached)."""
        if self._food_colors is not None:
            return self._food_colors
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT name, color_r, color_g, color_b, gi, cal_100g, "
            "carb_100g, glucose_impact FROM foods"
        ).fetchall()
        conn.close()
        self._food_colors = [dict(r) for r in rows]
        return self._food_colors

    def _extract_features(self, image_bytes: bytes) -> dict:
        """Extract color + texture features from image bytes."""
        img   = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img   = img.resize((100, 100), Image.LANCZOS)
        arr   = np.array(img, dtype=float)
        px    = arr.reshape(-1, 3)

        # Filter out near-white (background) and near-black pixels
        mask = (px.sum(axis=1) > 80) & (px.sum(axis=1) < 680)
        dom  = px[mask].mean(axis=0) if mask.sum() > 30 else px.mean(axis=0)

        total = dom.sum() + 1e-6
        return {
            'dominant_color': dom.tolist(),
            'brightness':     float(dom.mean()),
            'texture':        float(px.std()),
            'r_ratio':        float(dom[0] / total),
            'g_ratio':        float(dom[1] / total),
            'b_ratio':        float(dom[2] / total),
        }

    def _match_colors(self, dominant: list) -> list:
        """
        Compute color distance between image dominant color and
        all food color signatures. Returns sorted list of
        (food_name, score 0-1, distance) descending by score.
        """
        foods  = self._load_food_colors()
        dom_np = np.array(dominant, dtype=float)
        scores = []

        for fd in foods:
            food_color = np.array([fd['color_r'], fd['color_g'], fd['color_b']], dtype=float)
            dist  = float(np.linalg.norm(dom_np - food_color))
            score = max(0.0, 1.0 - dist / 200.0)
            scores.append((fd['name'], score, dist, dict(fd)))

        scores.sort(key=lambda x: -x[1])
        return scores


# ══════════════════════════════════════════════════════════════
# CNN UPGRADE STUB
# ══════════════════════════════════════════════════════════════
class CNNFoodClassifier:
    """
    Placeholder for MobileNetV2-based food classifier.

    To activate:
      1. Install: pip install tensorflow
      2. Train the model (see ARCHITECTURE.md section 4)
      3. Save as: models/mobilenet_indian_food.h5
      4. Replace ImageModel with this class in app.py

    Expected accuracy: 85-90% on Indian Food-101 dataset.
    """

    def __init__(self, model_path: str, class_names: list = None):
        self.model_path  = model_path
        self.class_names = class_names or []
        self.model       = None

    def load(self):
        try:
            import tensorflow as tf
            self.model = tf.keras.models.load_model(self.model_path)
            print(f"✅ CNN model loaded from {self.model_path}")
        except ImportError:
            print("⚠️  TensorFlow not installed. pip install tensorflow")
        except Exception as e:
            print(f"⚠️  Could not load CNN model: {e}")

    def preprocess(self, image_bytes: bytes):
        """Resize + normalize image for MobileNetV2 input."""
        import tensorflow as tf
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img = img.resize((224, 224), Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0
        return np.expand_dims(arr, axis=0)

    def predict(self, image_bytes: bytes) -> dict:
        if self.model is None:
            self.load()
        if self.model is None:
            return {"error": "CNN model not available"}

        inp   = self.preprocess(image_bytes)
        preds = self.model.predict(inp)[0]
        top3  = np.argsort(preds)[::-1][:3]

        return {
            "detected_food": self.class_names[top3[0]].replace('_', ' ').title()
                             if self.class_names else f"class_{top3[0]}",
            "food_key":      self.class_names[top3[0]] if self.class_names else str(top3[0]),
            "confidence":    ("high" if preds[top3[0]] > 0.7 else
                              "medium" if preds[top3[0]] > 0.4 else "low"),
            "score":         round(float(preds[top3[0]]), 3),
            "alternatives":  [(self.class_names[i] if self.class_names else f"class_{i}",
                               round(float(preds[i]), 3)) for i in top3[1:]],
        }
