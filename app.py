"""
app.py — Flask Backend API
==========================
Endpoints
---------
GET  /health          → service health check
GET  /model-info      → model metadata and evaluation metrics
POST /predict         → toxicity prediction for a single comment
POST /predict-batch   → toxicity prediction for a list of comments
"""

import os
import re
import json
import string
import joblib

from flask import Flask, request, jsonify

app = Flask(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(os.path.dirname(BASE_DIR), "model")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")

# ── Load model artifacts (once at startup) ────────────────────────────────────
model      = joblib.load(os.path.join(MODEL_DIR, "toxic_model.pkl"))
vectorizer = joblib.load(os.path.join(MODEL_DIR, "vectorizer.pkl"))

with open(METRICS_PATH, "r", encoding="utf-8") as f:
    METRICS = json.load(f)

# ── Portuguese stopwords ───────────────────────────────────────────────────────
PT_STOPWORDS = {
    "a","ao","aos","aquela","aquelas","aquele","aqueles","aquilo","as","ate",
    "com","como","da","das","de","dela","delas","dele","deles","depois","do",
    "dos","e","ela","elas","ele","eles","em","entre","era","essa","essas",
    "esse","esses","esta","estas","este","estes","eu","foi","foram","há",
    "isso","isto","ja","lhe","lhes","lo","mais","mas","me","mesmo","meu",
    "meus","minha","minhas","muito","na","nas","nao","nem","no","nos",
    "nossa","nossas","nosso","nossos","num","numa","o","os","ou","para",
    "pela","pelas","pelo","pelos","por","pra","qual","quando","que","quem",
    "se","seja","sem","seu","seus","si","so","sua","suas","também","te",
    "tem","ter","teu","teus","tua","tuas","um","uma","umas","uns","você",
    "vocês","vos","à","às","é","não","ser","uma","mais","pela","pelo",
    "user","https","http","rt","co","via","pra","vc","tb","td","ne",
}

def clean_text(text: str) -> str:
    """Mirror of the same cleaning used during training."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#\w+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    tokens = [t for t in tokens if t not in PT_STOPWORDS and len(t) > 2]
    return " ".join(tokens)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "Toxic Comment Detection API"})


@app.route("/model-info", methods=["GET"])
def model_info():
    return jsonify({
        "model": METRICS.get("model"),
        "vectorizer": METRICS.get("vectorizer"),
        "training_samples": METRICS.get("train_samples"),
        "test_samples": METRICS.get("test_samples"),
        "total_samples": METRICS.get("total_samples"),
        "class_distribution": METRICS.get("class_distribution"),
        "metrics": {
            "accuracy":  METRICS.get("accuracy"),
            "precision": METRICS.get("precision"),
            "recall":    METRICS.get("recall"),
            "f1_score":  METRICS.get("f1_score"),
        },
        "confusion_matrix": METRICS.get("confusion_matrix"),
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not data or "comment" not in data:
        return jsonify({"error": "Missing 'comment' field in JSON body."}), 400

    comment = data["comment"]
    if not isinstance(comment, str) or not comment.strip():
        return jsonify({"error": "Comment must be a non-empty string."}), 400

    cleaned  = clean_text(comment)
    vec      = vectorizer.transform([cleaned])
    pred     = int(model.predict(vec)[0])
    proba    = model.predict_proba(vec)[0]

    return jsonify({
        "comment":    comment,
        "prediction": pred,
        "label":      "Toxic" if pred == 1 else "Non-Toxic",
        "probability": {
            "non_toxic": round(float(proba[0]), 4),
            "toxic":     round(float(proba[1]), 4),
        }
    })


@app.route("/predict-batch", methods=["POST"])
def predict_batch():
    data = request.get_json(silent=True)
    if not data or "comments" not in data:
        return jsonify({"error": "Missing 'comments' list in JSON body."}), 400

    comments = data["comments"]
    if not isinstance(comments, list) or len(comments) == 0:
        return jsonify({"error": "'comments' must be a non-empty list."}), 400

    results = []
    for comment in comments:
        cleaned = clean_text(str(comment))
        vec     = vectorizer.transform([cleaned])
        pred    = int(model.predict(vec)[0])
        proba   = model.predict_proba(vec)[0]
        results.append({
            "comment":    comment,
            "prediction": pred,
            "label":      "Toxic" if pred == 1 else "Non-Toxic",
            "probability": {
                "non_toxic": round(float(proba[0]), 4),
                "toxic":     round(float(proba[1]), 4),
            }
        })

    return jsonify({"results": results, "count": len(results)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
