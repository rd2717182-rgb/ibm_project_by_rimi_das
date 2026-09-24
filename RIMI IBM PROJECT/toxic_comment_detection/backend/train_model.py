"""
train_model.py
==============
Trains a Logistic Regression classifier for toxic comment detection
on the comentarios_toxicos_ptBR dataset (Brazilian Portuguese).

Outputs
-------
../model/toxic_model.pkl   — trained Logistic Regression pipeline
../model/vectorizer.pkl    — fitted TF-IDF vectorizer
../model/metrics.json      — accuracy, precision, recall, F1 + confusion matrix
../report_images/          — EDA and evaluation charts
"""

import os
import re
import json
import string
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATA_PATH   = os.path.join(PROJECT_DIR, "data", "comentarios_toxicos_ptBR.csv")
MODEL_DIR   = os.path.join(PROJECT_DIR, "model")
IMG_DIR     = os.path.join(PROJECT_DIR, "report_images")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(IMG_DIR,   exist_ok=True)

# ── Portuguese stopwords (built-in, no NLTK download needed) ──────────────────
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

# ── Text cleaning ──────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)          # remove URLs
    text = re.sub(r"@\w+", " ", text)                     # remove @mentions
    text = re.sub(r"#\w+", " ", text)                     # remove hashtags
    text = re.sub(r"\d+", " ", text)                      # remove digits
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    tokens = [t for t in tokens if t not in PT_STOPWORDS and len(t) > 2]
    return " ".join(tokens)

# ── Load & clean dataset ───────────────────────────────────────────────────────
print("[1/6] Loading dataset …")
df = pd.read_csv(DATA_PATH, index_col=0, encoding="utf-8")
print(f"      Loaded {len(df):,} rows. Columns: {df.columns.tolist()}")

# Drop nulls in text columns
before = len(df)
df.dropna(subset=["text"], inplace=True)
df.drop_duplicates(subset=["text"], inplace=True)
after = len(df)
print(f"      After cleaning: {after:,} rows (removed {before - after} null/duplicate rows)")

print("[2/6] Cleaning text …")
df["clean_text"] = df["text"].apply(clean_text)

# ── EDA charts ────────────────────────────────────────────────────────────────
print("[3/6] Generating EDA charts …")

# Chart 1 — Class distribution
fig, ax = plt.subplots(figsize=(6, 4))
counts = df["toxic"].value_counts().sort_index()
bars = ax.bar(["Non-Toxic (0)", "Toxic (1)"], counts.values,
              color=["#3b82d4", "#e74c3c"], edgecolor="white", linewidth=0.8)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 100,
            f"{val:,}", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_title("Class Distribution — Toxic vs Non-Toxic", fontsize=13, fontweight="bold")
ax.set_ylabel("Number of Comments")
ax.set_xlabel("Label")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "class_distribution.png"), dpi=150)
plt.close()

# Chart 2 — Comment length distribution
df["text_len"] = df["text"].str.len()
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(df[df["toxic"] == 0]["text_len"], bins=60, alpha=0.65,
        label="Non-Toxic", color="#3b82d4")
ax.hist(df[df["toxic"] == 1]["text_len"], bins=60, alpha=0.65,
        label="Toxic", color="#e74c3c")
ax.set_title("Comment Length Distribution", fontsize=13, fontweight="bold")
ax.set_xlabel("Character Count")
ax.set_ylabel("Frequency")
ax.legend()
ax.set_xlim(0, 500)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "length_distribution.png"), dpi=150)
plt.close()

# Chart 3 — Top 20 words in toxic comments
from collections import Counter
toxic_words = " ".join(df[df["toxic"] == 1]["clean_text"].values)
word_counts = Counter(toxic_words.split()).most_common(20)
words, freqs = zip(*word_counts)
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(list(words)[::-1], list(freqs)[::-1], color="#e74c3c", edgecolor="white")
ax.set_title("Top 20 Most Frequent Words in Toxic Comments", fontsize=13, fontweight="bold")
ax.set_xlabel("Frequency")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "top_toxic_words.png"), dpi=150)
plt.close()

# ── TF-IDF + Model training ────────────────────────────────────────────────────
print("[4/6] Training model …")
X = df["clean_text"]
y = df["toxic"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

model = LogisticRegression(
    C=1.0, max_iter=1000, solver="lbfgs",
    class_weight="balanced", random_state=42
)
model.fit(X_train_vec, y_train)

# ── Evaluation ────────────────────────────────────────────────────────────────
print("[5/6] Evaluating …")
y_pred = model.predict(X_test_vec)

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)
cm        = confusion_matrix(y_test, y_pred).tolist()
report    = classification_report(y_test, y_pred, target_names=["Non-Toxic", "Toxic"])

print(f"\n{'='*50}")
print(f"  Accuracy : {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall   : {recall:.4f}")
print(f"  F1-Score : {f1:.4f}")
print(f"\nClassification Report:\n{report}")
print(f"{'='*50}\n")

# Chart 4 — Confusion matrix
fig, ax = plt.subplots(figsize=(5, 4))
cm_arr = np.array(cm)
sns.heatmap(cm_arr, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Non-Toxic", "Toxic"],
            yticklabels=["Non-Toxic", "Toxic"],
            linewidths=0.5, ax=ax)
ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold")
ax.set_ylabel("True Label")
ax.set_xlabel("Predicted Label")
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "confusion_matrix.png"), dpi=150)
plt.close()

# ── Save model artifacts ───────────────────────────────────────────────────────
print("[6/6] Saving model artifacts …")
joblib.dump(model,      os.path.join(MODEL_DIR, "toxic_model.pkl"))
joblib.dump(vectorizer, os.path.join(MODEL_DIR, "vectorizer.pkl"))

metrics = {
    "accuracy":  round(accuracy,  4),
    "precision": round(precision, 4),
    "recall":    round(recall,    4),
    "f1_score":  round(f1,        4),
    "confusion_matrix": cm,
    "train_samples": int(len(X_train)),
    "test_samples":  int(len(X_test)),
    "total_samples": int(len(df)),
    "class_distribution": {
        "non_toxic": int(counts[0]),
        "toxic":     int(counts[1])
    },
    "model": "LogisticRegression",
    "vectorizer": "TF-IDF (unigrams + bigrams, max 20k features)"
}
with open(os.path.join(MODEL_DIR, "metrics.json"), "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

print("Done! Artifacts saved to ../model/")
print(json.dumps(metrics, indent=2))
