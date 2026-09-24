"""
ui.py — Streamlit Frontend
==========================
Provides a browser-based UI for the Toxic Comment Detection system.
Connects to the Flask backend at http://localhost:5000.
"""

import os
import re
import string
import json
import joblib
import requests
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# ── Config ─────────────────────────────────────────────────────────────────────
API_BASE = "http://localhost:5000"
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
IMG_DIR   = os.path.join(os.path.dirname(BASE_DIR), "report_images")
DATA_PATH = os.path.join(os.path.dirname(BASE_DIR), "data", "comentarios_toxicos_ptBR.csv")

st.set_page_config(
    page_title="Toxic Comment Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Helpers ────────────────────────────────────────────────────────────────────
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

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#\w+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = [t for t in text.split() if t not in PT_STOPWORDS and len(t) > 2]
    return " ".join(tokens)


@st.cache_data
def load_dataset():
    df = pd.read_csv(DATA_PATH, index_col=0, encoding="utf-8")
    df.dropna(subset=["text"], inplace=True)
    df.drop_duplicates(subset=["text"], inplace=True)
    df["clean_text"] = df["text"].apply(clean_text)
    df["text_len"]   = df["text"].str.len()
    return df


def call_api_predict(comment):
    try:
        resp = requests.post(f"{API_BASE}/predict",
                             json={"comment": comment}, timeout=5)
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


def call_api_model_info():
    try:
        resp = requests.get(f"{API_BASE}/model-info", timeout=5)
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🛡️ Toxic Comment Detector")
    st.markdown("---")
    st.markdown("**Dataset:** `comentarios_toxicos_ptBR`")
    st.markdown("**Language:** Brazilian Portuguese 🇧🇷")
    st.markdown("**Model:** Logistic Regression + TF-IDF")
    st.markdown("---")
    page = st.radio("Navigate", ["🔍 Predict", "📊 Dashboard", "ℹ️ Model Info"])

# ── Page: Predict ──────────────────────────────────────────────────────────────
if page == "🔍 Predict":
    st.header("🔍 Toxicity Prediction")
    st.markdown("Enter a comment below to check whether it is toxic or not.")

    comment_input = st.text_area("Comment", height=120,
                                  placeholder="Type or paste a comment here…")

    col1, col2 = st.columns([1, 5])
    with col1:
        predict_btn = st.button("Predict", type="primary")

    if predict_btn and comment_input.strip():
        with st.spinner("Analyzing…"):
            result = call_api_predict(comment_input)

        if "error" in result:
            st.error(f"API error: {result['error']}\n\nMake sure the Flask backend is running on port 5000.")
        else:
            label = result["label"]
            toxic_prob    = result["probability"]["toxic"]
            nontoxic_prob = result["probability"]["non_toxic"]

            if label == "Toxic":
                st.error(f"⚠️ **{label}** — Toxicity probability: {toxic_prob:.1%}")
            else:
                st.success(f"✅ **{label}** — Non-toxic probability: {nontoxic_prob:.1%}")

            st.markdown("#### Probability Breakdown")
            prob_df = pd.DataFrame({
                "Category":    ["Non-Toxic", "Toxic"],
                "Probability": [nontoxic_prob, toxic_prob]
            })
            fig, ax = plt.subplots(figsize=(5, 2.5))
            colors = ["#3b82d4", "#e74c3c"]
            ax.barh(prob_df["Category"], prob_df["Probability"], color=colors)
            ax.set_xlim(0, 1)
            ax.set_xlabel("Probability")
            ax.spines[["top","right"]].set_visible(False)
            for i, val in enumerate(prob_df["Probability"]):
                ax.text(val + 0.01, i, f"{val:.1%}", va="center", fontsize=10)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    elif predict_btn:
        st.warning("Please enter a comment before clicking Predict.")

    st.markdown("---")
    st.markdown("#### Batch Prediction")
    batch_input = st.text_area("Enter multiple comments (one per line)",
                                height=150,
                                placeholder="Comment 1\nComment 2\n…")
    if st.button("Predict Batch"):
        lines = [l.strip() for l in batch_input.splitlines() if l.strip()]
        if lines:
            try:
                resp = requests.post(f"{API_BASE}/predict-batch",
                                     json={"comments": lines}, timeout=10)
                batch_results = resp.json().get("results", [])
                rows = []
                for r in batch_results:
                    rows.append({
                        "Comment": r["comment"][:80],
                        "Label":   r["label"],
                        "Toxic %": f"{r['probability']['toxic']:.1%}",
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
            except Exception as e:
                st.error(f"Backend error: {e}")
        else:
            st.warning("Please enter at least one comment.")

# ── Page: Dashboard ────────────────────────────────────────────────────────────
elif page == "📊 Dashboard":
    st.header("📊 Dataset Dashboard")

    df = load_dataset()
    counts = df["toxic"].value_counts().sort_index()
    toxic_count    = int(counts.get(1, 0))
    nontoxic_count = int(counts.get(0, 0))
    total          = len(df)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Comments", f"{total:,}")
    c2.metric("Non-Toxic",      f"{nontoxic_count:,}", f"{nontoxic_count/total:.1%}")
    c3.metric("Toxic",          f"{toxic_count:,}",    f"{toxic_count/total:.1%}")

    st.markdown("---")

    # Class distribution
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Class Distribution")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.bar(["Non-Toxic", "Toxic"], [nontoxic_count, toxic_count],
               color=["#3b82d4", "#e74c3c"], edgecolor="white")
        for i, v in enumerate([nontoxic_count, toxic_count]):
            ax.text(i, v + 100, f"{v:,}", ha="center", fontweight="bold")
        ax.set_ylabel("Count")
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_b:
        st.subheader("Comment Length Distribution")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.hist(df[df["toxic"]==0]["text_len"], bins=50, alpha=0.7,
                label="Non-Toxic", color="#3b82d4")
        ax.hist(df[df["toxic"]==1]["text_len"], bins=50, alpha=0.7,
                label="Toxic", color="#e74c3c")
        ax.set_xlim(0, 500)
        ax.set_xlabel("Characters")
        ax.set_ylabel("Frequency")
        ax.legend()
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.subheader("Top 20 Most Frequent Words in Toxic Comments")
    toxic_words = " ".join(df[df["toxic"]==1]["clean_text"].values)
    word_counts = Counter(toxic_words.split()).most_common(20)
    words, freqs = zip(*word_counts)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(list(words)[::-1], list(freqs)[::-1], color="#e74c3c", edgecolor="white")
    ax.set_xlabel("Frequency")
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ── Page: Model Info ───────────────────────────────────────────────────────────
elif page == "ℹ️ Model Info":
    st.header("ℹ️ Model Information & Metrics")

    info = call_api_model_info()
    if "error" in info:
        st.error(f"Could not reach backend: {info['error']}")
        st.info("Start the Flask backend with: `python backend/app.py`")
    else:
        m = info.get("metrics", {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy",  f"{m.get('accuracy', 0):.2%}")
        c2.metric("Precision", f"{m.get('precision', 0):.2%}")
        c3.metric("Recall",    f"{m.get('recall', 0):.2%}")
        c4.metric("F1-Score",  f"{m.get('f1_score', 0):.2%}")

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Model Details")
            st.json({
                "Model":      info.get("model"),
                "Vectorizer": info.get("vectorizer"),
                "Train set":  f"{info.get('training_samples'):,} samples",
                "Test set":   f"{info.get('test_samples'):,} samples",
            })
        with col2:
            st.subheader("Confusion Matrix")
            cm = info.get("confusion_matrix", [[0,0],[0,0]])
            fig, ax = plt.subplots(figsize=(4, 3.5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=["Non-Toxic","Toxic"],
                        yticklabels=["Non-Toxic","Toxic"],
                        linewidths=0.5, ax=ax)
            ax.set_ylabel("True")
            ax.set_xlabel("Predicted")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
