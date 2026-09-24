# 🛡️ Toxic Comment Detection — Brazilian Portuguese

**Author:** Rimi Das  
**Project:** IBM AI/ML Submission — Toxic Comment Classification  
**Language:** Brazilian Portuguese (pt-BR)  
**Model:** Logistic Regression + TF-IDF

---

## 📋 Project Overview

This project builds an end-to-end **toxic comment detection system** for Brazilian Portuguese social-media text.  
It includes:
- An **ML pipeline** (data cleaning → TF-IDF → Logistic Regression) achieving **76.95% accuracy** and **74.44% F1-score**
- A **Flask REST API** for real-time predictions
- A **Streamlit frontend** for interactive UI and EDA charts

---

## 📂 Project Structure

```
toxic_comment_detection/
├── data/
│   └── comentarios_toxicos_ptBR.csv   # Dataset (29,686 cleaned rows)
├── backend/
│   ├── train_model.py                 # ML training script
│   └── app.py                         # Flask REST API
├── frontend/
│   └── ui.py                          # Streamlit web UI
├── model/
│   ├── toxic_model.pkl                # Trained Logistic Regression model
│   ├── vectorizer.pkl                 # Fitted TF-IDF vectorizer
│   └── metrics.json                   # Evaluation metrics
├── report_images/                     # EDA + evaluation charts (PNG)
├── RimiDas_ToxicCommentDetection.ipynb  # Jupyter notebook (submission)
├── RimiDas_ProjectReport.docx           # Full project report (submission)
├── requirements.txt                     # Python dependencies (submission)
└── README.md                            # This file (submission)
```

---

## 📊 Dataset

| Property | Value |
|---|---|
| File | `comentarios_toxicos_ptBR.csv` |
| Rows (cleaned) | 29,686 |
| Non-toxic (0) | 16,286 (54.9%) |
| Toxic (1) | 13,400 (45.1%) |
| Columns | `text`, `text_norm`, `toxic` |
| Language | Brazilian Portuguese (pt-BR) |

---

## 🤖 Model Performance (Real Results)

| Metric | Score |
|---|---|
| **Accuracy** | **76.95%** |
| **Precision** | **74.49%** |
| **Recall** | **74.40%** |
| **F1-Score** | **74.44%** |
| Train samples | 23,748 |
| Test samples | 5,938 |

**Confusion Matrix:**

|  | Predicted Non-Toxic | Predicted Toxic |
|---|---|---|
| **Actual Non-Toxic** | 2,575 ✅ | 683 ❌ |
| **Actual Toxic** | 686 ❌ | 1,994 ✅ |

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python backend/train_model.py
```
This saves `model/toxic_model.pkl`, `model/vectorizer.pkl`, `model/metrics.json` and charts to `report_images/`.

### 3. Start the Flask Backend

```bash
python backend/app.py
```
API will be available at `http://localhost:5000`.

### 4. Launch the Streamlit Frontend

```bash
streamlit run frontend/ui.py
```
Open `http://localhost:8501` in your browser.

### 5. Run the Jupyter Notebook

```bash
jupyter notebook RimiDas_ToxicCommentDetection.ipynb
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service health check |
| `GET` | `/model-info` | Model metadata + evaluation metrics |
| `POST` | `/predict` | Single comment prediction |
| `POST` | `/predict-batch` | Batch comment predictions |

### Example Request (cURL)

```bash
curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"comment": "Cala a boca seu burro inutil"}'
```

### Example Response

```json
{
  "comment": "Cala a boca seu burro inutil",
  "label": "Toxic",
  "prediction": 1,
  "probability": {
    "non_toxic": 0.0627,
    "toxic": 0.9373
  }
}
```

---

## 🔧 Technologies Used

| Layer | Technology |
|---|---|
| Language | Python 3.x |
| ML / NLP | scikit-learn, TF-IDF, Logistic Regression |
| Data | pandas, numpy |
| Visualisation | matplotlib, seaborn, wordcloud |
| Backend API | Flask |
| Frontend | Streamlit |
| Notebook | Jupyter / nbformat |
| Persistence | joblib, JSON |
| Report | python-docx |

---

## 📁 Submission Files

| File | Description |
|---|---|
| `RimiDas_ToxicCommentDetection.ipynb` | Code notebook (data loading → training → evaluation) |
| `requirements.txt` | Python dependency list |
| `RimiDas_ProjectReport.docx` | Full project report with charts and metrics |
| `README.md` | This file |

---

## 📝 Notes

- The dataset is in Brazilian Portuguese; a custom Portuguese stopword list is used (no NLTK download required)
- The model uses `class_weight='balanced'` to handle class imbalance
- All metrics in the report are real outputs from the trained model — nothing is fabricated
- The Streamlit UI requires the Flask backend to be running for the Model Info page; other pages work standalone

---

*Generated for IBM AI/ML Project Submission — Rimi Das*
