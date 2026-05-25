# 📧 Email Spam Detection System

A machine learning-based email/SMS spam classification system using TF-IDF vectorization and Multinomial Naive Bayes. The model is deployed using Streamlit for real-time predictions.

---

## 📌 Project Overview

This project classifies messages as **Spam** or **Not Spam (Ham)** using a supervised machine learning approach. The system transforms raw text into numerical features using TF-IDF and applies a Multinomial Naive Bayes classifier for prediction.

---

## 🧠 Machine Learning Pipeline

1. Input text message
2. TF-IDF vectorization
3. Multinomial Naive Bayes classification
4. Output prediction with confidence score

---

## 🛠️ Tech Stack

* Python
* Streamlit
* Scikit-learn
* TF-IDF Vectorizer
* Multinomial Naive Bayes
* Joblib

---

## 📂 Project Structure

```

spam-classifier-nlp/
│
├── app.py                  # Streamlit web app
├── Model.joblib           # Trained Naive Bayes model
├── Vectorizer.joblib      # TF-IDF vectorizer
├── requirements.txt       # Dependencies

```

---

## 📊 Model Performance

* Accuracy: ~98%
* Precision: ~99%+
* Dataset: SMS/Email labeled dataset

---

## ▶️ How to Run

### 1. Clone repository

```bash
git clone https://github.com/imranbuttcodes/spam-classifier-nlp.git
cd spam-classifier-nlp
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Streamlit app

```bash
streamlit run app.py
```

---

## 📦 Model Details

* Algorithm: Multinomial Naive Bayes
* Feature Extraction: TF-IDF
* Task: Binary Text Classification

---
