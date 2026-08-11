# Disaster Tweet Classification using SMOTE & Bidirectional LSTM

A deep learning project that classifies tweets into **Disaster-related** or **Non-Disaster** categories using Natural Language Processing (NLP), **SMOTE** (Synthetic Minority Over-sampling Technique) for class balancing, and a **Bidirectional LSTM** neural network.

---

## 📌 Project Overview
During natural disasters and emergencies, social media (especially Twitter) is a critical source of real-time information. However, many tweets use disaster-related keywords metaphorically (e.g., *"The new album is fire!"*). This project aims to accurately distinguish between actual emergency disaster alerts and normal conversation tweets.

---

## 📊 Exploratory Data Analysis (EDA) Visualizations

### 1. Word Clouds (Disaster vs Non-Disaster Tweets)
![Word Clouds](plots/wordclouds.png)

### 2. Tweet Length Distribution
![Tweet Length Distribution](plots/tweet_length_distribution.png)

### 3. Top 10 Keywords (Disaster & Non-Disaster)
![Top 10 Keywords](plots/top_keywords.png)

---

## 🛠️ Features & Architecture
1. **Preprocessing & Cleaning**: Strips URLs, HTML tags, special characters, and handles missing keyword/location metadata.
2. **Class Imbalance Handling**: Employs **SMOTE** (`imblearn.over_sampling.SMOTE`) to balance minority and majority classes.
3. **Deep Learning Model**:
   - **Embedding Layer**: 128 dimensions (`max_length=100`)
   - **Bidirectional LSTM Layer**: 128 units with dropout (`0.2`)
   - **Dense Layers**: 64 units (ReLU) + Dropout (`0.3`) -> 1 unit (Sigmoid)
4. **Validation & Metrics**: Evaluated using F1 Score, Confusion Matrix, and Loss/Accuracy curves.

---

## 📈 Model Performance & Results
- **Validation F1 Score**: **0.7290 (72.9%)**
- **Test Predictions**: Saved in `submission.csv` (3,263 test tweets).

### Live Inference Test Examples:
- **Tweet**: *"Forest fire near Rocky Mountain National Park! Evacuate immediately!"*
  - **Prediction**: 🚨 **ALERT: DISASTER TWEET** (Probability: `99.78%`)
- **Tweet**: *"Just enjoying a hot cup of coffee on a rainy Sunday morning."*
  - **Prediction**: ✅ **NORMAL TWEET** (Disaster Probability: `14.01%`)

---

## 📁 Repository Structure
```
disaster-tweet-classification-smote/
├── data/
│   ├── train_1.csv
│   └── test_1.csv
├── plots/
│   ├── wordclouds.png
│   ├── tweet_length_distribution.png
│   ├── top_keywords.png
│   ├── top_10_disaster_keywords.png
│   ├── top_10_nondisaster_keywords.png
│   ├── training_curves.png
│   └── confusion_matrix.png
├── saved_model/
│   └── tokenizer.pickle
├── disaster_tweet_classification_smote.py
├── predict_tweet.py
├── submission.csv
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Model & Generate EDA Plots
```bash
python disaster_tweet_classification_smote.py
```

### 3. Run Tweet Predictions
```bash
python predict_tweet.py
```
