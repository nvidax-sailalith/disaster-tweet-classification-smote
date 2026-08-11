import os
import re
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout

def main():
    print("=== Starting Disaster Tweet Classification (SMOTE & Bi-LSTM) ===")
    
    # 1. Load Datasets
    train_path = 'data/train_1.csv'
    test_path = 'data/test_1.csv'
    
    if not os.path.exists(train_path):
        train_path = 'C:/Users/lalit/Downloads/train _1.csv'
        test_path = 'C:/Users/lalit/Downloads/test _1.csv'

    print(f"Loading train data from: {train_path}")
    print(f"Loading test data from: {test_path}")

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    print(f"Train dataset shape: {train_data.shape}")
    print(f"Test dataset shape: {test_data.shape}")

    # 2. Handle missing values
    for df in [train_data, test_data]:
        df['keyword'] = df['keyword'].fillna('')
        df['location'] = df['location'].fillna('')

    # 3. Text cleaning function
    def clean_text(text):
        if not isinstance(text, str):
            return ""
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text.lower().strip()

    train_data['text_clean'] = train_data['text'].apply(clean_text)
    test_data['text_clean'] = test_data['text'].apply(clean_text)

    # 4. Generate Exploratory Data Analysis (EDA) Plots
    os.makedirs('plots', exist_ok=True)
    print("Generating EDA Visualizations (WordClouds, Tweet Length Distribution, Top Keywords)...")

    # EDA 1: Word Clouds for Disaster and Non-Disaster Tweets
    disaster_tweets = train_data[train_data['target'] == 1]['text_clean']
    non_disaster_tweets = train_data[train_data['target'] == 0]['text_clean']

    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    wordcloud_disaster = WordCloud(width=800, height=400, background_color='white').generate(' '.join(disaster_tweets))
    plt.imshow(wordcloud_disaster, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud - Disaster Tweets', fontsize=14, pad=10)

    plt.subplot(1, 2, 2)
    wordcloud_non_disaster = WordCloud(width=800, height=400, background_color='white').generate(' '.join(non_disaster_tweets))
    plt.imshow(wordcloud_non_disaster, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud - Non-Disaster Tweets', fontsize=14, pad=10)
    plt.tight_layout()
    plt.savefig('plots/wordclouds.png', dpi=300)
    plt.close()

    # EDA 2: Tweet Length Distribution
    train_data['tweet_length'] = train_data['text'].astype(str).apply(len)
    plt.figure(figsize=(10, 6))
    sns.histplot(train_data[train_data['target'] == 1]['tweet_length'], color='red', label='Disaster Tweets', kde=True, stat="density", element="step")
    sns.histplot(train_data[train_data['target'] == 0]['tweet_length'], color='green', label='Non-Disaster Tweets', kde=True, stat="density", element="step")
    plt.xlabel('Tweet Length')
    plt.ylabel('Density')
    plt.title('Tweet Length Distribution')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/tweet_length_distribution.png', dpi=300)
    plt.close()

    # EDA 3: Top 10 Keywords (Disaster & Non-Disaster)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    disaster_keywords = train_data[train_data['target'] == 1]['keyword'].replace('', np.nan).dropna().value_counts().head(10)
    disaster_keywords.sort_values().plot(kind='barh', color='red')
    plt.title('Top 10 Disaster Keywords')
    plt.xlabel('Count')
    plt.ylabel('keyword')

    plt.subplot(1, 2, 2)
    nondisaster_keywords = train_data[train_data['target'] == 0]['keyword'].replace('', np.nan).dropna().value_counts().head(10)
    nondisaster_keywords.sort_values().plot(kind='barh', color='green')
    plt.title('Top 10 Non-Disaster Keywords')
    plt.xlabel('Count')
    plt.ylabel('keyword')
    plt.tight_layout()
    plt.savefig('plots/top_keywords.png', dpi=300)
    plt.close()

    # Individual plot for Top 10 Disaster Keywords
    plt.figure(figsize=(6, 6))
    disaster_keywords.sort_values().plot(kind='barh', color='red')
    plt.title('Top 10 Disaster Keywords')
    plt.xlabel('Count')
    plt.ylabel('keyword')
    plt.tight_layout()
    plt.savefig('plots/top_10_disaster_keywords.png', dpi=300)
    plt.close()

    # Individual plot for Top 10 Non-Disaster Keywords
    plt.figure(figsize=(6, 6))
    nondisaster_keywords.sort_values().plot(kind='barh', color='green')
    plt.title('Top 10 Non-Disaster Keywords')
    plt.xlabel('Count')
    plt.ylabel('keyword')
    plt.tight_layout()
    plt.savefig('plots/top_10_nondisaster_keywords.png', dpi=300)
    plt.close()

    # 5. Combine text with keyword and location
    train_data['combined_text'] = train_data['text_clean'] + ' ' + train_data['keyword'] + ' ' + train_data['location']
    test_data['combined_text'] = test_data['text_clean'] + ' ' + test_data['keyword'] + ' ' + test_data['location']

    # 6. Tokenization and Padding
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_data['combined_text'])
    max_length = 100
    
    X = pad_sequences(tokenizer.texts_to_sequences(train_data['combined_text']), maxlen=max_length, padding='post')
    X_test = pad_sequences(tokenizer.texts_to_sequences(test_data['combined_text']), maxlen=max_length, padding='post')
    y = train_data['target'].values

    print(f"Vocabulary size: {len(tokenizer.word_index) + 1}")
    print(f"Original class distribution: {np.bincount(y)}")

    # 7. Apply SMOTE for Class Balancing
    print("Applying SMOTE for class balancing...")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    print(f"Resampled class distribution: {np.bincount(y_resampled)}")

    # 8. Train/Validation Split
    X_train, X_val, y_train, y_val = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
    print(f"X_train shape: {X_train.shape}, X_val shape: {X_val.shape}")

    # 9. Build Model Architecture
    vocab_size = len(tokenizer.word_index) + 1
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=128, input_length=max_length),
        Bidirectional(LSTM(units=128, dropout=0.2, recurrent_dropout=0.2)),
        Dense(units=64, activation='relu'),
        Dropout(0.3),
        Dense(units=1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    # 10. Model Training
    print("Training Bidirectional LSTM model for 10 epochs...")
    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=32,
        validation_data=(X_val, y_val),
        verbose=1
    )

    # 11. Model Evaluation
    y_val_prob = model.predict(X_val)
    y_val_pred = (y_val_prob > 0.5).astype(int).reshape(-1)
    
    val_f1 = f1_score(y_val, y_val_pred)
    print(f"\n==========================================")
    print(f"Validation F1 Score: {val_f1:.4f}")
    print(f"==========================================\n")
    print("Classification Report:")
    print(classification_report(y_val, y_val_pred, target_names=['Non-Disaster (0)', 'Disaster (1)']))

    # 12. Save Predictions for Test Set
    test_probs = model.predict(X_test)
    test_preds = (test_probs > 0.5).astype(int).reshape(-1)
    
    submission = pd.DataFrame({'id': test_data['id'], 'target': test_preds})
    submission.to_csv('submission.csv', index=False)
    print("Successfully saved predictions to 'submission.csv'")

    # 13. Save Trained Model & Tokenizer artifacts
    os.makedirs('saved_model', exist_ok=True)
    model.save('saved_model/disaster_lstm_model.keras')
    with open('saved_model/tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Saved trained model and tokenizer to 'saved_model/' directory.")

    # 14. Save Model Performance Visualizations
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy', color='#3b82f6', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Val Accuracy', color='#10b981', linewidth=2)
    plt.title('Model Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss', color='#ef4444', linewidth=2)
    plt.plot(history.history['val_loss'], label='Val Loss', color='#f59e0b', linewidth=2)
    plt.title('Model Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/training_curves.png', dpi=300)
    plt.close()

    # Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(confusion_matrix(y_val, y_val_pred), annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-Disaster', 'Disaster'], yticklabels=['Non-Disaster', 'Disaster'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Validation Confusion Matrix')
    plt.tight_layout()
    plt.savefig('plots/confusion_matrix.png', dpi=300)
    plt.close()

    print("Saved all evaluation and EDA plots to 'plots/' directory.")
    print("=== Pipeline Execution Complete! ===")

if __name__ == '__main__':
    main()
