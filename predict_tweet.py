import re
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.lower().strip()

def predict_disaster_tweet(tweet_text, keyword="", location="", model_path="saved_model/disaster_lstm_model.keras", tokenizer_path="saved_model/tokenizer.pickle"):
    # Load model and tokenizer
    model = tf.keras.models.load_model(model_path)
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)
        
    # Preprocess
    cleaned = clean_text(tweet_text)
    combined = f"{cleaned} {keyword} {location}".strip()
    
    # Tokenize and Pad
    seq = tokenizer.texts_to_sequences([combined])
    padded = pad_sequences(seq, maxlen=100, padding='post')
    
    # Predict probability
    prob = float(model.predict(padded, verbose=0)[0][0])
    is_disaster = prob > 0.5
    
    return {
        "tweet": tweet_text,
        "is_disaster": is_disaster,
        "confidence_score": round(prob if is_disaster else 1 - prob, 4),
        "disaster_probability": round(prob, 4)
    }

if __name__ == "__main__":
    test_tweets = [
        "Forest fire near Rocky Mountain National Park! Evacuate immediately!",
        "Just enjoying a hot cup of coffee on a rainy Sunday morning."
    ]
    
    print("=== Testing Sample Tweet Inference ===")
    for text in test_tweets:
        res = predict_disaster_tweet(text)
        status = "ALERT: DISASTER TWEET" if res["is_disaster"] else "NORMAL TWEET"
        print(f"\nTweet: \"{text}\"")
        print(f"Prediction: {status} (Prob: {res['disaster_probability']:.2%}, Confidence: {res['confidence_score']:.2%})")
