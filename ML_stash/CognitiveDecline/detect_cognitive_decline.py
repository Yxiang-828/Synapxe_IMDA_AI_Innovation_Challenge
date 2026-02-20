import torch
import librosa
from transformers import HubertModel, Wav2Vec2FeatureExtractor
import os
import numpy as np
from sklearn.neural_network import MLPClassifier

def extract_hubert_embeddings(audio_file):
    print(f"Loading HuBERT model for {audio_file}...")
    model_name = "facebook/hubert-base-ls960"
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    model = HubertModel.from_pretrained(model_name)
    
    # Load audio
    y, sr = librosa.load(audio_file, sr=16000)
    
    # Process audio
    inputs = feature_extractor(y, sampling_rate=sr, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get last hidden state and mean pool over time
    last_hidden_state = outputs.last_hidden_state
    embedding = torch.mean(last_hidden_state, dim=1).squeeze().numpy()
    
    return embedding

def main():
    audio_file = os.path.join(os.path.dirname(__file__), "../sample.wav")
    
    if not os.path.exists(audio_file):
        print(f"Error: {audio_file} not found. Key: Please run generate_audio.py first.")
        return

    print("Extracting embeddings (HuBERT)...")
    try:
        # Note: HuBERT model is large, this might take a moment to download/load
        features = extract_hubert_embeddings(audio_file)
        print(f"Embeddings extracted successfully. Shape: {features.shape}")
        
        # --- Mock Classifier ---
        # Train a small MLP to simulate cognitive decline detection
        print("\n--- Simulating Classifier Training ---")
        X_dummy = np.random.rand(50, features.shape[0])
        y_dummy = np.random.randint(0, 2, 50) # 0=Healthy, 1=Cognitive Decline
        
        clf = MLPClassifier(hidden_layer_sizes=(64,), max_iter=500)
        clf.fit(X_dummy, y_dummy)
        print("Classifier trained on dummy data.")
        
        # --- Inference ---
        print("\n--- Running Inference ---")
        prediction = clf.predict([features])[0]
        probability = clf.predict_proba([features])[0][1]
        
        label = "Cognitive Decline Risk" if prediction == 1 else "Healthy"
        print(f"Prediction: {label}")
        print(f"Risk Probability: {probability:.4f}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
