import torch
import torchaudio
from speechbrain.inference.classifiers import EncoderClassifier
import os
import numpy as np
from sklearn.linear_model import LogisticRegression

import soundfile as sf

def extract_features(audio_file):
    print(f"Loading ECAPA-TDNN model for {audio_file}...")
    # Using the pre-trained ECAPA-TDNN model from generic speaker recognition
    # In a real scenario, this would be fine-tuned on fatigue datasets.
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
    
    # Load audio using soundfile directly to avoid torchaudio backend issues
    signal_np, fs = sf.read(audio_file)
    # Convert to tensor and ensure shape (1, T) or (Batch, T)
    signal = torch.from_numpy(signal_np).float()
    
    # If stereo, mix down or take one channel (ECAPA expects mono usually)
    if len(signal.shape) > 1 and signal.shape[1] > 1:
        signal = signal.mean(dim=1) # mix to mono
    
    if len(signal.shape) == 1:
        signal = signal.unsqueeze(0) # (1, T)
        
    # Resample if needed. ECAPA expects 16k usually.
    if fs != 16000:
        import torchaudio.transforms as T
        resampler = T.Resample(fs, 16000)
        signal = resampler(signal)
        
    embeddings = classifier.encode_batch(signal)
    
    # Flatten the embedding to a 1D array
    return embeddings.squeeze().detach().numpy()

def main():
    audio_file = os.path.join(os.path.dirname(__file__), "../sample.wav")
    
    if not os.path.exists(audio_file):
        print(f"Error: {audio_file} not found. Please run generate_audio.py first.")
        return

    print("Extracting features (ECAPA-TDNN embeddings)...")
    try:
        features = extract_features(audio_file)
        print(f"Features extracted successfully. Shape: {features.shape}")
        
        # --- Mock Classifier ---
        # Since we don't have a pre-trained fatigue classifier, we'll train a dummy one
        # to demonstrate the pipeline structure.
        print("\n--- Simulating Classifier Training ---")
        X_dummy = np.random.rand(100, features.shape[0]) # 100 dummy samples
        y_dummy = np.random.randint(0, 2, 100) # Binary labels: 0=Normal, 1=Fatigued
        
        clf = LogisticRegression()
        clf.fit(X_dummy, y_dummy)
        print("Classifier trained on dummy data.")
        
        # --- Inference ---
        print("\n--- Running Inference ---")
        prediction = clf.predict([features])[0]
        probability = clf.predict_proba([features])[0][1]
        
        label = "Fatigued" if prediction == 1 else "Normal"
        print(f"Prediction: {label}")
        print(f"Fatigue Probability: {probability:.4f}")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
