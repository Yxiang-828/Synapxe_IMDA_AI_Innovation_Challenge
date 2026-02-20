import librosa
import numpy as np
import os
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

def extract_acoustic_features(audio_file):
    print(f"Extracting acoustic features for {audio_file}...")
    y, sr = librosa.load(audio_file)
    
    # Extract features relevant to depression (prosody, spectral, voice quality)
    # 1. MFCCs (Mel-frequency cepstral coefficients)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    
    # 2. Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    
    # 3. Spectral features
    spec_cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    spec_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    
    # Combine into a single feature vector
    features = np.concatenate([
        mfcc_mean, 
        chroma_mean, 
        [spec_cent, spec_bw, spec_rolloff, zcr]
    ])
    
    return features

def main():
    audio_file = os.path.join(os.path.dirname(__file__), "../sample.wav")
    
    if not os.path.exists(audio_file):
        print(f"Error: {audio_file} not found. Please run generate_audio.py first.")
        return

    print("Extracting features (Acoustic)...")
    try:
        features = extract_acoustic_features(audio_file)
        print(f"Features extracted successfully. Vector length: {len(features)}")
        
        # --- Mock Classifier ---
        # Simulating a depression screening model (SVM/Random Forest)
        print("\n--- Simulating Classifier Training ---")
        X_dummy = np.random.rand(100, len(features))
        y_dummy = np.random.randint(0, 2, 100) # 0=Low Risk, 1=High Risk
        
        # Using Random Forest as it's robust for tabular acoustic features
        clf = RandomForestClassifier(n_estimators=100)
        clf.fit(X_dummy, y_dummy)
        print("Classifier trained on dummy data.")
        
        # --- Inference ---
        print("\n--- Running Inference ---")
        prediction = clf.predict([features])[0]
        
        # Soft voting or probability
        probability = clf.predict_proba([features])[0][1]
        
        label = "Depression Risk" if prediction == 1 else "Normal Range"
        print(f"Prediction: {label}")
        print(f"Depression Probability Score: {probability:.4f}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
