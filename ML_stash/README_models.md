# Health Screening ML Models

This directory contains prototype scripts for detecting health conditions from voice data.

## Setup

1.  Make sure you have Python installed (3.8+ recommended).
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

First, generate a sample audio file if one doesn't exist:
```bash
python generate_audio.py
```
This creates `sample.wav` in this directory.

### 1. Voice Fatigue Detection
Uses `speechbrain` (ECAPA-TDNN) to extract embeddings and a classifier to detect fatigue.
```bash
python VoiceFatigue/detect_voice_fatigue.py
```

### 2. Cognitive Decline Screening
Uses `transformers` (HuBERT) to extract embeddings for cognitive assessment.
```bash
python CognitiveDecline/detect_cognitive_decline.py
```

### 3. Depression Screening
Uses `librosa` and `pyAudioAnalysis` for acoustic feature extraction.
```bash
python DepressionScreening/detect_depression.py
```
