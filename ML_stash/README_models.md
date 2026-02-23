# Health Screening ML Analysis - User Guide

This directory (`ML_stash`) contains the analysis engine for screening health conditions from voice data.

## 1. Environment Setup (First Run Only)

**Step 1:** Activate the environment.
In terminal, process root, run:
```cmd
activate
```
*(This uses the `activate.bat` in the root folder)*

**Step 2:** Install libraries.
```bash
pip install -r ML_stash/requirements.txt
```

---

## 2. How to Use

### Step A: Add Audio Files
Place any audio files you want to analyze (`.mp3`, `.wav`, or `.flac`) into the **Input Folder**:
> `ML_stash/audio_input/`

### Step B: Run Analysis
Run the main reporting script to analyze all files for all conditions at once:
```bash
python ML_stash/run_full_analysis.py
```

This will generate a **Combined_Health_Report.md** in the output folder.

---

## 3. Methodology & Architecture

The system uses a **pipelined architecture** where raw audio is processed by specialized feature extractors before classification.

### A. Pre-Processing & Feature Extraction
We use state-of-the-art pre-trained models to convert audio into numerical embeddings. These models are "frozen" (we do not retrain them), ensuring robust feature detection even with small datasets.

| Condition | Feature Extractor | Source Model | Why? |
| :--- | :--- | :--- | :--- |
| **Voice Fatigue** | **ECAPA-TDNN** | `speechbrain/spkrec-ecapa-voxceleb` | Industry standard for speaker verification; excellent at capturing prolonged vocal strain and timbre changes. |
| **Cognitive Decline** | **HuBERT** | `facebook/hubert-base-ls960` | Self-supervised model that learns hidden speech units. Effectively captures pauses, slurring, and articulation issues common in MCI. |
| **Depression** | **Prosodic Features** | `Librosa` (Signal Processing) | Direct calculation of Pitch (Chroma), Loudness (RMS), and Timbre (Spectral Centroid) to detect "flat affect" (monotone/quiet speech). |

### B. Classification (The "Brain")
Once features are extracted:
1.  **Input:** The numerical vectors from Step A (e.g., a 192-dimensional vector from ECAPA-TDNN).
2.  **Classifier:** A lightweight classifier (MLP Neural Network or Logistic Regression) makes the final decision.
    *   *Note:* Currently, these classifiers are **untrained prototypes** initialized with random weights, as clinical datasets (DementiaBank/DAIC-WOZ) require research approval.
    *   *To Make Real:* You must obtain labeled `.wav` files and run a training script to "teach" this layer the difference between Healthy and Sick.

### C. Output Generation
The system aggregates the confidence scores from all three models into a single Markdown report (`Combined_Health_Report.md`) for easy review by clinicians.

---

## Folder Structure
- **`audio_input/`**: Drop your MP3/WAV files here.
- **`audio_output/`**: Reports appear here.
- **`VoiceFatigue/`**: Engine for fatigue detection.
- **`CognitiveDecline/`**: Engine for cognitive screening.
- **`DepressionScreening/`**: Engine for depression screening.
