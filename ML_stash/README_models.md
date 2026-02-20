# Health Screening ML Models - Testing Guide

This directory (`ML_stash`) contains prototype scripts for detecting health conditions from voice data.

## 1. Environment Setup

**Step 1:** Open your terminal in the project root: `C:\Users\xiang\Synapxe`

**Step 2:** Locate your Python Virtual Environment
We need to use the specific Python installation where the libraries are installed.
- On Windows (PowerShell), use: `.\.venv\Scripts\python.exe`

**Step 3:** Install Dependencies (if not already done)
```powershell
.\.venv\Scripts\python.exe -m pip install -r ML_stash/requirements.txt
```

---

## 2. Generating Test Data

Generate the dummy audio file using the virtual environment python:

```powershell
.\.venv\Scripts\python.exe ML_stash/generate_audio.py
```

**Expected Output:**
> sample.wav created successfully.

---

## 3. Testing the Models

### A. Voice Fatigue Detection
**Run:**
```powershell
.\.venv\Scripts\python.exe ML_stash/VoiceFatigue/detect_voice_fatigue.py
```
**Expected Output:**
> Prediction: Fatigued (or Normal)
> Fatigue Probability: 1.0000

### B. Cognitive Decline Screening
**Run:**
```powershell
.\.venv\Scripts\python.exe ML_stash/CognitiveDecline/detect_cognitive_decline.py
```
**Expected Output:**
> Prediction: Healthy (or Cognitive Decline Risk)
> Risk Probability: 0.3005

### C. Depression Screening
**Run:**
```powershell
.\.venv\Scripts\python.exe ML_stash/DepressionScreening/detect_depression.py
```
**Expected Output:**
> Prediction: Depression Risk (or Normal Range)
> Depression Probability Score: 0.6200

---

## Troubleshooting
- **"ModuleNotFoundError"**: This happens if you use the wrong `python` command. Always use `.\.venv\Scripts\python.exe` instead of just `python`.
