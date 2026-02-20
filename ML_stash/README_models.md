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
Run the specific analysis script you need:

**1. Voice Fatigue Detection:**
```bash
python ML_stash/VoiceFatigue/detect_voice_fatigue.py
```

**2. Cognitive Decline Screening:**
```bash
python ML_stash/CognitiveDecline/detect_cognitive_decline.py
```

**3. Depression Screening:**
```bash
python ML_stash/DepressionScreening/detect_depression.py
```

### Step C: View Results
Once the script finishes, check the **Output Folder**:
> `ML_stash/audio_output/`

You will find a generated Markdown report (e.g., `myfile_fatigue_report.md`) for every audio file processed.

---

## Folder Structure
- **`audio_input/`**: Drop your MP3/WAV files here.
- **`audio_output/`**: Reports appear here.
- **`VoiceFatigue/`**: Engine for fatigue detection.
- **`CognitiveDecline/`**: Engine for cognitive screening.
- **`DepressionScreening/`**: Engine for depression screening.
