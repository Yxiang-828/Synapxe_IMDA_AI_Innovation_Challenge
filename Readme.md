# ML Tech Stack Recommendation: Chronic Disease Management AI
## NUS-SYNAPXE-IMDA AI Innovation Challenge 2026

---

## Executive Summary

Based on your proposed solution architecture—using MERaLiON and SEA-LION as conversational interfaces to collect multimodal health data through voice and video interactions—I've researched and designed a complementary ML tech stack that aligns with Singapore's healthcare context, the challenge requirements, and practical deployment constraints.

**Key Design Principles:**
1. **Privacy-first**: On-device inference where possible, minimal data transmission
2. **Lightweight**: Models that run efficiently on mobile devices
3. **Culturally appropriate**: Works with Singapore's multilingual, multi-ethnic population
4. **Clinically validated**: Based on peer-reviewed research and open-source implementations
5. **Modular**: Each component can be developed, tested, and deployed independently

---

## Understanding Your Architecture

Before diving into the stack, let me map how I understand your proposed flow:

```
State 0 (Default): LLM prompts user for chat → User accepts → State 1
State 1 (Voice Chat): Bidirectional voice conversation → Build rapport + collect audio data
                    → Random trigger → Camera request → Category A (camera on) or B (camera off)
State 2 (Backend): ML models process collected data → Update LLM context → Health alerts if needed
```

This is a **multimodal sensing pipeline** with the LLM as the orchestrator. The ML stack I'm recommending will power the "acquire ML data" and "acquire result" steps in your State 2.

---

## Proposed ML Tech Stack

### 1. Audio/Voice Analysis Pipeline

**Purpose**: Extract health indicators from voice during natural conversations

#### 1.1 Speech Emotion Recognition (SER)
**Recommended Model**: **Emotion2Vec** (open-source, self-supervised)
- **Why this model**: Specifically designed for emotion recognition from speech; outperforms wav2vec 2.0 and HuBERT on emotion tasks; supports multilingual contexts
- **What it detects**: Anger, happiness, neutral, sadness, surprise, disgust, fear
- **Health relevance**: Persistent sadness/depression indicators, stress levels, emotional well-being
- **Implementation**: Available on HuggingFace (`Emotion2Vec/emotion2vec_plus_large`)
- **Inference**: Can run on-device with quantization (~50MB model)

**Alternative**: **SenseVoice-Small** (Alibaba, open-source)
- Multilingual support including Mandarin, Cantonese, English
- Very low latency
- Good accuracy for emotion recognition

#### 1.2 Voice Fatigue Detection
**Recommended Approach**: **ECAPA-TDNN-based classifier**
- **Why**: Research shows 93% accuracy in detecting voice fatigue; uses embeddings that capture speaker characteristics over time
- **Health relevance**: Fatigue is a key indicator of chronic condition management, medication side effects, sleep quality issues
- **Implementation**: Pre-trained ECAPA-TDNN from SpeechBrain + lightweight CNN classifier
- **Features**: Fundamental frequency (F0), jitter, shimmer, HNR, MFCCs

#### 1.3 Cognitive Decline Screening (Optional but valuable)
**Recommended Approach**: **HuBERT embeddings + lightweight classifier**
- **Why**: Research from 2025 ICASSP PROCESS Challenge shows HuBERT embeddings + LLM-derived features achieve 55% F1 for dementia/MCI detection (top-20 globally)
- **Health relevance**: Early detection of mild cognitive impairment, especially relevant for elderly chronic patients
- **Implementation**: Extract HuBERT embeddings from conversational speech, classify with small MLP or LSTM
- **Note**: This requires longer speech samples (30+ seconds of continuous speech)

#### 1.4 Depression Screening (Optional)
**Recommended Approach**: **Acoustic feature extraction + SVM/Random Forest**
- **Why**: Research shows 91.67% accuracy for women, 80% for men in detecting depression from voice
- **Features**: 68 acoustic features (MFCCs, chroma, energy, ZCR) using PyAudioAnalysis
- **Health relevance**: Depression is common comorbidity with chronic diseases
- **Implementation**: Lightweight, rule-based triggering for deeper screening

---

### 2. Video/Visual Analysis Pipeline

**Purpose**: Extract physiological signals and behavioral indicators when camera is available

#### 2.1 Remote Photoplethysmography (rPPG) - Heart Rate Estimation
**Recommended Model**: **TS-CAN** or **EfficientPhys** via **open-rppg toolbox**
- **Why**: TS-CAN achieved 1.07 BPM MAE in research; EfficientPhys uses transformers for better robustness; open-rppg provides unified interface
- **Health relevance**: Heart rate variability (HRV) indicates stress, cardiovascular health, autonomic nervous system function
- **Implementation**: 
  ```python
  import rppg
  model = rppg.Model()  # defaults to FacePhys.rlap
  results = model.process_video("user_video.mkv")
  hr = results['hr']  # BPM
  ```
- **Requirements**: 10-30 seconds of face-visible video, reasonable lighting
- **Limitations**: Struggles with low light and high heart rates (>120 BPM) based on research

**Alternative Classical Methods**: GREEN, POS, CHROM algorithms
- More robust to low light
- No training required
- Slightly lower accuracy but more reliable across conditions

#### 2.2 Respiration Rate Estimation
**Recommended Approach**: **rPPG-derived respiration** or **chest movement analysis**
- **Why**: Respiration information is embedded in rPPG signals; research shows feasibility with weakly supervised learning
- **Health relevance**: Respiratory rate is a vital sign; changes indicate distress, sleep apnea, COPD exacerbation
- **Implementation**: Extract respiratory component from rPPG signal (0.1-0.5 Hz band)
- **Alternative**: Use pose estimation to track chest/shoulder movement

#### 2.3 Facial Emotion Recognition
**Recommended Model**: **Py-Feat** or **DeepFace** (emotion module)
- **Why**: Py-Feat is open-source, research-grade, provides Action Units + emotions; DeepFace is easier to implement
- **What it detects**: 7 basic emotions + 20+ Facial Action Units
- **Health relevance**: Emotional state, pain detection, engagement level
- **Implementation**: 
  ```python
  from feat import Detector
  detector = Detector()
  faces = detector.detect_faces(frame)
  emotions = detector.detect_emotion(faces)
  ```

#### 2.4 Fatigue/Drowsiness Detection from Video
**Recommended Approach**: **Eye aspect ratio (EAR) + PERCLOS + head pose**
- **Why**: EAR detects eye closure; PERCLOS (percentage of eye closure) is clinically validated for drowsiness; head pose indicates alertness
- **Health relevance**: Sleep quality, medication side effects, depression
- **Implementation**: MediaPipe Face Mesh for facial landmarks → calculate EAR and PERCLOS
- **Thresholds**: EAR < 0.2 indicates closed eyes; PERCLOS > 0.15 indicates drowsiness

#### 2.5 Pose Estimation & Activity Analysis
**Recommended Model**: **MediaPipe Pose** (BlazePose)
- **Why**: Lightweight (runs at 30+ FPS on mobile), 33 keypoints, proven in fitness/rehab applications
- **Health relevance**: Mobility assessment, fall risk, activity level, rehabilitation progress
- **Implementation**:
  ```python
  import mediapipe as mp
  mp_pose = mp.solutions.pose
  pose = mp_pose.Pose(static_image_mode=False)
  results = pose.process(image)
  ```
- **Derived metrics**: 
  - Sit-to-stand time (frailty indicator)
  - Gait speed (if video walking)
  - Posture slouching (depression/fatigue indicator)
  - Activity counts (movement frequency)

---

### 3. Data Fusion & Decision Layer

**Purpose**: Combine multimodal signals into actionable health insights

#### 3.1 Multimodal Fusion Strategy
**Approach**: **Late fusion with rule-based + lightweight ML**
- **Why**: Late fusion allows each modality to fail gracefully; rules provide interpretability; ML captures complex patterns
- **Architecture**:
  ```
  Audio features ──┐
                  ├──→ Feature concatenation ──→ LightGBM/XGBoost ──→ Health score
  Video features ──┘     (heart rate, emotion, fatigue, etc.)
  
  LLM context ────→ Rule-based triggers ──→ Alert thresholds
  (conversation content, user responses)
  ```

#### 3.2 Health Score Calculation
**Recommended**: **Weighted ensemble of anomaly detectors**
- Vital signs (HR, RR): Z-score deviation from personal baseline
- Emotional state: Sentiment trend analysis over time
- Fatigue: Composite of voice + eye + activity metrics
- Cognitive: Linguistic complexity from LLM interaction logs

#### 3.3 Alert Triggering Logic
**Rule-based thresholds** (examples):
- Heart rate > 100 BPM or < 50 BPM for >5 minutes → Alert
- PERCLOS > 0.3 during conversation → Fatigue alert
- Negative emotion detected for 3+ consecutive days → Depression screening trigger
- No activity detected for >24 hours → Wellness check

---

### 4. Infrastructure & Deployment

#### 4.1 On-Device vs Cloud Inference
| Component | Placement | Rationale |
|-----------|-----------|-----------|
| Voice activity detection | On-device | Privacy, low latency |
| Emotion recognition (audio) | On-device | Privacy, real-time |
| rPPG (heart rate) | On-device | Privacy, continuous monitoring |
| Face detection | On-device | Privacy, preprocessing |
| Pose estimation | On-device | Real-time feedback |
| Complex fusion models | Cloud | Model size, battery life |
| LLM (MERaLiON/SEA-LION) | Cloud | Model size, state management |
| Alert decision engine | Cloud | Integration with care systems |

#### 4.2 Model Optimization for Mobile
- **Quantization**: INT8 quantization reduces model size by 4x with minimal accuracy loss
- **Pruning**: Remove 30-50% of weights from pose estimation models
- **Knowledge distillation**: Train smaller student models from larger teachers
- **Frame skipping**: Process video at 5-10 FPS instead of 30 FPS for efficiency

#### 4.3 Recommended Tech Stack Summary
| Layer | Technology | Purpose |
|-------|------------|---------|
| Mobile Framework | React Native / Flutter | Cross-platform app |
| On-device ML | TensorFlow Lite / ONNX Runtime | Model inference |
| Audio Processing | Librosa, PyAudioAnalysis | Feature extraction |
| Video Processing | OpenCV, MediaPipe | Preprocessing, pose |
| Backend ML | Python, scikit-learn, LightGBM | Fusion, classification |
| LLM Integration | MERaLiON/SEA-LION APIs | Conversation orchestration |
| Database | PostgreSQL + Redis | User data, session cache |
| Cloud | AWS/GCP Singapore region | Data residency compliance |

---

## Integration with Your LLM Architecture

### How the Pieces Fit Together

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER MOBILE DEVICE                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │   Camera     │  │  Microphone  │  │  On-Device   │  │     UI       │    │
│  │   Stream     │  │   Stream     │  │    ML        │  │  (Chat/Game) │    │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘    │
│         │                 │                 │                 │            │
│         └─────────────────┴─────────────────┘                 │            │
│                           │                                   │            │
│                    ┌──────▼───────┐                    ┌──────▼───────┐    │
│                    │ Data Buffer  │                    │  LLM Client  │    │
│                    │ (Privacy)    │                    │   (Voice)    │    │
│                    └──────┬───────┘                    └──────┬───────┘    │
└───────────────────────────┼───────────────────────────────────┼────────────┘
                            │                                   │
                            ▼                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLOUD BACKEND                                   │
│  ┌─────────────────────────────────────┐  ┌─────────────────────────────┐   │
│  │      Multimodal ML Pipeline         │  │   LLM Orchestrator          │   │
│  │  ┌─────────┐ ┌─────────┐ ┌────────┐ │  │  ┌─────────────────────┐   │   │
│  │  │  rPPG   │ │  Pose   │ │ Emotion│ │  │  │  MERaLiON/SEA-LION  │   │   │
│  │  │  Model  │ │Analysis │ │ Model  │ │  │  │                     │   │   │
│  │  └────┬────┘ └────┬────┘ └───┬────┘ │  │  │  • Context awareness│   │   │
│  │       └─────────────┴────────┘      │  │  │  • Sentiment analysis│   │   │
│  │                │                    │  │  │  • Response generation│   │   │
│  │         ┌──────▼───────┐            │  │  └─────────────────────┘   │   │
│  │         │ Health Score │            │  │                            │   │
│  │         │   Fusion     │            │  │  ┌─────────────────────┐   │   │
│  │         └──────┬───────┘            │  │  │   Decision Engine   │   │   │
│  │                │                    │  │  │  • Trigger games    │   │   │
│  │         ┌──────▼───────┐            │  │  │  • Request camera   │   │   │
│  │         │ Alert System │◄───────────┼──┼──┤  • Escalate to care │   │   │
│  │         └──────────────┘            │  │  └─────────────────────┘   │   │
│  └─────────────────────────────────────┘  └─────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### LLM-ML Interaction Flow

1. **LLM initiates conversation** → Records audio stream → Sends to ML pipeline
2. **ML pipeline processes** → Returns emotion, fatigue, cognitive metrics → Updates LLM context
3. **LLM decides to trigger camera** → Requests video capture → ML extracts rPPG, pose, facial expressions
4. **Health fusion model** → Combines all signals → Generates health score
5. **Decision engine** → If score exceeds threshold → LLM notified to alert caregiver
6. **LLM crafts empathetic message** → Informs user of concern → Offers support

---

## Suggested Implementation Phases

### Phase 1: MVP (Weeks 1-4)
**Focus**: Core voice + basic health metrics
- Implement voice emotion recognition (Emotion2Vec)
- Integrate with MERaLiON for voice chat
- Simple rule-based alerting (e.g., 3 negative emotions → alert)
- Demo via web app (as you mentioned)

### Phase 2: Multimodal Expansion (Weeks 5-8)
**Focus**: Add video capabilities
- Implement rPPG (heart rate) from camera
- Add facial emotion recognition
- Create "games" that elicit specific health data (e.g., "let's do some deep breathing" for rPPG)
- Basic pose estimation for activity tracking

### Phase 3: Intelligence Layer (Weeks 9-12)
**Focus**: Fusion and personalization
- Implement health score fusion model
- Personal baselines (each user has their own "normal")
- Trend analysis (detecting decline over weeks)
- Caregiver dashboard integration

---

## Key Considerations for Singapore Context

### 1. Multilingual Support
- **MERaLiON** already handles Singlish and code-switching
- **Emotion2Vec** works across languages (trained on multilingual data)
- **SenseVoice** supports Mandarin, Cantonese, English
- Consider: Some elderly may prefer dialects (Hokkien, Teochew, Cantonese) - MERaLiON has some dialect support

### 2. Cultural Sensitivity
- Emotional expression varies by culture (e.g., less facial expressiveness in some Asian cultures)
- Calibrate emotion models on local data if possible
- Privacy concerns may make camera usage sensitive - always require explicit consent

### 3. Healthcare Integration
- Synapxe's systems likely use HL7 FHIR - plan for data export format
- Alerts should go to appropriate channels (GP, caregiver, emergency services)
- Consider MOH's HealthHub integration for patient access to their own data

### 4. Regulatory Compliance
- **PDPA**: Explicit consent for data collection, right to deletion
- **MOH guidelines**: Medical device classification if making diagnostic claims
- **IMDA**: Data localization requirements (use Singapore cloud regions)

---

## Research Citations & Resources

### Audio/Emotion
- Emotion2Vec: https://huggingface.co/Emotion2Vec/emotion2vec_plus_large
- SenseVoice: https://github.com/FunAudioLLM/SenseVoice
- SpeechBrain: https://speechbrain.github.io/

### Video/rPPG
- open-rppg toolbox: https://github.com/KegangWangCCNU/open-rppg
- rPPG-Toolbox: https://github.com/ubicomplab/rPPG-Toolbox
- MediaPipe Pose: https://mediapipe-studio.webapps.google.com/demo/pose_landmarker

### Facial Analysis
- Py-Feat: https://py-feat.org/
- DeepFace: https://github.com/serengil/deepface

### Cognitive/Depression
- PROCESS Challenge 2025: https://www.process-challenge.com/
- Dementia detection research: https://www.frontiersin.org/journals/neuroinformatics/articles/10.3389/fninf.2025.1679664/full

### Singapore Context
- CNA Article on Elderly Care AI: https://www.channelnewsasia.com/today/big-read/ai-technology-elderly-care-active-ageing-5477456
- Lions Befrienders IM-HAPPY: Voice-based AI agent for senior check-ins

---

## Final Recommendations

### What to Keep from Your Original Plan
✅ **LLM as orchestrator** - MERaLiON/SEA-LION are excellent choices for Singapore context  
✅ **State machine approach** - Clear, implementable architecture  
✅ **Gamification** - Effective for engagement and eliciting specific health data  
✅ **Privacy-first design** - On-device preprocessing, consent-based camera usage  

### What to Add/Modify
🔧 **Start with audio-only MVP** - Video adds complexity; prove voice health monitoring first  
🔧 **Use established open-source models** - Don't train from scratch; fine-tune if needed  
🔧 **Plan for failure modes** - Each ML component should have a fallback (e.g., if rPPG fails, use self-reported heart rate)  
🔧 **Build feedback loops** - When alerts are triggered, track outcomes to improve thresholds  

### Critical Success Factors
1. **User trust** - Elderly users must feel comfortable talking to the AI; MERaLiON's Singlish support helps
2. **Clinical validation** - Partner with a GP or polyclinic to validate health correlations
3. **Caregiver buy-in** - Alerts are useless if caregivers don't act on them; design the escalation workflow together
4. **Battery life** - Continuous monitoring drains batteries; optimize inference and use cloud for heavy lifting

---

## Questions to Discuss as a Team

1. **Which chronic conditions are you prioritizing?** (Diabetes, hypertension, heart disease, dementia - each has different monitoring needs)
2. **What's your clinical validation strategy?** (Partnership with healthcare provider?)
3. **How will you handle false positives?** (Alert fatigue is a real problem for caregivers)
4. **What's your data strategy for model improvement?** (Synthetic data? Partnership with hospitals?)
5. **Camera usage**: Always optional, or required for certain features?

---

*This recommendation is based on research of current open-source models, peer-reviewed publications from 2024-2025, and Singapore's healthcare context. The models suggested are all open-source or have open-source alternatives, aligning with the challenge's focus on demonstrating functional prototypes.*
