# Aiko's Master Plan: Multimodal Remote Monitoring (Zero-Data Workaround)

## Pre-Plan Assessment

### 1. What information would you need to make this plan significantly more accurate?
- Exactly which open-source pre-trained models (e.g., MediaPipe, Emotion2Vec) you are allowed or intend to use for extracting the raw base metrics (landmarks, pitch) before applying your scoring heuristics.
- Whether you will use MERaLiON via API or locally (affects the integration logic with OpenClaw).
- Specifics on what hardware the server will be running on (can it handle background video processing if sent from Telegram?).

### 2. What parts of this problem are outside your reliable knowledge?
- The exact criteria the human judges will use to evaluate the "functional demonstration."
- Telegram's exact size limits for WebRTC streams / inline app video uploads via bots if the user is asked to submit a video directly.

### 3. What are you most likely to get wrong in this plan and why?
- The integration boundary between OpenClaw and the Next.js frontend. OpenClaw operates primarily on textual/file interfaces, so chaining it seamlessly to a Web App where OpenClaw directly reads the Web App's state might be brittle. I might over-assume OpenClaw's out-of-the-box ability to deeply hook into an external web session without custom API polling.

---

## PASS 1: The Blueprint Skeleton

**[KNOWN]** We lack raw datasets to train a deep learning classifier from scratch. 
**[KNOWN]** The challenge expects a functional prototype of multimodal wellness monitoring.
**[INFERRED]** A deterministic scoring system built on top of pre-trained zero-shot models (like MediaPipe) is the most viable "workaround" to bypass the lack of training data while still utilizing AI.
**[ASSUMED]** The users have access to Telegram and a smartphone camera.
**[UNCERTAIN]** Maximum video/audio payload size via Telegram bot to our server.

### Architecture Components

1. **The Communicator (Telegram + OpenClaw)**
   - [NEEDS INPUT: Confirm if OpenClaw has direct Telegram Bot integration you are confident in, or if we need a custom Python wrapper that feeds text updates to OpenClaw.]
   - Handles State 0 (Passive Conversations).
   - Utilizes MERaLiON for empathetic, localized responses.

2. **The Assessor (Determinism over Classification)**
   - Takes raw signals (audio from Telegram, video from Next.js Mini App).
   - Uses zero-shot open-source models to extract features (e.g., MediaPipe Pose for joints, PyAudioAnalysis for voice jitter).
   - Computes an **Interval Data Score** instead of classifying "Sick" vs "Healthy" (e.g., Mobility Score = 85/100 based on skeletal symmetry and speed).

3. **The Interactor (Next.js Telegram Mini App)**
   - Handles State 1 (Active Assessment).
   - [NEEDS INPUT: Do you want the Mini App to process video locally within the browser via WebAssembly (MediaPipe JS), or stream it back to the server for processing?]

### Phased Execution

#### Phase 1: Establish the State 0 Pipeline
- Connect Telegram Bot to OpenClaw.
- Give OpenClaw instructions to route voice messages to the MERaLiON API.
- Create a baseline scoring script for audio data (e.g., detecting pauses/fatigue heuristically) and map it to interval data.

#### Phase 2: Build the State 1 Escalation Protocol
- Implement logic where OpenClaw triggers the Next.js Mini-App link if State 0 audio scores drop below a threshold.
- Develop the "Workaround" visual assessment (e.g., TUG Gait test or Facial Symmetry test via MediaPipe) inside the Next.js app.

#### Phase 3: Data Aggregation & Presentation Layer
- Build a dashboard/log to visualize the interval scores over time (proving the concept of "continuous monitoring" required by the prompt).

--- STOP. Review pass 1 with the user. ---