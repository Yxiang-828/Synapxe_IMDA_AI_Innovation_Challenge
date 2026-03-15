# 🦁 MERaLiON Health

**An empathetic, voice-driven chronic patient monitoring system built for Singapore.** Developed for the Synapxe IMDA AI Innovation Challenge, this web-based mobile prototype leverages Synapxe’s open-source LLMs (**SeaLion** and **MERaLiON**) to provide natural, Singlish-fluent companionship and proactive health monitoring for elderly and chronic patients.

---

## 💡 The Problem & Solution
Chronic disease monitoring often relies on expensive wearables, 24/7 CCTV, or intrusive caretaker check-ins. 

**MERaLiON Health** utilizes the one device patients always have with them: their mobile phone. By mimicking a natural, voice-only conversation with a trusted digital companion, the app bypasses the friction of clinical health apps. 

Through casual daily check-ins ("Eh, how are you feeling today?"), the app seamlessly transitions into quick, engaging mini-games designed to capture critical ML data (audio clarity, facial symmetry, cognitive reflex) to monitor for sudden health deterioration like strokes or extreme fatigue.

## ⚙️ Core Architecture


This repository contains a full-stack web prototype designed to simulate a native mobile application. 

* **Frontend (The Mobile UI):** Next.js & Tailwind CSS. Constrained to a mobile aspect ratio with a highly accessible, voice-centric UI.
* **Backend (The AI Bridge):** Python & FastAPI. Handles the logic, routes LLM prompts, and triggers the simulated ML data collection events.
* **Native Integrations:** * `Web Speech API`: Handles real-time speech-to-text and text-to-speech for seamless voice chatting.
    * `MediaDevices API`: Captures webcam data for the visual health-check mini-games.

---

## 🎮 How It Works (The Demo Flow)

1.  **State 1 (General Chat):** The LLM initiates a casual conversation in Singlish to check on the patient's day. The user responds via voice.
2.  **The Trigger:** Depending on the user's response (e.g., mentioning they are "tired" or "sleepy"), the backend dynamically shifts priority to a health check.
3.  **State 2 (Data Acquisition):** The LLM seamlessly invites the user to play a quick game. 
    * *Category A (Audio):* Memory recall or tongue twisters to assess cognitive delay and speech slurring.
    * *Category B (Visual):* "Emotion Mimic" (e.g., "Show me a big smile!") which activates the camera to check facial symmetry.
4.  **Resolution:** The acquired data is (theoretically) sent to a machine learning pipeline, and caretakers are alerted if thresholds are breached.

> **Note for Judges (Prototype Scope):** For the purpose of this hackathon demo, the complex 24/7 background ML pipeline (State 2) is simulated. The application successfully demonstrates the *critical path*: capturing user voice, routing to the LLM backend, parsing the response, and triggering the hardware camera/mic for data acquisition based on conversational context.

---

## 🚀 Getting Started (Local Development)

To run this prototype on your local machine, you will need two terminal windows running simultaneously.

### 1. Start the Backend (FastAPI)
Navigate to the backend directory, activate the virtual environment, and start the Python server:
```bash
cd backend
source .venv/bin/activate
# Install dependencies if you haven't yet: pip install fastapi uvicorn pydantic
uvicorn main:app --host 127.0.0.1 --port 8080 --reload