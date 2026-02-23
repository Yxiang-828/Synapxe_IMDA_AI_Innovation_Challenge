import datetime
print(f"Generating codebase structure at {datetime.datetime.now()}")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import random

app = FastAPI()

# Allow the Next.js frontend to communicate with this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessage(BaseModel):
    text: str

@app.post("/api/chat")
async def chat_endpoint(message: ChatMessage):
    text = message.text.lower()
    
    # Mock ML Trigger Logic: If user mentions being tired, trigger Category B Game
    if "tired" in text or "sleepy" in text or "game" in text:
        return {
            "reply": "Aiya, you sound tired. Let's do a quick facial check to make sure you are okay. Turn on your camera and show me a big smile!",
            "triggerGame": "emotion_mimic"
        }
    
    # Standard Singlish responses for State 1
    singlish_responses = [
        "Wah, your day sounds not bad hor! Have you eaten?",
        "Don't stress too much lah. Take a break if you need!",
        "Okay lor, I note that down. Make sure to drink more water!"
    ]
    
    return {
        "reply": random.choice(singlish_responses),
        "triggerGame": None
    }