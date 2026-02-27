from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import random

app = FastAPI()

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
    
    # If they say "bored", send them to the audio game route
    if "bored" in text or "play" in text:
        return {
            "reply": "You want to play ah? Okay let's test your memory. Get ready for a word blitz!",
            "route": "/audio-game"
        }
    
    # Standard responses
    singlish_responses = [
        "Wah, your day sounds not bad hor! Have you eaten?",
        "Don't stress too much lah. Take a break if you need!",
        "Okay lor, I note that down. Make sure to drink more water!"
    ]
    
    return {
        "reply": random.choice(singlish_responses),
        "route": None
    }