from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
import asyncio
import database
from bot import create_bot_app

# --- Bot Lifespan Management ---
bot_app = create_bot_app()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize SQLite Database
    database.init_db()
    print("Health Database Initialized.")
    
    # Start Telegram Bot running in the background alongside FastAPI
    await bot_app.initialize()
    await bot_app.start()
    await bot_app.updater.start_polling()
    print("Telegram State 0 Companion Online!")
    yield
    # Shutdown Telegram Bot
    await bot_app.updater.stop()
    await bot_app.stop()
    await bot_app.shutdown()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessage(BaseModel):
    text: str

class MiniGameScore(BaseModel):
    telegram_id: str
    game_type: str  # "facial_symmetry", "spiral_test"
    score: float

@app.post("/api/chat")
async def chat_endpoint(message: ChatMessage):
    # Backward compatibility for the web UI 
    text = message.text.lower()
    
    if "bored" in text or "play" in text:
        return {
            "reply": "You want to play ah? Okay let's test your memory. Get ready for a word blitz!",
            "route": "/audio-game"
        }
    
    return {
        "reply": "Talking via web right now. For the true experience, message my Telegram Bot!",
        "route": None
    }

@app.post("/api/log_score")
async def log_score_endpoint(data: MiniGameScore):
    """
    Endpoint for the Next.js Mini App to POST results back to the server
    after running MediaPipe on the device.
    """
    database.log_interaction(
        telegram_id=data.telegram_id,
        interaction_type=data.game_type,
        score_delta=data.score,
        note="Score recorded from Next.js Mini App"
    )
    return {"status": "success", "recorded_score": data.score}