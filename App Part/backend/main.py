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

from typing import Optional, Dict

class MiniGameScore(BaseModel):
    telegram_id: str
    game_type: str  # "facial_symmetry", "spiral_test"
    score: float
    metrics: Optional[Dict[str, float]] = None

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
    
    # Process professional LLM analysis!
    import httpx
    import re
    
    if data.game_type == "mobility_score":
        prompt = (f"The user just completed a clinical physical mobility test running locally on their device. "
                  f"They successfully completed {data.score} out of 9 total target reps across three exercises "
                  f"(Seated Knee Extension, Shoulder Abduction, and Standing March). ")
        prompt += (f"Analyze these results professionally. If they scored 9/9, praise them for great skeletal mobility and joint health. "
                   f"If they scored less, gently warn them that their range of motion (ROM) might be stiff and suggest they stretch more every day to prevent joint degradation. "
                   f"Wrap it up strictly acting as Aiko (a sharp but secretly caring tsundere assistant).")

    else:
        prompt = (f"The user just completed a clinical facial symmetry test running locally on their device. "
                  f"They received an overall rating of {data.score}%. ")

        if data.metrics:
            prompt += (f"Here are the specific metrics captured across the 10 second span: "
                       f"Best/Highest peak symmetry reached: {data.metrics.get('bestSymmetry')}%. "
                       f"Lowest dropped symmetry: {data.metrics.get('lowestSymmetry')}%. "
                       f"Mean symmetry: {data.metrics.get('meanSymmetry')}%, Median symmetry: {data.metrics.get('medianSymmetry')}%. "
                       f"Variance (muscle flutter): {data.metrics.get('variance')}. ")

        prompt += (f"Analyze these results professionally. If there is a huge difference between 'bestSymmetry' and 'medianSymmetry' "
                   f"or high variance, point out that they could hold a smile initially but dropped due to muscle fatigue. "
                   f"Break down what facial symmetry means for potential neurological or fatigue indicators "
                   f"(Bell's palsy, stroke risk, or exhaustion). "
                   f"IMPORTANT RULE: If their score is 80% or higher, DO NOT tell them to see a doctor. "
                   f"Wrap it up strictly acting as Aiko (a sharp but secretly caring tsundere assistant).")

    try:
        # Wrap sending logic in a background task properly without blocking FastAPI's return
        async def background_send():
            llm_response = None
            try:
                await bot_app.bot.send_chat_action(chat_id=data.telegram_id, action="typing")
                
                # Fetch LLM response directly via Ollama
                payload = {
                    "model": "aisingapore/Llama-SEA-LION-v3.5-8B-R:latest",
                    "system": "You are Aiko, the user's sharp tsundere health assistant.",
                    "prompt": prompt,
                    "stream": False
                }
                async with httpx.AsyncClient(timeout=45.0) as client:
                    resp = await client.post("http://127.0.0.1:11434/api/generate", json=payload)
                    resp.raise_for_status()
                    raw_text = resp.json().get("response", "Hmph, my system couldn't read that right now.")
                    llm_response = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL).strip()
                
                # Fallback model parsing for telegram markdown
                text_response = llm_response.replace("*", "**").replace("_", "__") # Escape some basic conflicts
                await bot_app.bot.send_message(
                    chat_id=data.telegram_id,
                    text=f"📊 **Health Check Result:**\n\n{text_response}",
                    parse_mode="Markdown"
                )
            except Exception as inner_e:
                print(f"Inner task sending error: {inner_e}")
                llm_response_fallback = llm_response if llm_response else "Failed to analyze score..."
                await bot_app.bot.send_message(
                    chat_id=data.telegram_id,
                    text=f"📊 Health Check Result:\n\n{llm_response_fallback}"
                )

        asyncio.create_task(background_send())

    except Exception as e:
        print(f"Failed to schedule telegram webhook message: {e}")

    return {"status": "success", "recorded_score": data.score}



