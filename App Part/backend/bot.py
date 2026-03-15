import logging
import asyncio
import httpx
import json
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, WebAppInfo
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters
from database import get_or_create_patient, log_interaction

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Config from your Sandbox!
TOKEN = "7835998888:AAHjGHvPGKB-9AoM8r05MN85iioLbY6CUu8"
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
# Using SEA-LION to score points for the IMDA hackathon!
AI_MODEL = "aisingapore/Llama-SEA-LION-v3.5-8B-R:latest" # Fallback to llama3.2 if it fails

# Next.js Telegram Mini App URL (for now, pointing to local tunnel or localhost)
# In production, this must be HTTPS (e.g. ngrok or vercel)
MINI_APP_URL = "https://your-ngrok-url.app/camera-game" 

async def get_llm_response(prompt: str) -> str:
    system_prompt = (
        "You are 'Health Buddy', a friendly, empathetic digital companion for elderly patients in Singapore. "
        "You speak politely with a natural mix of English and very mild Singlish (like 'lah', 'hor', 'wah'). "
        "Keep your answers short (1-2 sentences), caring, and encourage them to share how they feel."
    )
    
    payload = {
        "model": AI_MODEL,
        "system": system_prompt,
        "prompt": prompt,
        "stream": False
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(OLLAMA_URL, json=payload)
            resp.raise_for_status()
            return resp.json().get("response", "I'm sorry, I didn't quite catch that. How are you doing?")
    except Exception as e:
        logger.error(f"Ollama error: {e}")
        # Fallback if SEA-LION isn't loaded
        return "Alamak, my brain a bit stuck right now! How's your day going?"

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    user_id = str(update.message.chat_id)
    user_name = update.message.chat.first_name or "Ah Ma"
    
    # Ensure patient exists in DB
    patient = get_or_create_patient(user_id, user_name)
    
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action='typing')
    
    # 1. Get Conversation Response
    response_msg = await get_llm_response(user_text)
    
    # 2. Heuristic Check (Simulated trigger for State 1)
    # If the user says they are tired, dizzy, or bored, we trigger the mini-app game
    lower_text = user_text.lower()
    if any(word in lower_text for word in ["tired", "bored", "dizzy", "pain", "play", "sick"]):
        log_interaction(user_id, "trigger_game", 0.0, "User indicated negative state, triggered mini-app")
        
        keyboard = [
            [InlineKeyboardButton("🎮 Play Quick Check-up Game", web_app=WebAppInfo(url=f"{MINI_APP_URL}?uid={user_id}"))]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            f"{response_msg}\n\nEh, since you mentioned that, let's do a quick activity to see how you are doing! Tap the button below.", 
            reply_markup=reply_markup
        )
    else:
        log_interaction(user_id, "chat", 0.0, "Standard conversation")
        await update.message.reply_text(response_msg)

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Here we would download the voice note, run `detect_voice_fatigue.py`, and update the DB score.
    user_id = str(update.message.chat_id)
    file_id = update.message.voice.file_id
    
    await update.message.reply_text("Wah, nice voice note! I am listening to it right now... (Simulating Audio Analysis)")
    # TODO: Connect to your ML_stash python scripts here!

def create_bot_app():
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    return app
