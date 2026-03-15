import logging
import asyncio
import httpx
import re
import os
from datetime import datetime
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, WebAppInfo
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, CommandHandler, filters
from database import get_or_create_patient, log_interaction, log_chat_message, get_recent_history, update_patient_interval, update_last_prompted
import ml_bridge  # Import our new Transcription mapping!

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global Queue Tracking ---
processing_lock = asyncio.Lock()
queue_count = 0
# -----------------------------

# Hackathon specific bot token
TOKEN = "8602020200:AAFw9atyXJBlek5TSz3VW-k3c84DSbpIdM4"
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
# Using SEA-LION to score points for the IMDA hackathon!
AI_MODEL = "aisingapore/Llama-SEA-LION-v3.5-8B-R:latest" # Fallback to llama3.2 if it fails

# Next.js Telegram Mini App URL (for now, pointing to local tunnel or localhost)
# In production, this must be HTTPS (e.g. ngrok or vercel)
# You MUST replace this with your actual tunneling service URL (ngrok, localtunnel, etc.)
BASE_URL = "https://victorian-whatever-buyers-mrna.trycloudflare.com"
SMILE_APP_URL = f"{BASE_URL}/camera-game"
WORKOUT_APP_URL = f"{BASE_URL}/mobility-game" 

import re

async def get_llm_response(user_id: str, prompt: str, user_name: str, is_proactive: bool = False) -> str:
    # 1. Fetch the user's recent chat history
    recent_history = get_recent_history(user_id, limit=6)
    
    if is_proactive:
        # State 0 - Proactive AI Ping: The LLM creates the first message to wake them up.
        system_prompt = (
            f"You are 'Mera', a warm, encouraging, and caring digital companion for {user_name} with a playful, very mild tsundere streak. You deeply care about their health and always cheer them on. "
            f"You speak politely with a natural mix of English and very mild Singlish (like 'lah', 'hor', 'wah'). "
            f"Your goal right now is to initiate a conversation with them and check in on how they are feeling today. "
            f"Keep your message short (1-2 sentences), caring, and ask them a specific question related to their past history if relevant. "
            f"Here is your recent conversation history. Use this to continue the relationship:\n\n{recent_history}"
        )
        # If it's proactive, it's an automatic timer, so we are allowed to ask them to play
        actual_prompt = "Generate your friendly check-in message for today. Playfully suggest that they either play 'Smile Checker' or 'Mobility Workout' to stretch their body and prove they are healthy. (Do not output raw URLs)."
    else:
        # Standard conversation response
        system_prompt = (
            f"You are 'Mera', a warm, encouraging, and caring digital companion for {user_name} with a playful, very mild tsundere streak. You deeply care about their health and always cheer them on. "
            f"You speak politely with a natural mix of English and very mild Singlish (like 'lah', 'hor', 'wah'). "
            f"Keep your answers short (1-2 sentences), caring, and encourage them to share how they feel. "
            f"IMPORTANT: DO NOT ask them to play a game, and DO NOT mention 'Smile Checker' or 'Mobility Workout' UNLESS they explicitly ask to play, workout, or do an exercise. "
            f"Here is your previous conversation context with {user_name}:\n{recent_history}"
        )
        actual_prompt = f"User says: {prompt}\nOnly if the user explicitly asks for a game or workout, mention 'Smile Checker' or 'Mobility Workout'. Respond as Mera:"

    payload = {
        "model": AI_MODEL,
        "system": system_prompt,
        "prompt": actual_prompt,
        "stream": False
    }

    try:
        async with httpx.AsyncClient(timeout=45.0) as client:
            resp = await client.post(OLLAMA_URL, json=payload)
            resp.raise_for_status()
            raw_text = resp.json().get("response", "I'm sorry, I didn't quite catch that. How are you doing?")
            
            # Strip out <think>...</think> tags which reasoning models sometimes output
            clean_text = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL).strip()
            clean_text = clean_text.replace('</s>', '').strip()
            return clean_text if clean_text else raw_text
    except Exception as e:
        logger.error(f"Ollama error: {e}")
        # Fallback if SEA-LION isn't loaded
        return "Alamak, my brain a bit stuck right now! How's your day going?"

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global queue_count
    
    # Ensure there is text to process
    if not update.message.text:
        return
        
    user_name = update.message.from_user.first_name or "Ah Ma"
    raw_user_text = update.message.text
    chat_type = update.message.chat.type
    
    logger.info(f"[TELEGRAM EVENT - TEXT] Received from {user_name} in {chat_type}: '{raw_user_text}'")
        
    # --- Group Chat Protection ---
    if chat_type in ['group', 'supergroup']:
        bot_username = context.bot.username
        text_lower = raw_user_text.lower()
        bot_username_lower = f"@{bot_username.lower()}"
        
        is_mentioned = bot_username_lower in text_lower
        is_reply = (update.message.reply_to_message and 
                    update.message.reply_to_message.from_user.id == context.bot.id)
        
        # If it's a group, only respond if pinged or replied to
        if not (is_mentioned or is_reply):
            logger.info(f"Skipping group message (bot not pinged): '{raw_user_text}'")
            return
            
        # Strip the mention from the text so the LLM doesn't get confused (case-insensitive replace)
        import re
        # We MUST NOT modify update.message.text directly as it is read-only in this library version
        user_text = re.sub(f"(?i)@{bot_username}", "", raw_user_text).strip()
        logger.info(f"Stripped group mention. Final text to LLM: '{user_text}'")
    else:
        user_text = raw_user_text
    # -----------------------------

    # Fix: update.message.chat_id gets the group's ID in a group chat, causing memories to merge.
    # We must use update.message.from_user.id to strictly isolate patient memories!
    user_id = str(update.message.from_user.id)

    # Ensure patient exists in DB
    patient = get_or_create_patient(user_id, user_name)
    
    # Add to chat history immediately!
    log_chat_message(user_id, "user", user_text)
    
    # --- Rate Limiting & Queue ---
    if processing_lock.locked():
        await update.message.reply_text(f"Hold on ah {user_name}! I am currently replying to {queue_count} other message(s). Give me a short while...")
    
    queue_count += 1
    try:
        async with processing_lock:
            await context.bot.send_chat_action(chat_id=update.effective_chat.id, action='typing')
            
            # 1. Get Conversation Response with context
            response_msg = await get_llm_response(user_id, user_text, user_name, is_proactive=False)
            
            # Log bot response to DB!
            log_chat_message(user_id, "assistant", response_msg)
            logger.info(f"[OLLAMA REPLY]: '{response_msg}'")

            # Reset timer because user is actively talking to us so we don't randomly interrupt them later
            update_last_prompted(user_id)

            # Send both games if the LLM output suggests playing
            if "Smile Checker" in response_msg or "Mobility Workout" in response_msg:
                if chat_type in ['group', 'supergroup']:
                    bot_username = context.bot.username
                    keyboard = [[InlineKeyboardButton("🎮 Play Games (Tap to open in DM)", url=f"https://t.me/{bot_username}")]]
                    reply_markup = InlineKeyboardMarkup(keyboard)
                    await update.message.reply_text(response_msg + "\n\n(Mini apps don't work in group chats, message me directly!)", reply_markup=reply_markup)
                else:
                    keyboard = [
                        [InlineKeyboardButton("🎮 Play Smile Checker", web_app=WebAppInfo(url=f"{SMILE_APP_URL}?uid={user_id}"))],
                        [InlineKeyboardButton("🏃 Play Mobility Workout", web_app=WebAppInfo(url=f"{WORKOUT_APP_URL}?uid={user_id}"))]
                    ]
                    reply_markup = InlineKeyboardMarkup(keyboard)
                    await update.message.reply_text(response_msg, reply_markup=reply_markup)
            else:
                await update.message.reply_text(response_msg)


    finally:

        queue_count -= 1

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global queue_count
    
    user_name = update.message.from_user.first_name or "Ah Ma"
    chat_type = update.message.chat.type
    
    logger.info(f"[TELEGRAM EVENT - VOICE] Received voice from {user_name} in {chat_type}")
    
    # --- Group Chat Protection ---
    if chat_type in ['group', 'supergroup']:
        is_reply = (update.message.reply_to_message and 
                    update.message.reply_to_message.from_user.id == context.bot.id)
        
        # In groups, we only process voice notes if they explicitly replied to the bot
        if not is_reply:
            logger.info(f"Skipping voice note from group (was not a direct reply to bot).")
            return
    # -----------------------------

    user_id = str(update.message.from_user.id)
    file_id = update.message.voice.file_id

    # --- Rate Limiting & Queue ---
    if processing_lock.locked():
        await update.message.reply_text(f"Hold on ah {user_name}! I am currently replying to {queue_count} other message(s). Give me a short while...")
    
    queue_count += 1
    try:
        async with processing_lock:
            # 1. Acknowledge Receipt
            await update.message.reply_text("Wah, nice voice note! Let me listen to it real quick...")
            await context.bot.send_chat_action(chat_id=update.effective_chat.id, action='typing')
            
            # 2. Download the audio file physically to the server
            new_file = await context.bot.get_file(file_id)
            file_path = f"temp_voice_{user_id}.ogg"
            await new_file.download_to_drive(file_path)
            
            # 3. Transcribe the Voice to Text (AI Hearing!)
            # We use a thread to not block the Telegram event loop while Whisper runs
            transcribed_text = await asyncio.to_thread(ml_bridge.transcribe_audio, file_path)
            logger.info(f"[WHISPER TRANSCRIPTION {user_id}]: {transcribed_text}")
            
            # Clean up the file
            if os.path.exists(file_path):
                os.remove(file_path)
            
            if not transcribed_text or "Could not transcribe" in transcribed_text:
                await update.message.reply_text("Alamak, the audio was a bit muffled, I couldn't quite hear it. Can say again?")
                return

            log_chat_message(user_id, "user", f"[Voice Note]: {transcribed_text}")

            # 4. Generate the empathetic LLM response based on WHAT they actually said
            response_msg = await get_llm_response(user_id, transcribed_text, user_name, is_proactive=False)
            
            log_chat_message(user_id, "assistant", response_msg)
            
            # 5. Extract "Hidden" biomarkers (Simulating the VoiceFatigue ML_stash code)
            fatigue_score = ml_bridge.analyze_voice_fatigue(file_path)
            
            if fatigue_score > 0.7:
                log_interaction(user_id, "voice_analysis", -0.5, f"Transcript: {transcribed_text} | Fatigue: High")

                if chat_type in ['group', 'supergroup']:
                    bot_username = context.bot.username
                    keyboard = [[InlineKeyboardButton("🎮 Play Face Check Game (Tap to open in DM)", url=f"https://t.me/{bot_username}")]]
                    reply_markup = InlineKeyboardMarkup(keyboard)
                    await update.message.reply_text(
                        f"{response_msg}\n\nBy the way... your voice sounds a tiny bit tired today lah. Telegram group chats won't let me open the facial stretch game here though! Tap the button to message me privately and we can do it.",
                        reply_markup=reply_markup
                    )
                else:
                    keyboard = [
                        [InlineKeyboardButton("🎮 Play Face Check Game", web_app=WebAppInfo(url=f"{SMILE_APP_URL}?uid={user_id}"))]
                    ]
                    reply_markup = InlineKeyboardMarkup(keyboard)

                    await update.message.reply_text(
                        f"{response_msg}\n\nBy the way... your voice sounds a tiny bit tired today lah. Are you feeling okay?\nLet's do a quick facial stretch to make sure! Tap the button below to turn on your camera.",
                        reply_markup=reply_markup
                    )
            else:
                log_interaction(user_id, "voice_analysis", 0.1, f"Transcript: {transcribed_text} | Fatigue: Normal")
                # Send both games if the LLM output suggests playing
                if "Smile Checker" in response_msg or "Mobility Workout" in response_msg:
                    keyboard = [
                        [InlineKeyboardButton("🎮 Play Smile Checker", web_app=WebAppInfo(url=f"{SMILE_APP_URL}?uid={user_id}"))],
                        [InlineKeyboardButton("🏃 Play Mobility Workout", web_app=WebAppInfo(url=f"{WORKOUT_APP_URL}?uid={user_id}"))]
                    ]
                    reply_markup = InlineKeyboardMarkup(keyboard)
                    await update.message.reply_text(response_msg, reply_markup=reply_markup)
                else:
                    await update.message.reply_text(response_msg)
    finally:
        queue_count -= 1

# --- Scheduler Jobs (State 0 Proactive System) ---
async def set_timer_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Allows users to set how often the bot proactively messages them."""
    user_id = str(update.message.chat_id)
    user_name = update.message.from_user.first_name or "Ah Ma"
    get_or_create_patient(user_id, user_name)  # ensure they exist in DB
    
    args = context.args
    if len(args) != 2:
        await update.message.reply_text("Please use this format to set my reminder:\n\n/set_timer <hours> <minutes>\n\nExample for 2 hours and 30 mins: /set_timer 2 30")
        return
        
    try:
        hours = int(args[0])
        minutes = int(args[1])
        total_mins = (hours * 60) + minutes
        
        if total_mins < 1:
            await update.message.reply_text("The timer must be at least 1 minute lah! Don't make me spam you!")
            return
            
        update_patient_interval(user_id, total_mins)
        # Also reset their last prompted time so the timer starts fresh now
        update_last_prompted(user_id)
        
        await update.message.reply_text(f"Okay! I've set my alarm. I will proactively check in on you every {hours} hour(s) and {minutes} minute(s).")
    except ValueError:
        await update.message.reply_text("Ah boy/Ah girl, please give me numbers! Example: /set_timer 2 30")

async def scheduled_daily_checkin(context: ContextTypes.DEFAULT_TYPE):
    """
    This function will be called automatically by the job queue every 1 minute.
    It checks if enough time (interval_minutes) has elapsed since the last prompt,
    and if so, reaches out to the patient.
    """
    import sqlite3
    from database import DB_PATH
    from datetime import datetime
    
    logger.info("[CRON] Checking for users who need a proactive message...")
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT telegram_id, name, interval_minutes, last_prompted FROM patients")
    patients = cursor.fetchall()
    conn.close()
    
    now = datetime.now()
    
    for tg_id, name, interval_minutes, last_prompted_str in patients:
        try:
            # Parse the last prompted time
            if last_prompted_str:
                last_prompted = datetime.fromisoformat(last_prompted_str)
            else:
                last_prompted = now
                update_last_prompted(tg_id)
                continue
                
            elapsed_mins = (now - last_prompted).total_seconds() / 60.0
            
            if elapsed_mins >= interval_minutes:
                logger.info(f"Generating proactive message for {name} ({tg_id}) (Elapsed: {elapsed_mins:.1f}m > Interval: {interval_minutes}m)")
                
                # 1. Ask the LLM to generate a creative, contextual check-in
                proactive_msg = await get_llm_response(tg_id, "", name, is_proactive=True)
                
                # 2. Log it
                log_chat_message(tg_id, "assistant", f"[Proactive] {proactive_msg}")
                
                # 3. Send it
                keyboard = [
                    [InlineKeyboardButton("🎮 Play Smile Checker", web_app=WebAppInfo(url=f"{SMILE_APP_URL}?uid={tg_id}"))],
                    [InlineKeyboardButton("🏃 Play Mobility Workout", web_app=WebAppInfo(url=f"{WORKOUT_APP_URL}?uid={tg_id}"))]
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)
                await context.bot.send_message(chat_id=tg_id, text=proactive_msg, reply_markup=reply_markup)
                
                # 4. Reset their timer
                update_last_prompted(tg_id)
                logger.info(f"Successfully sent proactive message to {name}.")
        except Exception as e:
            logger.error(f"Failed to check/send scheduled message for {tg_id}: {e}")

async def handle_webapp_data(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles incoming data returned by the WebApp frontend when window.Telegram.WebApp.sendData() is called."""
    web_app_data = update.message.web_app_data
    if not web_app_data:
        return
        
    try:
        import json
        payload = json.loads(web_app_data.data)
        logger.info(f"[WEBAPP EVENT] Received payload: {payload}")
        
        reply = "Data received, but type is unknown!"
        
        if payload.get("type") == "face_symmetry_score":
            score = payload.get("value")
            
            # Use professional prompt combining clinical analysis and personality
            user = get_or_create_patient(update.effective_user.id)
            prompt = (f"The user just completed a clinical facial symmetry test running MediaPipe tasks vision. "
                      f"They scored {score}%. Analyze this result professionally, breakdown what facial symmetry "
                      f"(checking mouth left/right offsets to nose) means for potential neurological or fatigue "
                      f"indicators (like Bell's palsy, stroke signs, or just being extremely tired). "
                      f"Then wrap it up. Write this strictly as Mera (a warm, encouraging digital companion with a very mild playful tsundere streak).")
            
            # Since generating response takes a bit, send 'typing...'
            await context.bot.send_chat_action(chat_id=update.message.chat_id, action="typing")
            
            llm_response = ml_bridge.generate_chat_response(prompt, [])
            
            # Remove the extra inline buttons here by replying with ReplyKeyboardRemove,
            # so the big ugly Mini App button clears after they play
            from telegram import ReplyKeyboardRemove
            
            await update.message.reply_text(
                f"📊 *Health Check Result:*\n{llm_response}", 
                parse_mode="Markdown", 
                reply_markup=ReplyKeyboardRemove()
            )
    except Exception as e:
        logger.error(f"Failed to handle webapp data: {e}")


async def play_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.message.chat_id)
    args = context.args
    
    if args and args[0].lower() == "smile":
        keyboard = [[InlineKeyboardButton("🎮 Play Smile Checker", web_app=WebAppInfo(url=f"{SMILE_APP_URL}?uid={user_id}"))]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text("Here is your Smile Checker game! Show me your best smile! 😊", reply_markup=reply_markup)
    elif args and args[0].lower() == "workout":
        keyboard = [[InlineKeyboardButton("🏃 Play Mobility Workout", web_app=WebAppInfo(url=f"{WORKOUT_APP_URL}?uid={user_id}"))]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text("Here is your Mobility Workout. Let's get those stretches in, you can do it! 💪", reply_markup=reply_markup)
    else:
        keyboard = [
            [InlineKeyboardButton("🎮 Play Smile Checker", web_app=WebAppInfo(url=f"{SMILE_APP_URL}?uid={user_id}"))],
            [InlineKeyboardButton("🏃 Play Mobility Workout", web_app=WebAppInfo(url=f"{WORKOUT_APP_URL}?uid={user_id}"))]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text("Which game do you want to play today? I've got both ready for you! ✨", reply_markup=reply_markup)

def create_bot_app():
    app = ApplicationBuilder().token(TOKEN).build()
    
    app.add_handler(CommandHandler("set_timer", set_timer_command))
    app.add_handler(CommandHandler("play", play_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    app.add_handler(MessageHandler(filters.StatusUpdate.WEB_APP_DATA, handle_webapp_data))
    
    # Proactive Job Queue Configuration
    # In a real app, this runs every morning. For demo, we run it every 5 minutes.
    job_queue = app.job_queue
    job_queue.run_repeating(scheduled_daily_checkin, interval=60, first=20) # Triggers 20 seconds after boot!
    
    return app

if __name__ == "__main__":
    app = create_bot_app()
    logger.info("Starting Telegram Bot (Polling Mode)...")
    app.run_polling()
