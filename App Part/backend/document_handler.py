import asyncio
import os
import httpx
import re
import logging
from telegram import Update
from telegram.ext import ContextTypes
from database import log_chat_message, get_recent_history
import fitz
import pymupdf4llm

logger = logging.getLogger(__name__)

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
AI_MODEL = "aisingapore/Llama-SEA-LION-v3.5-8B-R:latest"

def sanitize_telegram_text(text: str) -> str:
    """
    Strip common Markdown markers so Telegram displays clean plain text.
    """
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"__(.*?)__", r"\1", text)
    text = re.sub(r"_(.*?)_", r"\1", text)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    return text

async def summarize_document(user_id: str, doc_text: str, user_name: str, images_b64: list[str] | None = None) -> str:
    recent_history = get_recent_history(user_id, limit=6)
    system_prompt = (
        f"You are 'Mera', an empathetic health digital companion for {user_name}. "
        f"Read the following medical document or report extracted via OCR. "
        f"Summarize the key findings, prescriptions, or warnings in a very short, friendly, and layman-understandable way. "
        f"Do not use complex medical jargon. Speak naturally with a natural mix of English and very mild Singlish (like 'lah', 'hor', 'wah'). "
        f"Here is your recent conversation history with {user_name}, you MUST refer back to it if it helps understand the document context:\n\n{recent_history}"
    )
    truncated_text = doc_text[:4000] + "\n...[truncated]" if len(doc_text) > 4000 else doc_text
    actual_prompt = f"Here is the medical document content:\n\n{truncated_text}\n\nSummarize it now:"

    payload = {
        "model": AI_MODEL,
        "system": system_prompt,
        "prompt": actual_prompt,
        "stream": False,
    }

    if images_b64:
        payload["images"] = images_b64

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(OLLAMA_URL, json=payload)
            resp.raise_for_status()
            raw_text = resp.json().get("response", "I could not read the document.")
            clean_text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
            clean_text = clean_text.replace("</s>", "").strip()
            return sanitize_telegram_text(clean_text)
    except httpx.HTTPStatusError as e:
        # In case images are not supported, retry without them
        logger.warning(f"Ollama HTTP error in summarize_document, retrying without images: {e}")
        if "images" in payload:
            del payload["images"]
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(OLLAMA_URL, json=payload)
                resp.raise_for_status()
                raw_text = resp.json().get("response", "I could not read the document.")
                clean_text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
                clean_text = clean_text.replace("</s>", "").strip()
                return sanitize_telegram_text(clean_text)
        except Exception as inner_e:
            logger.error(f"Ollama fallback error in summarize_document: {inner_e}")
            return "Alamak, my eyes a bit blur today, I cannot read this document properly! 😅"
    except Exception as e:
        logger.error(f"Ollama error in summarize_document: {e}")
        return "Alamak, my eyes a bit blurry today, I cannot read this document properly! 😅"

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    if not message or not message.document:
        return

    doc = message.document
    chat_type = message.chat.type

    # In groups, only react if mentioned or replied to
    if chat_type in ['group', 'supergroup']:
        bot_username = context.bot.username
        bot_username_lower = f"@{bot_username.lower()}"
        text_lower = (message.caption or "").lower()
        is_mentioned = bot_username_lower in text_lower
        is_reply = (message.reply_to_message and message.reply_to_message.from_user.id == context.bot.id)
        if not (is_mentioned or is_reply):
            return

    user_id = str(message.from_user.id)
    user_name = message.from_user.first_name or "Ah Ma"

    # We only handle PDFs for now
    if not doc.file_name.lower().endswith('.pdf'):
        await message.reply_text("I can only read PDF documents right now! Send me a PDF hor! 😉")
        return

    await message.reply_text("Wah, let me wear my reading glasses and look through this PDF... Give me a moment! 👓📄")
    
    # Send chat action typing
    await context.bot.send_chat_action(chat_id=message.chat_id, action='typing')
    
    # Download file
    file = await context.bot.get_file(doc.file_id)
    os.makedirs("temp_uploads", exist_ok=True)
    pdf_path = os.path.join("temp_uploads", doc.file_name)
    await file.download_to_drive(pdf_path)
    
    try:
        # Extract text + a couple of visual snapshots (run heavy work in a thread)
        def _extract_markdown_and_images(p: str) -> tuple[str, list[str]]:
            import base64

            md = ""
            img_b64_list: list[str] = []
            pdf_doc = fitz.open(p)
            num_pages = len(pdf_doc)

            for page_num in range(num_pages):
                page_md = pymupdf4llm.to_markdown(pdf_doc, pages=[page_num])
                md += page_md + "\n"

                # Capture first 2 pages as images for vision context
                if page_num < 2:
                    page = pdf_doc.load_page(page_num)
                    pix = page.get_pixmap(dpi=150)
                    img_bytes = pix.tobytes("png")
                    img_b64_list.append(base64.b64encode(img_bytes).decode("utf-8"))

            return md, img_b64_list

        md_text, img_b64_list = await asyncio.to_thread(_extract_markdown_and_images, pdf_path)
            
        if not md_text.strip():
            await message.reply_text("Aiyo, the document looks empty or I cannot read the text inside! 😔")
            return
            
        # Summarize with both extracted text and (optionally) page snapshots
        summary = await summarize_document(user_id, md_text, user_name, img_b64_list)
        
        # Log to DB
        log_chat_message(user_id, "user", f"[Sent PDF File: {doc.file_name}]")
        log_chat_message(user_id, "assistant", summary)
        
        await message.reply_text(summary)
        
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        await message.reply_text("Sorry ah, my system hit a small error trying to read the document. Try again later! 🙏")
    finally:
        # Cleanup
        if os.path.exists(pdf_path):
            os.remove(pdf_path)

import base64

# Initialize OCR reader lazily
_easyocr_reader = None

def get_easyocr_reader():
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr
        # Disable GPU if it causes issues, but try CPU for now to be safe
        _easyocr_reader = easyocr.Reader(['en'], gpu=False)
    return _easyocr_reader

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    if not message or not message.photo:
        return

    chat_type = message.chat.type
    if chat_type in ['group', 'supergroup']:
        bot_username = context.bot.username
        bot_username_lower = f"@{bot_username.lower()}"
        text_lower = (message.caption or "").lower()
        is_mentioned = bot_username_lower in text_lower
        is_reply = (message.reply_to_message and message.reply_to_message.from_user.id == context.bot.id)
        if not (is_mentioned or is_reply):
            return
    user_id = str(message.from_user.id)
    user_name = message.from_user.first_name or "Ah Ma"

    # Telegram gives multiple sizes, get the largest one
    photo = message.photo[-1]

    await message.reply_text("Wah, you sent a picture! Let me wear my reading glasses and analyze it using both my extracted text and SEA-LION... Give me a moment! 👓📸")
    
    # Send chat action typing
    await context.bot.send_chat_action(chat_id=message.chat_id, action='typing')

    # Download file
    file = await context.bot.get_file(photo.file_id)
    os.makedirs("temp_uploads", exist_ok=True)
    photo_path = os.path.join("temp_uploads", f"{photo.file_id}.jpg")
    await file.download_to_drive(photo_path)

    try:
        # Extract text via Python tool (easyocr) for MAX result
        reader = get_easyocr_reader()
        ocr_result = reader.readtext(photo_path, detail=0)
        extracted_text = " ".join(ocr_result)

        if not extracted_text.strip():
            extracted_text = "[No clear text found in this image by the python tool]"

        # Read image to base64 for LLM
        with open(photo_path, "rb") as img_file:
            img_b64 = base64.b64encode(img_file.read()).decode('utf-8')

        # Run SEA-LION to summarize
        summary = await summarize_photo(user_id, extracted_text, img_b64, user_name)

        # Log to DB
        log_chat_message(user_id, "user", "[Sent Photo]")
        log_chat_message(user_id, "assistant", summary)
        
        await message.reply_text(summary)

    except Exception as e:
        logger.error(f"Error processing photo: {e}")
        await message.reply_text(f"Sorry ah, my system hit a small error trying to read the photo. Try again later! 🙏")
    finally:
        if os.path.exists(photo_path):
            os.remove(photo_path)

async def summarize_photo(user_id: str, extracted_text: str, img_b64: str, user_name: str) -> str:
    recent_history = get_recent_history(user_id, limit=6)
    system_prompt = (
        f"You are 'Mera', an empathetic health digital companion for {user_name}. "
        f"The user sent a photo. A python tool extracted the following text: '{extracted_text}'. "
        f"Combine this python-extracted text with your own visual understanding (if possible) to explain what the image is or what it says. "
        f"Speak naturally with a natural mix of English and very mild Singlish (like 'lah', 'hor', 'wah'). Keep it friendly and concise. "
        f"Here is your recent conversation history with {user_name}, you MUST remember past context and answer ANY questions they ask about what was previously said:\n\n{recent_history}"
    )
    actual_prompt = "Here is the extracted text and image. Tell me what you see!"
    
    payload = {
        "model": AI_MODEL,
        "system": system_prompt,
        "prompt": actual_prompt,
        "images": [img_b64],
        "stream": False
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(OLLAMA_URL, json=payload)
            resp.raise_for_status()
            raw_text = resp.json().get("response", "I could not analyze the image.")
            clean_text = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL).strip()
            clean_text = clean_text.replace('</s>', '').strip()
            return sanitize_telegram_text(clean_text)
    except httpx.HTTPStatusError as e:
        # If model doesn't support images, fallback to text only
        logger.warning(f"Failed with images param. Fallback to text. {e}")
        del payload["images"]
        payload["system"] += " [Note: You could only read the python-extracted text, you cannot see the image directly.]"
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(OLLAMA_URL, json=payload)
            resp.raise_for_status()
            raw_text = resp.json().get("response", "I could not analyze the image.")
            clean_text = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL).strip()
            clean_text = clean_text.replace('</s>', '').strip()
            return sanitize_telegram_text(clean_text)
    except Exception as e:
        logger.error(f"Ollama error in summarize_photo: {e}")
        return "Alamak, my eyes a bit blurry today, I cannot see this photo properly! 😅"
