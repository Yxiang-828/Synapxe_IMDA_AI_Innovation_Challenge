import sqlite3
import os
from datetime import datetime

DB_PATH = "health_data.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # Table to store user conversations and health scores
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            telegram_id TEXT UNIQUE,
            name TEXT,
            fatigue_score REAL DEFAULT 0.0,
            mobility_score REAL DEFAULT 0.0,
            last_interaction TIMESTAMP
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS interaction_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            telegram_id TEXT,
            interaction_type TEXT, -- "chat", "voice_note", "mini_game"
            score_delta REAL,
            note TEXT,
            timestamp TIMESTAMP
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            telegram_id TEXT,
            role TEXT, -- "user" or "assistant"
            message TEXT,
            timestamp TIMESTAMP
        )
    ''')
    
    # Run migrations to add interval settings if they are missing
    try:
        cursor.execute("ALTER TABLE patients ADD COLUMN interval_minutes INTEGER DEFAULT 1440")
    except sqlite3.OperationalError:
        pass # Column exists
        
    try:
        cursor.execute("ALTER TABLE patients ADD COLUMN last_prompted TIMESTAMP")
    except sqlite3.OperationalError:
        pass # Column exists

    conn.commit()
    conn.close()

def get_or_create_patient(telegram_id: str, name: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, fatigue_score, mobility_score, interval_minutes, name FROM patients WHERE telegram_id = ?", (telegram_id,))
    row = cursor.fetchone()
    
    if not row:
        now_str = datetime.now().isoformat()
        cursor.execute(
            "INSERT INTO patients (telegram_id, name, last_interaction, last_prompted, interval_minutes) VALUES (?, ?, ?, ?, ?)",
            (telegram_id, name, now_str, now_str, 1440) # default interval 24 hours
        )
        conn.commit()
        row = (cursor.lastrowid, 0.0, 0.0, 1440, name)
    else:
        # Update name if it changed
        if row[4] != name and name != "Ah Ma":
            cursor.execute("UPDATE patients SET name = ? WHERE telegram_id = ?", (name, telegram_id))
            conn.commit()
            row = (row[0], row[1], row[2], row[3], name)
    
    conn.close()
    return {"id": row[0], "fatigue_score": row[1], "mobility_score": row[2], "interval_minutes": row[3], "name": row[4]}

def update_patient_interval(telegram_id: str, interval_minutes: int):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE patients SET interval_minutes = ? WHERE telegram_id = ?",
        (interval_minutes, telegram_id)
    )
    conn.commit()
    conn.close()

def update_last_prompted(telegram_id: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE patients SET last_prompted = ? WHERE telegram_id = ?",
        (datetime.now().isoformat(), telegram_id)
    )
    conn.commit()
    conn.close()

def log_interaction(telegram_id: str, interaction_type: str, score_delta: float, note: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO interaction_logs (telegram_id, interaction_type, score_delta, note, timestamp) VALUES (?, ?, ?, ?, ?)",
        (telegram_id, interaction_type, score_delta, note, datetime.now().isoformat())
    )
    # Update last interaction
    cursor.execute("UPDATE patients SET last_interaction = ? WHERE telegram_id = ?", (datetime.now().isoformat(), telegram_id))
    conn.commit()
    conn.close()

def log_chat_message(telegram_id: str, role: str, message: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO chat_history (telegram_id, role, message, timestamp) VALUES (?, ?, ?, ?)",
        (telegram_id, role, message, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()

def get_recent_history(telegram_id: str, limit: int = 6) -> str:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT role, message FROM chat_history WHERE telegram_id = ? ORDER BY timestamp DESC LIMIT ?",
        (telegram_id, limit)
    )
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        return "No recent history."
        
    # Reverse to chronological order
    history = []
    for role, msg in reversed(rows):
        history.append(f"{role.capitalize()}: {msg}")
        
    return "\n".join(history)
