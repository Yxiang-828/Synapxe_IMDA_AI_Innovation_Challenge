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
    conn.commit()
    conn.close()

def get_or_create_patient(telegram_id: str, name: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, fatigue_score, mobility_score FROM patients WHERE telegram_id = ?", (telegram_id,))
    row = cursor.fetchone()
    
    if not row:
        cursor.execute(
            "INSERT INTO patients (telegram_id, name, last_interaction) VALUES (?, ?, ?)",
            (telegram_id, name, datetime.now().isoformat())
        )
        conn.commit()
        row = (cursor.lastrowid, 0.0, 0.0)
    
    conn.close()
    return {"id": row[0], "fatigue_score": row[1], "mobility_score": row[2]}

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
