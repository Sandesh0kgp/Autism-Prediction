"""
Database module for managing user profiles and prediction history.
Uses SQLite for simple, portable storage.
"""

import sqlite3
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple


DB_NAME = "autism.db"


def get_connection():
    """Get database connection."""
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row  # Enable column access by name
    return conn


def init_db():
    """Initialize database tables if they don't exist."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            age INTEGER NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    
    # History table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            input_data TEXT,
            prediction TEXT,
            user_question TEXT,
            bot_response TEXT,
            timestamp TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
    """)
    
    conn.commit()
    conn.close()


def create_user(name: str, age: int) -> str:
    """
    Create a new user profile.
    
    Args:
        name: User's name
        age: User's age
        
    Returns:
        user_id: Unique identifier for the user
    """
    user_id = str(uuid.uuid4())
    created_at = datetime.now().isoformat()
    
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute(
        "INSERT INTO users (user_id, name, age, created_at) VALUES (?, ?, ?, ?)",
        (user_id, name, age, created_at)
    )
    
    conn.commit()
    conn.close()
    
    return user_id


def get_user(user_id: str) -> Optional[Dict]:
    """
    Get user profile by ID.
    
    Args:
        user_id: User's unique identifier
        
    Returns:
        Dictionary with user data or None if not found
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
    row = cursor.fetchone()
    
    conn.close()
    
    if row:
        return dict(row)
    return None


def save_prediction(user_id: str, input_data: Dict, prediction: str) -> int:
    """
    Save a prediction to history.
    
    Args:
        user_id: User's unique identifier
        input_data: Dictionary of input features
        prediction: Prediction result
        
    Returns:
        Record ID
    """
    timestamp = datetime.now().isoformat()
    input_json = json.dumps(input_data)
    
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute(
        """INSERT INTO history 
           (user_id, input_data, prediction, timestamp) 
           VALUES (?, ?, ?, ?)""",
        (user_id, input_json, prediction, timestamp)
    )
    
    record_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return record_id


def save_chat(user_id: str, user_question: str, bot_response: str) -> int:
    """
    Save a chat interaction to history.
    
    Args:
        user_id: User's unique identifier
        user_question: User's question
        bot_response: Bot's response
        
    Returns:
        Record ID
    """
    timestamp = datetime.now().isoformat()
    
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute(
        """INSERT INTO history 
           (user_id, user_question, bot_response, timestamp) 
           VALUES (?, ?, ?, ?)""",
        (user_id, user_question, bot_response, timestamp)
    )
    
    record_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return record_id


def get_user_history(user_id: str) -> List[Dict]:
    """
    Get all history for a user (predictions and chats).
    
    Args:
        user_id: User's unique identifier
        
    Returns:
        List of history records
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute(
        "SELECT * FROM history WHERE user_id = ? ORDER BY timestamp DESC",
        (user_id,)
    )
    
    rows = cursor.fetchall()
    conn.close()
    
    history = []
    for row in rows:
        record = dict(row)
        # Parse JSON input_data if present
        if record.get('input_data'):
            record['input_data'] = json.loads(record['input_data'])
        history.append(record)
    
    return history


def get_latest_prediction(user_id: str) -> Optional[Dict]:
    """
    Get the most recent prediction for a user.
    
    Args:
        user_id: User's unique identifier
        
    Returns:
        Latest prediction record or None
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute(
        """SELECT * FROM history 
           WHERE user_id = ? AND prediction IS NOT NULL 
           ORDER BY timestamp DESC LIMIT 1""",
        (user_id,)
    )
    
    row = cursor.fetchone()
    conn.close()
    
    if row:
        record = dict(row)
        if record.get('input_data'):
            record['input_data'] = json.loads(record['input_data'])
        return record
    return None


# Initialize database on module import
init_db()
