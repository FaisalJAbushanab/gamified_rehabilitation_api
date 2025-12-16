"""
SQLite database setup and models for user management
"""
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict
import hashlib

DB_PATH = Path(__file__).parent / "rehabilitation.db"

def get_db_connection():
    """Get database connection"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def _ensure_user_columns(cursor):
    """
    Ensure all expected columns exist on the users table.

    This is important when the database file was created before
    new columns (like total_points) were added to the schema.
    """
    cursor.execute("PRAGMA table_info(users)")
    existing_cols = {row[1] for row in cursor.fetchall()}

    # Only add columns that are missing
    if "total_points" not in existing_cols:
        cursor.execute("ALTER TABLE users ADD COLUMN total_points INTEGER DEFAULT 0")
    if "current_level" not in existing_cols:
        cursor.execute("ALTER TABLE users ADD COLUMN current_level INTEGER DEFAULT 1")
    if "current_streak" not in existing_cols:
        cursor.execute("ALTER TABLE users ADD COLUMN current_streak INTEGER DEFAULT 0")
    if "longest_streak" not in existing_cols:
        cursor.execute("ALTER TABLE users ADD COLUMN longest_streak INTEGER DEFAULT 0")
    if "total_exercises_completed" not in existing_cols:
        cursor.execute("ALTER TABLE users ADD COLUMN total_exercises_completed INTEGER DEFAULT 0")
    if "achievements" not in existing_cols:
        cursor.execute("ALTER TABLE users ADD COLUMN achievements TEXT DEFAULT '[]'")

def init_database():
    """Initialize database with user and sessions tables and run simple migrations."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Users table with all stats
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            level_of_severity TEXT,
            current_progress_word_id INTEGER DEFAULT 0,
            total_words_completed INTEGER DEFAULT 0,
            accuracy_percent REAL DEFAULT 0.0,
            avg_response_time_seconds REAL DEFAULT 0.0,
            last_session_date TEXT,
            avatar_url TEXT,
            -- Gamification stats
            total_points INTEGER DEFAULT 0,
            current_level INTEGER DEFAULT 1,
            current_streak INTEGER DEFAULT 0,
            longest_streak INTEGER DEFAULT 0,
            total_exercises_completed INTEGER DEFAULT 0,
            achievements TEXT DEFAULT '[]', -- JSON array of achievement IDs
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    
    # Run a lightweight migration for existing databases
    _ensure_user_columns(cursor)
    
    # Sessions table for session history
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            date TEXT NOT NULL,
            total_words INTEGER DEFAULT 0,
            correct_words INTEGER DEFAULT 0,
            incorrect_words INTEGER DEFAULT 0,
            accuracy_percent REAL DEFAULT 0.0,
            avg_response_time_ms REAL DEFAULT 0.0,
            total_points INTEGER DEFAULT 0,
            records TEXT NOT NULL, -- JSON array of session records
            stats TEXT, -- JSON object with session statistics
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    """)
    
    # Session records table for individual word attempts
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS session_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            word_id INTEGER NOT NULL,
            result TEXT NOT NULL, -- "correct" or "incorrect"
            cue_level INTEGER NOT NULL,
            response_time_ms INTEGER NOT NULL,
            points_earned INTEGER DEFAULT 0,
            timestamp TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
        )
    """)
    
    # Create indexes for better performance
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_date ON sessions(date)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_session_records_session_id ON session_records(session_id)")
    
    conn.commit()
    conn.close()

def hash_password(password: str) -> str:
    """Hash password using SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password: str, password_hash: str) -> bool:
    """Verify password against hash"""
    return hash_password(password) == password_hash

def create_user(
    username: str,
    password: str,
    level_of_severity: Optional[str] = None,
    avatar_url: Optional[str] = None
) -> Dict:
    """Create a new user"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        password_hash = hash_password(password)
        now = datetime.now().isoformat()
        
        cursor.execute("""
            INSERT INTO users (
                username, password_hash, level_of_severity,
                current_progress_word_id, total_words_completed,
                accuracy_percent, avg_response_time_seconds,
                last_session_date, avatar_url, total_points,
                current_level, current_streak, longest_streak,
                total_exercises_completed, achievements,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            username, password_hash, level_of_severity,
            0, 0, 0.0, 0.0, None, avatar_url, 0, 1, 0, 0, 0, '[]', now, now
        ))
        
        user_id = cursor.lastrowid
        conn.commit()
        
        # Return user data (without password)
        cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        user = cursor.fetchone()
        user_dict = dict(user)
        # Remove password_hash from returned dict
        user_dict.pop("password_hash", None)
        return user_dict
    except sqlite3.IntegrityError:
        raise ValueError("Username already exists")
    finally:
        conn.close()

def get_user_by_username(username: str) -> Optional[Dict]:
    """Get user by username"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
    user = cursor.fetchone()
    conn.close()
    
    return dict(user) if user else None

def get_user_by_username_with_password(username: str) -> Optional[Dict]:
    """Get user by username (includes password hash for verification)"""
    return get_user_by_username(username)

def get_user_by_id(user_id: int) -> Optional[Dict]:
    """Get user by ID"""
    import json
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    user = cursor.fetchone()
    conn.close()
    
    if user:
        user_dict = dict(user)
        # Remove password_hash from returned dict
        user_dict.pop("password_hash", None)
        # Parse achievements JSON
        if user_dict.get("achievements"):
            try:
                user_dict["achievements"] = json.loads(user_dict["achievements"])
            except:
                user_dict["achievements"] = []
        else:
            user_dict["achievements"] = []
        return user_dict
    return None

def get_all_users() -> List[Dict]:
    """Get all users (for login page)"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, username, avatar_url, last_session_date, 
               total_words_completed, accuracy_percent, level_of_severity
        FROM users
        ORDER BY last_session_date DESC, created_at DESC
    """)
    users = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    return users

def update_user_progress(
    user_id: int,
    current_progress_word_id: Optional[int] = None,
    total_words_completed: Optional[int] = None,
    accuracy_percent: Optional[float] = None,
    avg_response_time_seconds: Optional[float] = None,
    last_session_date: Optional[str] = None,
    total_points: Optional[int] = None,
    current_level: Optional[int] = None,
    current_streak: Optional[int] = None,
    longest_streak: Optional[int] = None,
    total_exercises_completed: Optional[int] = None,
    achievements: Optional[List[str]] = None
):
    """Update user progress"""
    import json
    conn = get_db_connection()
    cursor = conn.cursor()
    
    updates = []
    values = []
    
    if current_progress_word_id is not None:
        updates.append("current_progress_word_id = ?")
        values.append(current_progress_word_id)
    
    if total_words_completed is not None:
        updates.append("total_words_completed = ?")
        values.append(total_words_completed)
    
    if accuracy_percent is not None:
        updates.append("accuracy_percent = ?")
        values.append(accuracy_percent)
    
    if avg_response_time_seconds is not None:
        updates.append("avg_response_time_seconds = ?")
        values.append(avg_response_time_seconds)
    
    if last_session_date is not None:
        updates.append("last_session_date = ?")
        values.append(last_session_date)
    
    if total_points is not None:
        updates.append("total_points = ?")
        values.append(total_points)
    
    if current_level is not None:
        updates.append("current_level = ?")
        values.append(current_level)
    
    if current_streak is not None:
        updates.append("current_streak = ?")
        values.append(current_streak)
    
    if longest_streak is not None:
        updates.append("longest_streak = ?")
        values.append(longest_streak)
    
    if total_exercises_completed is not None:
        updates.append("total_exercises_completed = ?")
        values.append(total_exercises_completed)
    
    if achievements is not None:
        updates.append("achievements = ?")
        values.append(json.dumps(achievements))
    
    if updates:
        updates.append("updated_at = ?")
        values.append(datetime.now().isoformat())
        values.append(user_id)
        
        query = f"UPDATE users SET {', '.join(updates)} WHERE id = ?"
        cursor.execute(query, values)
        conn.commit()
    
    conn.close()

def create_session(user_id: int, records: List[Dict], stats: Dict) -> int:
    """Create a new session and return session ID"""
    import json
    conn = get_db_connection()
    cursor = conn.cursor()
    
    now = datetime.now().isoformat()
    
    cursor.execute("""
        INSERT INTO sessions (
            user_id, date, total_words, correct_words, incorrect_words,
            accuracy_percent, avg_response_time_ms, total_points,
            records, stats, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        user_id,
        now,
        stats.get("total_words", 0),
        stats.get("correct", 0),
        stats.get("incorrect", 0),
        stats.get("accuracy", 0.0),
        stats.get("avg_response_time", 0.0),
        stats.get("total_points", 0),
        json.dumps(records),
        json.dumps(stats),
        now
    ))
    
    session_id = cursor.lastrowid
    
    # Insert session records
    for record in records:
        cursor.execute("""
            INSERT INTO session_records (
                session_id, word_id, result, cue_level,
                response_time_ms, points_earned, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            session_id,
            record.get("word_id"),
            record.get("result"),
            record.get("cue_level"),
            record.get("response_time_ms"),
            record.get("points_earned", 0),
            record.get("timestamp", now)
        ))
    
    conn.commit()
    conn.close()
    
    return session_id

def get_user_sessions(user_id: int, limit: Optional[int] = None) -> List[Dict]:
    """Get all sessions for a user"""
    import json
    conn = get_db_connection()
    cursor = conn.cursor()
    
    query = "SELECT * FROM sessions WHERE user_id = ? ORDER BY date DESC"
    if limit:
        query += f" LIMIT {limit}"
    
    cursor.execute(query, (user_id,))
    sessions = []
    
    for row in cursor.fetchall():
        session = dict(row)
        # Parse JSON fields
        session["records"] = json.loads(session.get("records", "[]"))
        session["stats"] = json.loads(session.get("stats", "{}"))
        sessions.append(session)
    
    conn.close()
    return sessions

def get_session_by_id(session_id: int) -> Optional[Dict]:
    """Get a session by ID"""
    import json
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        session = dict(row)
        session["records"] = json.loads(session.get("records", "[]"))
        session["stats"] = json.loads(session.get("stats", "{}"))
        return session
    return None

# Initialize database on import
init_database()

