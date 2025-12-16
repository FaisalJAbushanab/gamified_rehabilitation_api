"""
Quick script to check database contents
"""
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "rehabilitation.db"

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Check user count
cursor.execute("SELECT COUNT(*) FROM users")
user_count = cursor.fetchone()[0]

# Check session count
cursor.execute("SELECT COUNT(*) FROM sessions")
session_count = cursor.fetchone()[0]

# Check session_records count
cursor.execute("SELECT COUNT(*) FROM session_records")
records_count = cursor.fetchone()[0]

print("Database Status:")
print(f"  Users: {user_count}")
print(f"  Sessions: {session_count}")
print(f"  Session Records: {records_count}")

conn.close()

