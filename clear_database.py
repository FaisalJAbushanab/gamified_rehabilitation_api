"""
Script to clear all data from the database
"""
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "rehabilitation.db"

def clear_all_data():
    """Clear all data from all tables"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        # Disable foreign key constraints temporarily
        cursor.execute("PRAGMA foreign_keys = OFF")
        
        # Clear all tables
        print("Clearing session_records table...")
        cursor.execute("DELETE FROM session_records")
        
        print("Clearing sessions table...")
        cursor.execute("DELETE FROM sessions")
        
        print("Clearing users table...")
        cursor.execute("DELETE FROM users")
        
        # Reset auto-increment counters
        print("Resetting auto-increment counters...")
        cursor.execute("DELETE FROM sqlite_sequence WHERE name IN ('users', 'sessions', 'session_records')")
        
        # Re-enable foreign key constraints
        cursor.execute("PRAGMA foreign_keys = ON")
        
        conn.commit()
        print("\nAll data cleared successfully!")
        print("Database is now empty and ready for fresh data.")
        
    except Exception as e:
        conn.rollback()
        print(f"\nError clearing database: {e}")
        raise
    finally:
        conn.close()

if __name__ == "__main__":
    import sys
    
    print("=" * 50)
    print("Database Cleanup Script")
    print("=" * 50)
    print("\nWARNING: This will delete ALL data from the database!")
    print("   - All users will be deleted")
    print("   - All sessions will be deleted")
    print("   - All session records will be deleted")
    print("\nThis action cannot be undone!")
    
    # Allow non-interactive mode with --yes flag
    if len(sys.argv) > 1 and sys.argv[1] == "--yes":
        clear_all_data()
    else:
        try:
            response = input("\nAre you sure you want to continue? (yes/no): ")
            if response.lower() in ['yes', 'y']:
                clear_all_data()
            else:
                print("\nOperation cancelled.")
        except (EOFError, KeyboardInterrupt):
            print("\nOperation cancelled.")

