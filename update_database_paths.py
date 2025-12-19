"""
Script to update words_database.py to use backend API paths
"""
import re
from pathlib import Path

WORDS_DB_PATH = Path(__file__).parent / "words_database.py"

# Read the current database file
with open(WORDS_DB_PATH, 'r', encoding='utf-8') as f:
    content = f.read()

# Mapping of old paths to new backend API paths
# Images: images/filename -> /api/files/images/filename
content = re.sub(
    r'"image_path":\s*"images/([^"]+)"',
    r'"image_path": "/api/files/images/\1"',
    content
)

# Audio: SoundRecordings/filename -> /api/files/audio/filename
content = re.sub(
    r'"word_audio":\s*"SoundRecordings/([^"]+)"',
    r'"word_audio": "/api/files/audio/\1"',
    content
)

content = re.sub(
    r'"cue_audio":\s*"SoundRecordings/([^"]+)"',
    r'"cue_audio": "/api/files/audio/\1"',
    content
)

# Write back to file
with open(WORDS_DB_PATH, 'w', encoding='utf-8') as f:
    f.write(content)

print("Updated words_database.py with backend API paths!")
print("All paths now point to /api/files/images/ and /api/files/audio/")

