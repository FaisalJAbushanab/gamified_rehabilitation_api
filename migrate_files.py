"""
Script to migrate images and audio files from frontend/public to backend/uploads
"""
import shutil
import os
import sys
from pathlib import Path

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
FRONTEND_PUBLIC = PROJECT_ROOT / "frontend" / "public"
FRONTEND_IMAGES = FRONTEND_PUBLIC / "images"
FRONTEND_AUDIO = FRONTEND_PUBLIC / "audio"

BACKEND_DIR = Path(__file__).parent
BACKEND_UPLOADS = BACKEND_DIR / "uploads"
BACKEND_IMAGES = BACKEND_UPLOADS / "images"
BACKEND_AUDIO = BACKEND_UPLOADS / "audio"

# Create backend upload directories
BACKEND_IMAGES.mkdir(parents=True, exist_ok=True)
BACKEND_AUDIO.mkdir(parents=True, exist_ok=True)

print("Migrating files from frontend/public to backend/uploads...\n")

# Copy images
if FRONTEND_IMAGES.exists():
    print("Copying images...")
    image_files = list(FRONTEND_IMAGES.glob("*"))
    for img_file in image_files:
        if img_file.is_file():
            dst = BACKEND_IMAGES / img_file.name
            shutil.copy2(img_file, dst)
            print(f"  Copied: {img_file.name}")
    print(f"  Total images copied: {len([f for f in image_files if f.is_file()])}\n")
else:
    print(f"Warning: Frontend images folder not found at {FRONTEND_IMAGES}\n")

# Copy audio files
if FRONTEND_AUDIO.exists():
    print("Copying audio files...")
    audio_files = list(FRONTEND_AUDIO.glob("*"))
    for audio_file in audio_files:
        if audio_file.is_file():
            dst = BACKEND_AUDIO / audio_file.name
            shutil.copy2(audio_file, dst)
            print(f"  Copied: {audio_file.name}")
    print(f"  Total audio files copied: {len([f for f in audio_files if f.is_file()])}\n")
else:
    print(f"Warning: Frontend audio folder not found at {FRONTEND_AUDIO}\n")

print("Migration complete!")
print(f"Images location: {BACKEND_IMAGES}")
print(f"Audio location: {BACKEND_AUDIO}")

