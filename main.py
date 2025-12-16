"""
FastAPI backend for Gamified Anomia Rehabilitation App
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict
import speech_recognition as sr
import tempfile
import os
import json
from datetime import datetime
from pathlib import Path
from database import (
    create_user, get_user_by_username, get_user_by_id,
    get_all_users, verify_password, update_user_progress
)

app = FastAPI(title="Anomia Rehabilitation API", version="1.0.0")

# CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load words database
WORDS_DB_PATH = Path(__file__).parent / "words_database.py"

# Simple in-memory storage (replace with database in production)
sessions_store = {}
user_progress_store = {}

# Pydantic models
class AudioResponse(BaseModel):
    success: bool
    result: str  # "correct" or "incorrect"
    confidence: float
    word_id: int
    transcription: str
    timestamp: str

class WordCard(BaseModel):
    id: int
    word: str
    word_audio: str
    cue_audio: str
    word_hint_audio: str
    semantic_cue: str
    frequency_level: int
    image_path: str

class SessionRecord(BaseModel):
    word_id: int
    result: str
    cue_level: int
    response_time_ms: int
    points_earned: int
    timestamp: str

class RegisterRequest(BaseModel):
    username: str
    password: str
    level_of_severity: Optional[str] = None
    avatar_url: Optional[str] = None

class LoginRequest(BaseModel):
    username: str
    password: str

class UserResponse(BaseModel):
    id: int
    username: str
    level_of_severity: Optional[str]
    current_progress_word_id: int
    total_words_completed: int
    accuracy_percent: float
    avg_response_time_seconds: float
    last_session_date: Optional[str]
    avatar_url: Optional[str]
    total_points: int = 0
    current_level: int = 1
    current_streak: int = 0
    longest_streak: int = 0
    total_exercises_completed: int = 0
    achievements: List[str] = []
    adaptive_time_limit: Optional[int] = None

# Arabic text normalization and matching functions
def normalize_arabic_text(text: str) -> str:
    """
    Normalize Arabic text by handling character variations and removing diacritics.
    Handles: ة/ه, ا/أ/إ/آ, ي/ى, and removes tashkeel (diacritics)
    This ensures that words with different character variations are treated as the same.
    """
    if not text:
        return ""
    
    text = text.strip()
    
    # Normalize ta marbuta (ة) to ha (ه) - common variation in Arabic
    # Both represent the same sound at the end of words
    text = text.replace("ة", "ه")
    
    # Normalize alif variations (ا, أ, إ, آ) -> ا
    # All represent the same long 'a' sound
    text = text.replace("أ", "ا")
    text = text.replace("إ", "ا")
    text = text.replace("آ", "ا")
    
    # Normalize ya variations (ي, ى) -> ي
    # Both represent the same 'ya' sound
    text = text.replace("ى", "ي")
    
    # Normalize other common variations
    # ت and ط are different, but we keep them separate
    # Remove tashkeel (diacritics) - these don't change the base word
    diacritics = "ًٌٍَُِّْ"
    for char in diacritics:
        text = text.replace(char, "")
    
    # Remove shadda (ّ) - indicates doubled consonant
    text = text.replace("ّ", "")
    
    # Remove zero-width characters and whitespace
    text = text.replace(" ", "").replace("\u200C", "").replace("\u200D", "")
    text = text.replace("\u200E", "").replace("\u200F", "")  # Left-to-right marks
    text = text.replace("\u200B", "")  # Zero-width space
    
    # Remove punctuation that might appear in transcriptions
    punctuation = ".,!?;:،"
    for char in punctuation:
        text = text.replace(char, "")
    
    return text

def calculate_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity between two Arabic texts (0.0 to 1.0)
    Uses multiple methods: exact match, Jaccard similarity, and sequence similarity
    """
    if not text1 or not text2:
        return 0.0
    
    # Normalize both texts
    norm1 = normalize_arabic_text(text1)
    norm2 = normalize_arabic_text(text2)
    
    # Exact match after normalization
    if norm1 == norm2:
        return 1.0
    
    # If one is empty after normalization, return 0
    if not norm1 or not norm2:
        return 0.0
    
    # Check substring match (for longer transcriptions that include the word)
    if norm1 in norm2 or norm2 in norm1:
        # If one contains the other, it's likely correct
        return 0.95
    
    # Calculate Jaccard similarity (character set overlap)
    set1 = set(norm1)
    set2 = set(norm2)
    
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    if union == 0:
        jaccard_similarity = 0.0
    else:
        jaccard_similarity = intersection / union
    
    # Calculate sequence similarity (matching characters in order)
    # Use Levenshtein-like approach but simpler
    matches = 0
    min_len = min(len(norm1), len(norm2))
    max_len = max(len(norm1), len(norm2))
    
    # Count exact character matches in position
    for i in range(min_len):
        if norm1[i] == norm2[i]:
            matches += 1
    
    # Also check if characters match but in different positions (for typos)
    char_matches = 0
    norm2_chars = list(norm2)
    for char in norm1:
        if char in norm2_chars:
            char_matches += 1
            norm2_chars.remove(char)  # Remove to avoid double counting
    
    sequence_similarity = matches / max_len if max_len > 0 else 0
    char_similarity = char_matches / max_len if max_len > 0 else 0
    
    # Return weighted average of similarities
    # Give more weight to sequence similarity (order matters in Arabic)
    final_similarity = (sequence_similarity * 0.6) + (char_similarity * 0.3) + (jaccard_similarity * 0.1)
    
    return final_similarity

def calculate_adaptive_time_limit(
    current_time_limit: Optional[int],
    initial_time_limit: int,
    recent_sessions: List[Dict]
) -> int:
    """
    Calculate adaptive time limit based on recent performance.
    
    Args:
        current_time_limit: Current stored time limit (ms)
        initial_time_limit: Initial time limit based on severity (ms)
        recent_sessions: List of recent session records
    
    Returns:
        New adaptive time limit in milliseconds
    """
    if not recent_sessions or len(recent_sessions) == 0:
        return initial_time_limit
    
    if current_time_limit is None:
        current_time_limit = initial_time_limit
    
    # Analyze last 3 sessions
    last_sessions = recent_sessions[-3:] if len(recent_sessions) >= 3 else recent_sessions
    
    # Calculate metrics
    total_words = 0
    correct_words = 0
    total_response_time = 0
    words_exceeding_time = 0
    
    for session in last_sessions:
        records = session.get("records", [])
        if isinstance(records, str):
            import json
            records = json.loads(records)
        
        for record in records:
            total_words += 1
            if record.get("result") == "correct":
                correct_words += 1
            response_time = record.get("response_time_ms", 0)
            if response_time:
                total_response_time += response_time
                # Count words that exceeded time limit
                if (record.get("cue_level") == 3 and 
                    record.get("result") == "incorrect" and 
                    response_time >= current_time_limit * 0.9):
                    words_exceeding_time += 1
    
    if total_words == 0:
        return current_time_limit
    
    avg_accuracy = (correct_words / total_words) * 100
    avg_response_time = total_response_time / total_words
    time_exceeded_rate = words_exceeding_time / total_words
    
    # Calculate adjustment
    adjustment = 0
    
    # Improving: high accuracy and fast responses -> decrease time
    if avg_accuracy > 70 and avg_response_time < current_time_limit * 0.6:
        adjustment = -5000  # Decrease by 5 seconds
    # Good performance: decrease slightly
    elif avg_accuracy > 60 and avg_response_time < current_time_limit * 0.7:
        adjustment = -3000  # Decrease by 3 seconds
    # Struggling: many words exceeding time -> increase
    elif time_exceeded_rate > 0.3:
        adjustment = 10000  # Increase by 10 seconds
    # Low accuracy or slow responses -> increase
    elif avg_accuracy < 50 or avg_response_time > current_time_limit * 0.8:
        adjustment = 5000  # Increase by 5 seconds
    # Moderate performance -> slight increase
    elif avg_accuracy < 60 and avg_response_time > current_time_limit * 0.7:
        adjustment = 3000  # Increase by 3 seconds
    
    # Calculate new limit
    new_limit = current_time_limit + adjustment
    
    # Clamp between 20s and 60s
    min_limit = 20000
    max_limit = 60000
    new_limit = max(min_limit, min(max_limit, new_limit))
    
    # Round to nearest 5 seconds
    new_limit = round(new_limit / 5000) * 5000
    
    return int(new_limit)

def is_match(target_word: str, transcription: str, threshold: float = 0.70) -> tuple:
    """
    Check if transcription matches target word using Arabic text normalization.
    Uses semantic/fuzzy matching to handle character variations.
    
    Returns: (is_correct: bool, confidence: float)
    """
    if not target_word or not transcription:
        return False, 0.0
    
    # Normalize both texts
    target_norm = normalize_arabic_text(target_word)
    trans_norm = normalize_arabic_text(transcription)
    
    # Exact match after normalization
    if target_norm == trans_norm:
        return True, 1.0
    
    # Calculate similarity using multiple methods
    similarity = calculate_similarity(target_word, transcription)
    
    # Adjust threshold based on word length (shorter words need higher similarity)
    word_len = len(target_norm)
    if word_len <= 3:
        adjusted_threshold = max(threshold, 0.80)  # Higher threshold for short words
    elif word_len <= 5:
        adjusted_threshold = max(threshold, 0.75)
    else:
        adjusted_threshold = threshold  # More lenient for longer words
    
    # Check if similarity meets threshold
    is_correct = similarity >= adjusted_threshold
    
    return is_correct, similarity

# Speech recognition function
def transcribe_audio_file(audio_file_path: str, language: str = "ar") -> Optional[str]:
    """Transcribe audio from a file using Google Speech Recognition"""
    r = sr.Recognizer()
    
    if not os.path.exists(audio_file_path):
        return None
    
    try:
        # Load audio file
        with sr.AudioFile(audio_file_path) as source:
            audio = r.record(source)
        
        # Transcribe (timeout parameter not supported in this version)
        text = r.recognize_google(audio, language=language)
        return text.strip()
    
    except sr.RequestError as e:
        print(f"Request error: {e}")
        return None
    except sr.UnknownValueError:
        print("Could not understand audio")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

# Convert audio format if needed (webm to wav)
def convert_audio_format(input_path: str, output_path: str) -> bool:
    """Convert audio file to WAV format if needed"""
    try:
        import subprocess
        # Try using ffmpeg if available
        result = subprocess.run(
            ["ffmpeg", "-i", input_path, "-ar", "16000", "-ac", "1", "-y", output_path],
            check=True,
            capture_output=True,
            text=True
        )
        return os.path.exists(output_path)
    except FileNotFoundError:
        print("ffmpeg not found. Please install ffmpeg for audio conversion.")
        return False
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg conversion error: {e.stderr}")
        return False
    except Exception as e:
        print(f"Conversion error: {e}")
        return False

# Load words from database
def load_words_database():
    """Load words from the words_database.py file"""
    words = []
    try:
        # Read the words_database.py file
        db_path = WORDS_DB_PATH
        if db_path.exists():
            with open(db_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Extract the words_database list (simple parsing)
                # This is a simple approach - in production, use proper Python import
                exec_globals = {}
                exec(content, exec_globals)
                words_db = exec_globals.get('words_database', [])
                
                # Image name mapping (database name -> actual filename)
                image_mapping = {
                    "banat.png": "banat.png",
                    "assad.png": "assad.png",
                    "feel.png": "feel.png",
                    "madrasa.png": "madrasa.png",
                    "samaka.png": "samaka.png",
                    "tofaha.png": "tofaha.png",
                    "walad.png": "walad.png",
                    "yaghshel.png": "yaghshel.png",
                    "yaqraa.png": "yaqraa.png",
                    "yashrab.png": "yashrab.png",
                }
                
                for idx, word_data in enumerate(words_db, start=1):
                    # Update image path to use public folder
                    image_path = word_data.get("image_path", "")
                    if image_path:
                        image_name = image_path.split("/")[-1].split("\\")[-1]  # Handle both / and \
                        # Check if it's already in the mapping, otherwise use the filename as-is
                        if image_name.lower() in image_mapping:
                            word_data["image_path"] = f"/images/{image_mapping[image_name.lower()]}"
                        else:
                            # Use the filename directly (handles Arabic filenames)
                            word_data["image_path"] = f"/images/{image_name}"
                    
                    # Update audio paths to use public folder
                    if "word_audio" in word_data and word_data["word_audio"]:
                        audio_path = word_data["word_audio"]
                        audio_name = audio_path.split("/")[-1].split("\\")[-1]  # Handle both / and \
                        word_data["word_audio"] = f"/audio/{audio_name}"
                    if "cue_audio" in word_data and word_data["cue_audio"]:
                        cue_path = word_data["cue_audio"]
                        cue_name = cue_path.split("/")[-1].split("\\")[-1]  # Handle both / and \
                        word_data["cue_audio"] = f"/audio/{cue_name}"
                    
                    words.append({
                        "id": idx,
                        **word_data
                    })
    except Exception as e:
        print(f"Error loading words database: {e}")
        # Return sample words if database can't be loaded
        words = [
            {
                "id": 1,
                "word": "بنات",
                "word_audio": "SoundRecordings/Banat.m4a",
                "cue_audio": "SoundRecordings/Banat_cue.m4a",
                "word_hint_audio": "بَ",
                "semantic_cue": "طفلة… مش وَلَد",
                "frequency_level": 1,
                "image_path": "Pic Storage/Banat.jpg"
            }
        ]
    
    return words

WORDS_DATABASE = load_words_database()

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Anomia Rehabilitation API", "status": "running"}

@app.get("/api/words", response_model=List[WordCard])
async def get_words():
    """Get all word cards"""
    return WORDS_DATABASE

@app.get("/api/words/{word_id}", response_model=WordCard)
async def get_word(word_id: int):
    """Get a specific word by ID"""
    word = next((w for w in WORDS_DATABASE if w["id"] == word_id), None)
    if not word:
        raise HTTPException(status_code=404, detail="Word not found")
    return word

# Word management endpoints
def save_words_database(words_list):
    """Save words back to words_database.py file"""
    try:
        # Convert to the format expected by words_database.py
        words_data = []
        for word in words_list:
            # Handle audio paths - keep /audio/ format for frontend
            word_audio = word.get("word_audio", "")
            cue_audio = word.get("cue_audio", "")
            # Only replace if it starts with /audio/
            if word_audio.startswith("/audio/"):
                word_audio = word_audio.replace("/audio/", "SoundRecordings/")
            if cue_audio.startswith("/audio/"):
                cue_audio = cue_audio.replace("/audio/", "SoundRecordings/")
            
            # Handle image path
            image_path = word.get("image_path", "")
            if image_path.startswith("/images/"):
                image_path = image_path.replace("/images/", "images/")
            
            words_data.append({
                "word": word.get("word", ""),
                "word_audio": word_audio,
                "cue_audio": cue_audio,
                "word_hint_audio": word.get("word_hint_audio", ""),
                "semantic_cue": word.get("semantic_cue", ""),
                "frequency_level": word.get("frequency_level", 1),
                "image_path": image_path
            })
        
        # Write to file with proper escaping
        file_content = "words_database = [\n"
        for word in words_data:
            # Escape quotes and special characters
            def escape_string(s):
                if not s:
                    return '""'
                return json.dumps(s, ensure_ascii=False)
            
            file_content += "    {\n"
            file_content += f'        "word": {escape_string(word["word"])},\n'
            file_content += f'        "word_audio": {escape_string(word["word_audio"])},\n'
            file_content += f'        "cue_audio": {escape_string(word["cue_audio"])},\n'
            file_content += f'        "word_hint_audio": {escape_string(word["word_hint_audio"])},\n'
            file_content += f'        "semantic_cue": {escape_string(word["semantic_cue"])},\n'
            file_content += f'        "frequency_level": {word["frequency_level"]},\n'
            file_content += f'        "image_path": {escape_string(word["image_path"])}\n'
            file_content += "    },\n"
        file_content = file_content.rstrip(",\n") + "\n]\n"
        file_content += "print(words_database)\n"
        
        with open(WORDS_DB_PATH, 'w', encoding='utf-8') as f:
            f.write(file_content)
        
        # Reload the database
        global WORDS_DATABASE
        WORDS_DATABASE = load_words_database()
        return True
    except Exception as e:
        print(f"Error saving words database: {e}")
        return False

@app.post("/api/words")
async def create_word(
    word: str = Form(...),
    word_hint_audio: str = Form(...),
    semantic_cue: str = Form(...),
    frequency_level: int = Form(...),
    image: Optional[UploadFile] = File(None),
    word_audio: Optional[UploadFile] = File(None),
    cue_audio: Optional[UploadFile] = File(None)
):
    """Create a new word"""
    try:
        # Get next ID
        next_id = max([w["id"] for w in WORDS_DATABASE], default=0) + 1
        
        # Handle file uploads
        image_path = ""
        word_audio_path = ""
        cue_audio_path = ""
        
        # Get frontend public directory path
        frontend_public = Path(__file__).parent.parent / "frontend" / "public"
        images_dir = frontend_public / "images"
        audio_dir = frontend_public / "audio"
        
        # Ensure directories exist
        images_dir.mkdir(parents=True, exist_ok=True)
        audio_dir.mkdir(parents=True, exist_ok=True)
        
        if image:
            # Save image
            image_ext = os.path.splitext(image.filename)[1] or ".png"
            image_filename = f"{word.lower().replace(' ', '_')}{image_ext}"
            image_path_full = images_dir / image_filename
            with open(image_path_full, "wb") as f:
                content = await image.read()
                f.write(content)
            image_path = f"/images/{image_filename}"
        
        if word_audio:
            # Save word audio
            audio_ext = os.path.splitext(word_audio.filename)[1] or ".m4a"
            audio_filename = f"{word.replace(' ', '_')}{audio_ext}"
            audio_path_full = audio_dir / audio_filename
            with open(audio_path_full, "wb") as f:
                content = await word_audio.read()
                f.write(content)
            word_audio_path = f"/audio/{audio_filename}"
        
        if cue_audio:
            # Save cue audio
            cue_ext = os.path.splitext(cue_audio.filename)[1] or ".m4a"
            cue_filename = f"{word.replace(' ', '_')} cue{cue_ext}"
            cue_path_full = audio_dir / cue_filename
            with open(cue_path_full, "wb") as f:
                content = await cue_audio.read()
                f.write(content)
            cue_audio_path = f"/audio/{cue_filename}"
        
        # Create new word
        new_word = {
            "id": next_id,
            "word": word,
            "word_audio": word_audio_path,
            "cue_audio": cue_audio_path,
            "word_hint_audio": word_hint_audio,
            "semantic_cue": semantic_cue,
            "frequency_level": frequency_level,
            "image_path": image_path
        }
        
        # Add to database
        WORDS_DATABASE.append(new_word)
        
        # Save to file
        if save_words_database(WORDS_DATABASE):
            return {"success": True, "word": new_word}
        else:
            raise HTTPException(status_code=500, detail="Failed to save word to database")
            
    except Exception as e:
        import traceback
        raise HTTPException(status_code=500, detail=f"Error creating word: {str(e)}\n{traceback.format_exc()}")

@app.put("/api/words/{word_id}")
async def update_word(
    word_id: int,
    word: Optional[str] = Form(None),
    word_hint_audio: Optional[str] = Form(None),
    semantic_cue: Optional[str] = Form(None),
    frequency_level: Optional[int] = Form(None),
    image: Optional[UploadFile] = File(None),
    word_audio: Optional[UploadFile] = File(None),
    cue_audio: Optional[UploadFile] = File(None)
):
    """Update an existing word"""
    try:
        # Find word
        word_index = next((i for i, w in enumerate(WORDS_DATABASE) if w["id"] == word_id), None)
        if word_index is None:
            raise HTTPException(status_code=404, detail="Word not found")
        
        existing_word = WORDS_DATABASE[word_index]
        
        # Get frontend public directory path
        frontend_public = Path(__file__).parent.parent / "frontend" / "public"
        images_dir = frontend_public / "images"
        audio_dir = frontend_public / "audio"
        
        # Update fields
        if word is not None:
            existing_word["word"] = word
        if word_hint_audio is not None:
            existing_word["word_hint_audio"] = word_hint_audio
        if semantic_cue is not None:
            existing_word["semantic_cue"] = semantic_cue
        if frequency_level is not None:
            existing_word["frequency_level"] = frequency_level
        
        # Handle file uploads
        if image:
            image_ext = os.path.splitext(image.filename)[1] or ".png"
            image_filename = f"{existing_word['word'].lower().replace(' ', '_')}{image_ext}"
            image_path_full = images_dir / image_filename
            with open(image_path_full, "wb") as f:
                content = await image.read()
                f.write(content)
            existing_word["image_path"] = f"/images/{image_filename}"
        
        if word_audio:
            audio_ext = os.path.splitext(word_audio.filename)[1] or ".m4a"
            audio_filename = f"{existing_word['word'].replace(' ', '_')}{audio_ext}"
            audio_path_full = audio_dir / audio_filename
            with open(audio_path_full, "wb") as f:
                content = await word_audio.read()
                f.write(content)
            existing_word["word_audio"] = f"/audio/{audio_filename}"
        
        if cue_audio:
            cue_ext = os.path.splitext(cue_audio.filename)[1] or ".m4a"
            cue_filename = f"{existing_word['word'].replace(' ', '_')} cue{cue_ext}"
            cue_path_full = audio_dir / cue_filename
            with open(cue_path_full, "wb") as f:
                content = await cue_audio.read()
                f.write(content)
            existing_word["cue_audio"] = f"/audio/{cue_filename}"
        
        # Update in database
        WORDS_DATABASE[word_index] = existing_word
        
        # Save to file
        if save_words_database(WORDS_DATABASE):
            return {"success": True, "word": existing_word}
        else:
            raise HTTPException(status_code=500, detail="Failed to save word to database")
            
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        raise HTTPException(status_code=500, detail=f"Error updating word: {str(e)}\n{traceback.format_exc()}")

@app.delete("/api/words/{word_id}")
async def delete_word(word_id: int):
    """Delete a word"""
    try:
        # Find word
        word_index = next((i for i, w in enumerate(WORDS_DATABASE) if w["id"] == word_id), None)
        if word_index is None:
            raise HTTPException(status_code=404, detail="Word not found")
        
        # Remove from database
        deleted_word = WORDS_DATABASE.pop(word_index)
        
        # Save to file
        if save_words_database(WORDS_DATABASE):
            return {"success": True, "message": "Word deleted successfully"}
        else:
            # Restore word if save failed
            WORDS_DATABASE.insert(word_index, deleted_word)
            raise HTTPException(status_code=500, detail="Failed to save word to database")
            
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        raise HTTPException(status_code=500, detail=f"Error deleting word: {str(e)}\n{traceback.format_exc()}")

# Authentication endpoints
@app.post("/api/auth/register", response_model=UserResponse)
async def register_user(request: RegisterRequest):
    """Register a new user"""
    try:
        user = create_user(
            username=request.username,
            password=request.password,
            level_of_severity=request.level_of_severity,
            avatar_url=request.avatar_url
        )
        # Normalize achievements field to a list for Pydantic
        achievements = user.get("achievements")
        if achievements is None:
            user["achievements"] = []
        elif isinstance(achievements, str):
            # Stored as JSON text in DB
            try:
                import json
                user["achievements"] = json.loads(achievements)
            except Exception:
                user["achievements"] = []
        # Return user response (password_hash was already removed in create_user)
        user.pop("password_hash", None)
        return UserResponse(**user)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating user: {str(e)}")

@app.post("/api/auth/login", response_model=UserResponse)
async def login_user(request: LoginRequest):
    """Login user"""
    from database import get_user_by_username_with_password
    user = get_user_by_username_with_password(request.username)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    
    if not verify_password(request.password, user.get("password_hash", "")):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    
    # Update last session date
    update_user_progress(
        user_id=user["id"],
        last_session_date=datetime.now().isoformat()
    )
    
    # Get fresh user data without password
    user_data = get_user_by_id(user["id"])
    user_data.pop("password_hash", None)
    return UserResponse(**user_data)

@app.get("/api/auth/users")
async def get_users():
    """Get all users (for login page with avatars)"""
    users = get_all_users()
    return {"users": users}

@app.get("/api/auth/user/{user_id}", response_model=UserResponse)
async def get_user(user_id: int):
    """Get user by ID"""
    user = get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return UserResponse(**user)

@app.put("/api/auth/user/{user_id}/progress")
async def update_user_progress_endpoint(user_id: int, progress_data: dict):
    """Update user progress"""
    from database import update_user_progress, get_user_by_id, get_user_sessions
    try:
        # Ensure achievements is a list, not a string or None
        achievements = progress_data.get("achievements")
        if achievements is not None and not isinstance(achievements, list):
            # If it's a string, try to parse it
            if isinstance(achievements, str):
                import json
                try:
                    achievements = json.loads(achievements)
                except:
                    achievements = []
            else:
                achievements = []
        
        # Calculate adaptive time limit if session was just completed
        adaptive_time_limit = progress_data.get("adaptive_time_limit")
        if adaptive_time_limit is None:
            # Calculate new adaptive time limit based on recent performance
            user = get_user_by_id(user_id)
            if user:
                # Get initial time limit based on severity
                severity_map = {
                    "mild": 30000,
                    "moderate": 45000,
                    "severe": 60000,
                }
                initial_limit = severity_map.get(user.get("level_of_severity", "moderate"), 45000)
                current_limit = user.get("adaptive_time_limit") or initial_limit
                
                # Get recent sessions
                recent_sessions = get_user_sessions(user_id, limit=5)
                
                # Calculate new adaptive limit
                adaptive_time_limit = calculate_adaptive_time_limit(
                    current_limit, initial_limit, recent_sessions
                )
        
        update_user_progress(
            user_id=user_id,
            current_progress_word_id=progress_data.get("current_progress_word_id"),
            total_words_completed=progress_data.get("total_words_completed"),
            accuracy_percent=progress_data.get("accuracy_percent"),
            avg_response_time_seconds=progress_data.get("avg_response_time_seconds"),
            last_session_date=progress_data.get("last_session_date"),
            total_points=progress_data.get("total_points"),
            current_level=progress_data.get("current_level"),
            current_streak=progress_data.get("current_streak"),
            longest_streak=progress_data.get("longest_streak"),
            total_exercises_completed=progress_data.get("total_exercises_completed"),
            achievements=achievements,
            adaptive_time_limit=adaptive_time_limit
        )
        return {"success": True, "adaptive_time_limit": adaptive_time_limit}
    except Exception as e:
        import traceback
        error_detail = f"Error updating progress: {str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_detail)

@app.post("/api/audio/transcribe", response_model=AudioResponse)
async def transcribe_audio(
    audio: UploadFile = File(...),
    word_id: int = Form(...)
):
    """
    Receive audio file, transcribe it, and compare with target word
    """
    # Get target word
    target_word_data = next((w for w in WORDS_DATABASE if w["id"] == word_id), None)
    if not target_word_data:
        raise HTTPException(status_code=404, detail="Word not found")
    
    target_word = target_word_data["word"].strip()
    
    # Determine file extension from content type or filename
    file_ext = ".webm"  # default
    if audio.filename:
        file_ext = os.path.splitext(audio.filename)[1] or ".webm"
    elif audio.content_type:
        if "webm" in audio.content_type:
            file_ext = ".webm"
        elif "wav" in audio.content_type:
            file_ext = ".wav"
        elif "m4a" in audio.content_type or "mp4" in audio.content_type:
            file_ext = ".m4a"
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
        tmp_path = tmp_file.name
        content = await audio.read()
        tmp_file.write(content)
    
    try:
        # Convert to WAV if needed (webm/m4a to wav)
        wav_path = tmp_path.rsplit(".", 1)[0] + ".wav"
        converted = False
        
        # Only convert if not already WAV
        if file_ext.lower() not in [".wav", ".aiff", ".flac"]:
            converted = convert_audio_format(tmp_path, wav_path)
        
        # Use converted file if conversion succeeded, otherwise try original
        audio_path = wav_path if converted and os.path.exists(wav_path) else tmp_path
        
        # Transcribe audio
        transcription = transcribe_audio_file(audio_path, language="ar")
        
        # Clean up temp files
        try:
            os.unlink(tmp_path)
            if os.path.exists(wav_path):
                os.unlink(wav_path)
        except:
            pass
        
        if transcription is None:
            raise HTTPException(
                status_code=500,
                detail="Failed to transcribe audio. Please check your internet connection."
            )
        
        # Use Arabic text matching with normalization
        # This handles variations like: ة/ه, ا/أ, diacritics, etc.
        is_correct, confidence = is_match(target_word, transcription, threshold=0.75)
        
        # Normalize transcription for display
        normalized_transcription = normalize_arabic_text(transcription)
        
        return AudioResponse(
            success=True,
            result="correct" if is_correct else "incorrect",
            confidence=confidence,
            word_id=word_id,
            transcription=transcription,
            timestamp=datetime.now().isoformat()
        )
    
    except HTTPException:
        raise
    except Exception as e:
        # Clean up on error
        try:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            if os.path.exists(wav_path):
                os.unlink(wav_path)
        except:
            pass
        
        raise HTTPException(
            status_code=500,
            detail=f"Error processing audio: {str(e)}"
        )

@app.post("/api/sessions")
async def save_session_endpoint(session_data: dict):
    """Save session data (legacy endpoint - use /api/sessions/create)"""
    from database import create_session
    try:
        user_id = session_data.get("user_id")
        records = session_data.get("records", [])
        stats = session_data.get("stats", {})
        
        if not user_id:
            raise HTTPException(status_code=400, detail="user_id is required")
        
        session_id = create_session(user_id, records, stats)
        return {"success": True, "session_id": session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating session: {str(e)}")

@app.get("/api/sessions/user/{user_id}")
async def get_user_sessions_endpoint(user_id: int, limit: Optional[int] = None):
    """Get all sessions for a user"""
    from database import create_session, get_user_sessions, get_session_by_id
    try:
        sessions = get_user_sessions(user_id, limit)
        return {"sessions": sessions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching sessions: {str(e)}")

@app.get("/api/sessions/{session_id}")
async def get_session_endpoint(session_id: int):
    """Get session data by ID"""
    from database import create_session, get_user_sessions, get_session_by_id
    try:
        session = get_session_by_id(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        return session
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching session: {str(e)}")

# Note: The route order matters! /api/sessions/user/{user_id} must come before /api/sessions/{session_id}
# FastAPI matches routes in order, so more specific routes should be defined first

@app.post("/api/progress")
async def update_progress(progress_data: dict):
    """Update user progress"""
    user_id = progress_data.get("user_id", "default")
    user_progress_store[user_id] = progress_data
    return {"success": True, "user_id": user_id}

@app.get("/api/progress/{user_id}")
async def get_progress(user_id: str):
    """Get user progress"""
    progress = user_progress_store.get(user_id, {
        "total_points": 0,
        "current_level": 1,
        "achievements": [],
        "session_history": [],
        "current_streak": 0,
        "longest_streak": 0,
        "total_exercises_completed": 0
    })
    return progress

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

