# Anomia Rehabilitation Backend API

FastAPI backend for the Gamified Anomia Rehabilitation application.

## Setup

1. **Install Python dependencies:**
```bash
cd backend
pip install -r requirements.txt
```

2. **Install ffmpeg (required for audio conversion):**
   - **Windows:** 
     - Download from https://ffmpeg.org/download.html
     - Or use: `choco install ffmpeg` or `winget install ffmpeg`
     - Add ffmpeg to your PATH
   - **Linux:** `sudo apt-get install ffmpeg`
   - **Mac:** `brew install ffmpeg`

3. **Run the server:**
```bash
# Option 1: Using the run script
python run.py

# Option 2: Using uvicorn directly
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Option 3: Windows batch file
start.bat
```

The API will be available at: `http://localhost:8000`

API documentation: `http://localhost:8000/docs`

## API Endpoints

### GET `/api/words`
Get all word cards

### GET `/api/words/{word_id}`
Get a specific word by ID

### POST `/api/audio/transcribe`
Transcribe audio and compare with target word
- **Form Data:**
  - `audio`: Audio file (webm, wav, etc.)
  - `word_id`: ID of the target word

**Response:**
```json
{
  "success": true,
  "result": "correct" | "incorrect",
  "confidence": 0.95,
  "word_id": 1,
  "transcription": "transcribed text",
  "timestamp": "2024-01-20T10:30:00"
}
```

### POST `/api/sessions`
Save session data

### GET `/api/sessions/{session_id}`
Get session data

### POST `/api/progress`
Update user progress

### GET `/api/progress/{user_id}`
Get user progress

## Notes

- **Internet Required:** The API uses Google Speech Recognition which requires an active internet connection
- **Audio Format:** Supports webm, wav, m4a formats. Webm files are automatically converted to WAV using ffmpeg
- **Temporary Storage:** Audio files are temporarily stored and automatically deleted after processing
- **CORS:** Enabled for localhost development (ports 3000, 5173)
- **Storage:** Currently uses in-memory storage. In production, use a proper database (PostgreSQL, MongoDB, etc.)
- **Language:** Configured for Arabic ("ar") speech recognition

## Troubleshooting

### Connection Timeout Error
If you get connection timeout errors:
1. Check your internet connection
2. Check if firewall is blocking Google services
3. Try using a VPN if Google services are restricted in your region

### ffmpeg Not Found
If audio conversion fails:
1. Make sure ffmpeg is installed
2. Add ffmpeg to your system PATH
3. Restart your terminal/command prompt after installation

### Audio Transcription Fails
- Ensure you have a stable internet connection
- Check that the audio file is not corrupted
- Try with a shorter audio clip (under 10 seconds)

