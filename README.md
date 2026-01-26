# MeetMind

RAG-based Meeting Transcript Search API — AI-powered search across your meeting transcriptions.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd meetmind
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy example configuration
cp .env.example .env

# Edit .env - select your provider
nano .env
```

### 3. Run

**Local Mode (with Ollama):**
```bash
# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Pull model (first time only)
ollama pull llama3

# Terminal 3: Start API
cd meetmind
source venv/bin/activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Cloud Mode (OpenAI):**
```bash
# Set in .env:
# EMBEDDINGS_PROVIDER=openai
# LLM_PROVIDER=openai
# OPENAI_API_KEY=sk-your-key

cd meetmind
source venv/bin/activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

---

## 📚 API Endpoints

### Health Check
```bash
curl http://localhost:8000/
```

### Upload Transcript
```bash
curl -X POST "http://localhost:8000/transcripts" -F "file=@meeting.txt"
```

### List Transcripts
```bash
curl http://localhost:8000/transcripts
```

### Search
```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"question": "When did we discuss the deadline?"}'
```

### Stats
```bash
curl http://localhost:8000/search/stats
```

### Transcribe Audio (WhisperX)
```bash
curl -X POST "http://localhost:8000/media/transcribe" \
  -F "file=@meeting.mp3" \
  -F "language=ru"
```

---

## 🎤 Audio Transcription Setup

### System Requirements

**ffmpeg** is required for audio processing:

```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows (with Chocolatey)
choco install ffmpeg
```

**CUDA** (recommended for GPU acceleration):
- CUDA 11.8+ and cuDNN required for GPU mode
- CPU mode works but is significantly slower

### WhisperX Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `WHISPERX_MODEL` | `large-v3` | Model: tiny, base, small, medium, large-v2, large-v3 |
| `WHISPERX_DEVICE` | `cuda` | Device: cuda or cpu |
| `WHISPERX_COMPUTE_TYPE` | `float16` | Compute: float16, int8 (GPU), float32 (CPU) |

Example `.env`:
```bash
WHISPERX_MODEL=large-v3
WHISPERX_DEVICE=cuda
WHISPERX_COMPUTE_TYPE=float16
```

---

## ⚙️ Provider Configuration

| Variable | Values | Description |
|----------|--------|-------------|
| `EMBEDDINGS_PROVIDER` | `local`, `openai`, `huggingface` | Embeddings provider |
| `LLM_PROVIDER` | `local`, `openai`, `huggingface` | LLM provider |

### Provider Modes

| Mode | Embeddings | LLM | Requirements |
|------|------------|-----|--------------|
| **local** | HuggingFace model (GPU) | Ollama | GPU 8GB+, Ollama installed |
| **openai** | OpenAI API | GPT-4 | API key |
| **huggingface** | HF Inference API | HF Inference API | HF token |

---

## 📁 Project Structure

```
meetmind/
├── app/
│   ├── main.py              # FastAPI application
│   ├── config.py            # Environment settings
│   ├── api/
│   │   ├── transcripts.py   # Upload & history endpoints
│   │   ├── search.py        # Search endpoint
│   │   └── media.py         # Audio transcription endpoint
│   ├── db/
│   │   ├── database.py      # SQLite connection
│   │   └── models.py        # Data models
│   └── services/
│       ├── embeddings/      # 3 embeddings providers
│       ├── llm/             # 3 LLM providers
│       ├── rag.py           # RAG pipeline
│       └── transcription.py # WhisperX transcription
├── chroma_db/               # Vector DB (auto-created)
├── meetmind.db              # SQLite (auto-created)
├── media_uploads/           # Temp audio files (auto-created)
├── .env.example
├── requirements.txt
└── README.md
```

---

## 🔧 Swagger UI

Available at: http://localhost:8000/docs
