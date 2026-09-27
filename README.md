# LearnBack - Backend Service (NeuroLearn Platform)

This is the Flask backend service for **NeuroLearn**, an end-to-end multi-modal AI tutoring platform. It powers document ingestion, **Vectorless PageIndex RAG**, Groq LLM reasoning, Text-to-Speech (Fal.ai/PlayAI), Tavus AI avatar lipsync video generation, and automated student assessments.

---

## 🚀 Quick Startup Guide

### Prerequisites
Make sure you have installed on your machine:
* **Python**: v3.10+ recommended (Tested on Python 3.13)
* **MongoDB**: Running locally on `mongodb://localhost:27017/` (or MongoDB Atlas connection string)
* **Git**

---

### Step 1: Clone & Navigate
```bash
git clone <your-repository-url>
cd LearnBack
```

---

### Step 2: Create & Activate Virtual Environment

* **PowerShell (Windows):**
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  ```
* **Command Prompt (CMD - Windows):**
  ```cmd
  python -m venv .venv
  .\.venv\Scripts\activate.bat
  ```
* **macOS / Linux / Git Bash:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

---

### Step 3: Install Dependencies
```bash
pip install -r requirement.txt
```

---

### Step 4: Environment Variables Setup (`.env`)
Create or update the `.env` file in the `LearnBack` folder with your API keys:

```env
# Groq API Keys (Supports single or comma-separated keys for auto rotation)
GROQ_API_KEY=gsk_your_groq_api_key_here

# PageIndex API Key (Optional fallback to Groq key)
PAGEINDEX_API_KEY=pageindex_demo_key

# TTS & Video Generation APIs
FAL_API_KEY=your_fal_ai_key
FAL_KOKORO_URL=https://api.fal.ai/kokoro/tts
TAVUS_API_KEY=your_tavus_api_key
REPLICA_ID=r9fa0878977a

# AWS S3 (For audio hosting)
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
AWS_REGION=ap-southeast-2
AWS_AUDIO_BUCKET=your-bucket-name

# MongoDB Database
MONGODB_URI=mongodb://localhost:27017/
MONGO_DB_NAME=neurolearn
```

---

### Step 5: Start the Backend Server

* **Development Mode:**
  ```bash
  python test_groq.py
  ```
  *The server will start at `http://127.0.0.1:5000`.*

* **Production Mode (Gunicorn / Render):**
  ```bash
  gunicorn test_groq:app
  ```

---

## 📄 Command-Line Document Ingestion (Manual Testing)

To manually ingest a PDF textbook/paper from the command line:
```bash
python ingest.py "path/to/your/document.pdf"
```

---

## 🔀 RAG Architecture Toggling (Vectorless vs. Traditional)

By default, the application runs **Vectorless RAG (PageIndex)** which uses < 80 MB RAM, making it 100% compatible with Render's 500 MB Free Tier.

If you ever want to switch to **Traditional Vector RAG (Qdrant + PyTorch)**, see the complete guide in [PIPELINE_TOGGLE_GUIDE.md](file:///d:/my%20projects/NeuroLearn/LearnForge/LearnBack/PIPELINE_TOGGLE_GUIDE.md).

---

## 📡 Core API Endpoints

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/api/upload` | `POST` | Uploads PDF and triggers automatic Vectorless PageIndex ingestion. |
| `/api/qa` | `POST` | Answers student questions using PageIndex reasoning + Groq with page citations. |
| `/api/qa-voice` | `POST` | Handles voice transcript queries. |
| `/api/lipsync/generate` | `POST` | Generates Tavus AI Avatar video lecture for a summary. |
| `/api/assessment/generate` | `GET` | Generates chapter MCQs or theoretical questions. |
| `/api/assessment/submit` | `POST` | Grades student responses with page-specific study pointers. |
