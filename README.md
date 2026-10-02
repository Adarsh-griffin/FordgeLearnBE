# LearnForge Backend (`LearnBack`)

This is the Flask + Python backend API server for **LearnForge**, powering document ingestion, RAG retrieval, LLaMA 3 LLM inference via Groq, GCP summary speech synthesis, AWS S3 audio management, and Serper web scraping.

---

## 🔗 Repositories

- **Frontend Repository (`LearnFront`)**: [https://github.com/Adarsh-griffin/FodgeLearnFront.git](https://github.com/Adarsh-griffin/FodgeLearnFront.git)
- **Backend Repository (`LearnBack`)**: [https://github.com/Adarsh-griffin/FordgeLearnBE.git](https://github.com/Adarsh-griffin/FordgeLearnBE.git)

---

## 🛠️ Tech Stack

- **Python 3.11+ & Flask**: Lightweight, scalable REST API web framework.
- **Groq API**: High-speed LLaMA 3 8B / 70B LLM inference and Whisper speech-to-text (STT).
- **MongoDB & GridFS (PyMongo)**: Storage for user profiles, document content, and audio GridFS metadata.
- **PageIndex Vectorless Retriever**: Semantic context extraction and document retrieval.
- **GCP Text-to-Speech / gTTS**: High-quality studio speech synthesis with fallback support.
- **AWS S3 Storage**: Cloud storage for generated TTS audio files.
- **Serper Web Scraping API**: Search engine integration for real-world educational reference links.

---

## ⚙️ Environment Setup

Create a `.env` file in the root of `LearnBack/`:

```env
# Groq API Keys (comma-separated for multi-key rotation)
GROQ_API_KEYS=your_groq_api_key_1,your_groq_api_key_2

# PageIndex API Key
PAGEINDEX_API_KEY=your_pageindex_api_key

# Serper API Key
SERPER_API_KEY=your_serper_api_key

# MongoDB Connection String & Target Database
MONGO_URI=mongodb+srv://user:password@cluster.mongodb.net/
MONGO_DB_NAME=LearnFodge

# AWS Credentials & S3 Bucket
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=ap-southeast-2
AWS_S3_BUCKET=your_s3_bucket_name

# Clerk Publishable Key (matches frontend)
CLERK_PUBLISHABLE_KEY=pk_test_...

# Allowed Frontend Origins (comma-separated)
ALLOWED_ORIGINS=http://localhost:5173,http://localhost:8080
```

---

## 🚀 Installation & Running Locally

1. **Create and Activate Virtual Environment**:
   ```bash
   # Windows (PowerShell)
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1

   # macOS / Linux
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install Python Dependencies**:
   ```bash
   pip install -r requirement.txt
   ```

3. **Start the Flask Backend Server**:
   ```bash
   python test_groq.py
   ```
   The API server will run on `http://127.0.0.1:5000`.

---

## 🌐 API Endpoints Overview

- `POST /api/upload`: Upload PDF or input topic name to start ingestion.
- `GET /api/processing-status/<file>`: Check status of document processing.
- `GET /api/files`: Get list of uploaded documents.
- `POST /api/qa`: Query the AI Tutor using document RAG context.
- `POST /api/learning-tts`: Generate summary speech audio (GCP TTS / S3 storage).
- `GET /api/tts-audio/<id>`: Stream TTS audio bytes from MongoDB GridFS.
- `GET /api/get_links`: Get web-scraped educational reference links.
- `GET /api/tutor/has-profile`: Check user profile status.
