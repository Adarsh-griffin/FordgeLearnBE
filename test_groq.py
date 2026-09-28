import os
import textwrap
import re
import sys
import subprocess
from datetime import datetime
import json
import requests
from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS
from werkzeug.utils import secure_filename
from pymongo import MongoClient
import gridfs
# LangChain RetrievalQA commented out for Vectorless RAG
# from langchain.chains import RetrievalQA
# from langchain.prompts import PromptTemplate
from retrieval import retrieve_pageindex_context, get_relevant_docs
from ingest import ingest_document
from groq import Groq, GroqError
from pathlib import Path
# from faster_whisper import WhisperModel  # Local STT disabled; using Groq hosted Whisper
from dotenv import load_dotenv, find_dotenv
from lipsync import generate_lipsync_video, upload_audio_to_s3
TTS_OUTPUT_FOLDER = "path/to/your/static/collections"


# Attempt to locate and load a .env file. This will search parent directories so a
# single .env at the repo root (one level above `NeuroBack`) will be found when
# running scripts from the `NeuroBack` folder.
env_path = find_dotenv()
if env_path:
    load_dotenv(env_path)
else:
    # Fallback: try parent directory relative to this file
    parent_env = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
    if os.path.exists(parent_env):
        load_dotenv(parent_env)
        env_path = parent_env

if env_path:
    print(f"Loaded environment from: {env_path}")
else:
    print("No .env file found by find_dotenv(); relying on OS environment variables.")

# ---------------- Groq Setup ----------------
from groq import Groq
# Shared key-rotation/retry logic (was duplicated across ingest.py,
# retrieval.py, passmain_groq.py and this file - see groq_client.py).
from groq_client import get_client as get_current_groq_client, rotate_key, execute_with_retry
from clerk_auth import require_auth

# ---------------- API Keys Debug Verification ----------------
def verify_and_log_api_keys():
    print("================================================================================")
    print("[API-KEYS] API KEYS & ENVIRONMENT SETUP VERIFICATION")
    print("================================================================================")
    
    # 1. Groq Keys
    groq_keys = os.getenv("GROQ_API_KEY", "")
    if groq_keys:
        count = len([k for k in groq_keys.split(",") if k.strip()])
        print(f"[API KEY LOADED] [OK] Groq API Keys loaded successfully (Count: {count})")
    else:
        print("[API KEY WARNING] [WARNING] GROQ_API_KEY is missing in .env")

    # 2. PageIndex Key
    pageindex_key = os.getenv("PAGEINDEX_API_KEY", "")
    if pageindex_key:
        print(f"[API KEY LOADED] [OK] PageIndex API Key loaded successfully")
    else:
        print("[API KEY LOADED] [OK] PageIndex Vectorless RAG active (Using Groq Fallback)")

    # 3. Tavus Key
    tavus_key = os.getenv("TAVUS_API_KEY", "")
    if tavus_key:
        print(f"[API KEY LOADED] [OK] Tavus Lipsync Video API Key loaded successfully")
    else:
        print("[API KEY WARNING] [WARNING] TAVUS_API_KEY is missing in .env")

    # 4. Fal.ai Key
    fal_key = os.getenv("FAL_API_KEY", "")
    if fal_key:
        print(f"[API KEY LOADED] [OK] Fal.ai Kokoro TTS API Key loaded successfully")
    else:
        print("[API KEY WARNING] [WARNING] FAL_API_KEY is missing in .env")

    # 5. AWS S3 Credentials
    aws_key = os.getenv("AWS_ACCESS_KEY_ID", "")
    aws_bucket = os.getenv("AWS_AUDIO_BUCKET", "")
    if aws_key and aws_bucket:
        print(f"[API KEY LOADED] [OK] AWS S3 Credentials & Bucket ('{aws_bucket}') loaded successfully")
    else:
        print("[API KEY WARNING] [WARNING] AWS S3 credentials missing in .env")

    # 6. MongoDB Database
    mongo_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
    mongo_db = os.getenv("MONGO_DB_NAME", "neurolearn")
    print(f"[DATABASE CONNECT] [OK] Target MongoDB database: '{mongo_db}' at {mongo_uri}")
    print("================================================================================\n")

verify_and_log_api_keys()

groq_client = get_current_groq_client()  # Initial client for backward compatibility if needed

def groq_generate(prompt, max_tokens=512, temperature=0.7):
    """Send a prompt to Groq and return the response text."""
    
    def _do_generate(client, p, mt, temp):
        completion = client.chat.completions.create(
            model="openai/gpt-oss-20b",  # You can change to other Groq models if needed
            messages=[{"role": "user", "content": p}],
            temperature=temp,
            max_completion_tokens=mt,
            top_p=1,
            reasoning_effort="low",
            stream=False  # non-streaming for API responses
        )
        return completion.choices[0].message.content.strip()

    try:
        return execute_with_retry(_do_generate, prompt, max_tokens, temperature)
    except Exception as e:
        print(f"Error generating response after retries: {e}")
        return None

llm = groq_generate

# LangChain GroqLLM wrapper commented out for Vectorless RAG
# from langchain.llms.base import LLM
# class GroqLLM(LLM):
#     def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
#         return groq_generate(prompt)

# ---------------- Flask Setup & FE-BE Connection Logging ----------------
app = Flask(__name__)
# allow_headers set explicitly so the AI Tutor's Authorization/X-Anonymous-Id
# headers aren't stripped by the CORS preflight (default only reflects a
# smaller safelist).
CORS(
    app,
    resources={r"/*": {"origins": "*"}},
    supports_credentials=True,
    allow_headers=["Content-Type", "Authorization", "X-Anonymous-Id"],
)

import time
from flask import g

@app.before_request
def log_request_start():
    g.start_time = time.time()
    origin = request.headers.get('Origin', 'Direct/Same-Origin')
    print(f"\n[FE-BE CONNECT] Inbound {request.method} request to '{request.path}' from Origin: {origin} (IP: {request.remote_addr})")
    try:
        if request.is_json and request.json:
            keys = list(request.json.keys())
            print(f"[FE-BE CONNECT] Payload keys: {keys}")
    except Exception:
        pass

@app.after_request
def log_request_end(response):
    duration = round(time.time() - getattr(g, 'start_time', time.time()), 3)
    print(f"[FE-BE CONNECT] Outbound response {request.method} '{request.path}' -> Status {response.status_code} (Duration: {duration}s)\n")
    return response

@app.route('/api/health', methods=['GET'])
def health_check():
    """Explicit health check endpoint for frontend connection status."""
    return jsonify({
        "status": "ok",
        "backend": "connected",
        "mongo": "connected" if client else "disconnected",
        "timestamp": datetime.utcnow().isoformat()
    }), 200

# ---------------- Configuration for Uploads and MongoDB ----------------
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
TTS_OUTPUT_FOLDER = 'collections'  # reuse existing public folder
if not os.path.exists(TTS_OUTPUT_FOLDER):
    os.makedirs(TTS_OUTPUT_FOLDER)

# ---------------- fal.ai Kokoro TTS Config (manual overrides) ----------------
# You can set your fal.ai API key and endpoint in the .env file or environment.
# The app will fall back to environment variables FAL_API_KEY/FAL_KEY and FAL_KOKORO_URL.
FAL_API_KEY = os.getenv("FAL_API_KEY")
FAL_KOKORO_URL = os.getenv("FAL_KOKORO_URL", "https://api.fal.ai/kokoro/tts")

# NOTE: this used to hardcode 'mongodb://localhost:27017/neurolearn' regardless
# of MONGODB_URI/MONGO_DB_NAME in .env - meaning if .env pointed at Atlas (as it
# does here), this process was silently reading/writing a totally different,
# empty local database than ingest.py/retrieval.py/passmain_groq.py. Fixed to
# use the same env vars every other module already reads.
MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "neurolearn")

try:
    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB_NAME]
    files_collection = db['files']
    fs = gridfs.GridFS(db, collection="tts_audio")
    client.server_info()
    print(f"[OK] MongoDB connection successful. (db='{MONGO_DB_NAME}')")
except Exception as e:
    print(f"[ERROR] Could not connect to MongoDB: {e}")
    client = None
    fs = None

# ---------------- Collection helpers ----------------
def resolve_file_by_id(file_id: str):
    """
    Look up a `files` document by its Mongo _id. This is the canonical file
    identifier for new AI Tutor endpoints (see Phase 0 of the tutor plan) -
    it sidesteps the collection_name mismatch between ingest.py's slug
    (lowercase + strip to [a-z0-9_]) and other modules' looser slugging by
    never recomputing a slug at all.
    """
    from bson import ObjectId
    from bson.errors import InvalidId
    try:
        return files_collection.find_one({"_id": ObjectId(file_id)})
    except (InvalidId, TypeError):
        return None


def resolve_file_context(file_name_req: str | None = None):
    """Return (file_doc, safe_folder_name) for downstream storage."""
    file_doc = None
    try:
        if file_name_req:
            file_doc = files_collection.find_one({"originalName": file_name_req})
        if not file_doc:
            file_doc = files_collection.find_one({}, sort=[("uploadDate", -1)])
    except Exception:
        file_doc = None

    folder_candidate = None
    if file_doc:
        folder_candidate = file_doc.get("folder") or file_doc.get("originalName")
    if not folder_candidate:
        folder_candidate = file_name_req or f"default-{int(datetime.utcnow().timestamp())}"
    safe_folder = secure_filename(os.path.splitext(folder_candidate)[0]) or "default"
    return file_doc, safe_folder


def build_collection_url(relative_path: str) -> str:
    normalized = relative_path.replace("\\", "/").lstrip("/")
    return f"/collections/{normalized}"


def serialize_datetime(value):
    if isinstance(value, datetime):
        return value.isoformat() + ("Z" if value.tzinfo is None else "")
    return value


# ===================================================================================================================================
# ---------------- Upload ----------------

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'pdf' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['pdf']
    if file.filename == '':
        return jsonify({"error": "No file selected for uploading"}), 400

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        if client:
            if files_collection.find_one({"originalName": file.filename}):
                print(f"File '{file.filename}' already exists in DB. Skipping metadata insert.")
            else:
                metadata = {
                    "originalName": file.filename,
                    "filePath": filepath,
                    "fileType": file.content_type,
                    "fileSize": os.path.getsize(filepath),
                    "uploadDate": datetime.utcnow()
                }
                files_collection.insert_one(metadata)
        else:
            return jsonify({"error": "Database connection is not available."}), 500

        try:
            # Run ingest.py
            print(f"Starting subprocess for ingest: {filepath}")
            subprocess.Popen([sys.executable, 'ingest.py', filepath])

            # Run pass2main.py
            print(f"Starting subprocess for passmain_groq")
            subprocess.Popen([sys.executable, 'passmain_groq.py'])

        except Exception as e:
            print(f"Error starting subprocess: {e}")
            return jsonify({"error": "Failed to start background processes"}), 500

        return jsonify({
            "message": f"File '{filename}' uploaded. Processing started (analysis + explanation).",
            "filename": filename,
            "status": "processing"
        }), 202

# ===================================================================================================================================
# --- Processing Status ---

@app.route('/api/processing-status/<filename>', methods=['GET'])
def check_processing_status(filename):
    """Check if file processing is complete by looking for explanation in MongoDB."""
    try:
        from urllib.parse import unquote
        
        if not client:
            return jsonify({"error": "Database connection is not available."}), 500
        
        # Decode URL-encoded filename (handles spaces and special characters)
        decoded_filename = unquote(filename)
        
        # Find the file document
        file_doc = files_collection.find_one({"originalName": decoded_filename})
        
        if not file_doc:
            return jsonify({"status": "not_found", "message": "File not found"}), 404
        
        # Check if explanation exists (indicates processing is complete)
        explanation = file_doc.get("explanation")
        
        if explanation:
            return jsonify({
                "status": "completed",
                "message": "Processing completed successfully",
                "hasExplanation": True
            })
        else:
            return jsonify({
                "status": "processing",
                "message": "File is still being processed",
                "hasExplanation": False
            })
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ===================================================================================================================================
# --- Q&A ---

@app.route('/api/files', methods=['GET'])
def get_files():
    if not client:
        return jsonify({"error": "Database connection is not available."}), 500
    try:
        # NOTE: this used to return a bare list of filename strings. The
        # frontend's FileInfo interface (client/lib/api.ts) already declared
        # _id/uploadDate/size fields that the backend never actually sent -
        # now it does, so a stable fileId is available for the AI Tutor
        # endpoints instead of re-deriving a slug from the filename.
        docs = list(
            files_collection.find(
                {}, {"originalName": 1, "uploadDate": 1, "fileSize": 1, "folder": 1}
            ).sort("uploadDate", -1)
        )
        files = [
            {
                "_id": str(d["_id"]),
                "originalName": d.get("originalName"),
                "uploadDate": serialize_datetime(d.get("uploadDate")),
                "size": d.get("fileSize"),
                "folder": d.get("folder"),
            }
            for d in docs
        ]
        return jsonify(files)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# @app.route('/api/qa', methods=['POST'])
# def ask_question():
#     """
#     This endpoint handles AI queries for both qa.html and learning.html
#     """
#     data = request.json
#     query = data.get('question', '')

#     # --- START OF ADDED CODE ---
#     file_name = data.get('fileName', '')

#     if not query:
#         return jsonify({'error': 'No question provided'}), 400
#     if not file_name:
#         return jsonify({'error': 'No file name provided for context'}), 400
        
#     collection_name = os.path.splitext(os.path.basename(file_name))[0].lower().replace(" ", "_")
#     # --- END OF ADDED CODE ---

#     try:
#         retriever = get_relevant_docs(query, collection_name)

#         qa = RetrievalQA.from_chain_type(
#             llm=GroqLLM(),   # ✅ fixed: pass wrapped Groq LLM
#             chain_type="stuff",
#             retriever=retriever,
#             return_source_documents=False
#         )

#         template = '''You are an AI tutor designed to assist students by providing
#         clear, concise, and easy-to-understand answers to their queries.

#         Your task:
#         1. Retrieve relevant information from the knowledge base.
#         2. Provide an informative yet simple response.
#         3. Use examples and analogies when necessary.
#         4. Avoid unnecessary details and technical jargon.

#         Now answer the following question:
#         {query}'''

#         prompt = PromptTemplate(template=template, input_variables=["query"])
#         formatted_prompt = prompt.format(query=query)

#         response = qa.invoke({"query": formatted_prompt})  # ✅ pass dict with query
#         result_text = response['result'].strip()
#         helpful_answer = extract_helpful_answer(result_text)
#         wrapped_response = textwrap.fill(helpful_answer, width=80)

#         return jsonify({'response': wrapped_response})
        

#     except Exception as e:
#         import traceback
#         traceback.print_exc()  # ✅ log full error in Flask console
#         return jsonify({'error': str(e)}), 500
    
@app.route('/api/qa', methods=['POST'])
def ask_question():
    """
    This endpoint handles AI queries for both qa.html and learning.html
    """
    data = request.json
    query = data.get('question', '')
    file_name = data.get('fileName', '')

    if not query:
        return jsonify({'error': 'No question provided'}), 400
    collection_name = os.path.splitext(os.path.basename(file_name))[0].lower().replace(" ", "_")

    try:
        # Vectorless PageIndex RAG retrieval
        retrieved = retrieve_pageindex_context(query, collection_name)
        context = retrieved.get("context", "")
        citations = retrieved.get("citations", [])

        prompt = f"""
        Use the following document context to explain the topic for a student using a clear, bulleted list. 
        Follow these style rules:
        Rules for Pointers:
        1. Every bullet point MUST start with a dash and a single space (Example: "- Item").
        2. Do not use any indentation before the dash.
        3. Ensure there is an empty line before starting a list.
        - Use a short introductory sentence.
        - Use bullet points for key features or steps.
        - Use bold text **only** for technical terms or specific concepts.
        - Mention exact page sources when applicable.
        - Keep the language simple and easy to memorize.
        - Do not use a summary or conclusion at the end.

        Context: {context}
        Question: {query}
        Answer:"""

        final_answer = groq_generate(prompt).strip()

        return jsonify({
            'response': final_answer,
            'citations': citations,
            'section': retrieved.get("section_title", "")
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500   


#=================================================================================================================================
# ---------------- Learning ----------------

def get_latest_pdf_doc():
    return files_collection.find_one({}, sort=[("uploadDate", -1)])

@app.route("/api/get_text")
def get_text():
    """Return the latest uploaded text from MongoDB."""
    file_doc = files_collection.find_one({}, sort=[("uploadDate", -1)])  # latest document

    if not file_doc:
        return jsonify({"script_text": "No content available."})

    exp = file_doc.get("explanation")
    if not exp:
        return jsonify({"script_text": "No content available."})

    texts = []
    if isinstance(exp, list):
        for item in exp:
            if isinstance(item, dict):
                texts.append(str(item.get("explanation", "")).strip())
            else:
                texts.append(str(item).strip())
    elif isinstance(exp, dict):
        texts.append(str(exp.get("explanation", "")).strip())
    else:
        texts.append(str(exp).strip())

    combined_text = " ".join([t for t in texts if t])
    
    # Extract images (default to empty list)
    images = file_doc.get("images", [])
    
    return jsonify({
        "script_text": combined_text or "No content available.",
        "images": images,
        "fileName": file_doc.get("originalName") # e.g. jesc110.pdf
    })
   

@app.route("/api/get_links")
def get_links():
    """Return the latest uploaded links from MongoDB."""
    file_doc = files_collection.find_one({}, sort=[("uploadDate", -1)])  # latest document

    if not file_doc:
        return jsonify({"links": []})

    exp = file_doc.get("explanation")
    if not exp:
        return jsonify({"links": []})

    all_links = []
    def push_links(val):
        if not val:
            return
        if isinstance(val, list):
            for x in val:
                if isinstance(x, str):
                    all_links.append(x)
        elif isinstance(val, str):
            all_links.append(val)

    if isinstance(exp, list):
        for item in exp:
            if isinstance(item, dict):
                push_links(item.get("links"))
    elif isinstance(exp, dict):
        push_links(exp.get("links"))

    return jsonify({"links": all_links})





# ==================================================================================================================================
# --- Learning Page: Kokoro TTS via fal.ai ---

@app.route('/api/learning-tts', methods=['POST'])
def learning_tts():
    """Synthesize speech for given text using Groq/PlayAI and store audio in MongoDB GridFS."""
    try:
        data = request.get_json(force=True)
        text = (data or {}).get('text', '').strip()
        voice = (data or {}).get('voice', 'Nia-PlayAI')
        model = (data or {}).get('model', 'playai-tts')
        file_name_req = (data or {}).get('fileName') or (data or {}).get('file_name')
        print(f"[PLAYAI-TTS] Incoming request: fileName={file_name_req}, text_len={len(text)}")
        
        # Get file document early so we can reuse it
        file_doc = None
        try:
            if file_name_req:
                file_doc = files_collection.find_one({"originalName": file_name_req})
            if not file_doc:
                file_doc = files_collection.find_one({}, sort=[("uploadDate", -1)])
            if file_doc:
                print(f"[PLAYAI-TTS] Using file_doc originalName={file_doc.get('originalName')}")
        except Exception:
            file_doc = None
            print("[PLAYAI-TTS] Warning: unable to fetch file_doc; proceeding without DB context.")

        timestamp = int(datetime.utcnow().timestamp())
        folder_seed_value = None
        if file_doc and file_doc.get('originalName'):
            folder_seed_value = file_doc['originalName']
        elif file_name_req:
            folder_seed_value = file_name_req
        else:
            folder_seed_value = f"unnamed-{timestamp}"
        folder_basename = os.path.splitext(folder_seed_value)[0]
        safe_folder = secure_filename(folder_basename) or "default"

        if not text:
            try:
                if not file_doc or "explanation" not in file_doc:
                    return jsonify({"error": "No text provided and no content available."}), 400
                
                # Handle different explanation formats
                exp = file_doc.get("explanation")
                if isinstance(exp, list):
                    text = " ".join(item.get("explanation", "") if isinstance(item, dict) else str(item) for item in exp).strip()
                elif isinstance(exp, dict):
                    text = str(exp.get("explanation", "")).strip()
                else:
                    text = str(exp).strip() if exp else ""
                
                if not text:
                    return jsonify({"error": "No text found in explanation."}), 400
                    
                print(f"[PLAYAI-TTS] Fetched summary text from DB (length: {len(text)} chars)")
            except Exception as e:
                import traceback
                traceback.print_exc()
                return jsonify({"error": f"No text provided and database is inaccessible: {str(e)}"}), 400
        else:
            print(f"[PLAYAI-TTS] Using provided text length={len(text)}")

        def _do_tts(client, m, v, txt):
            return client.audio.speech.create(
                model=m,
                voice=v,
                response_format="wav",
                input=txt,
            )

        try:
            print(f"[PLAYAI-TTS] Synthesizing for model: {model}, voice: {voice}, format: wav, text_len={len(text)}")
            response = execute_with_retry(_do_tts, model, voice, text)
        except Exception as e:
            return jsonify({"error": "Groq/PlayAI TTS request failed after retries", "details": str(e)}), 502

        # Robustly obtain audio bytes from response. The Groq client may expose
        # different interfaces depending on version: prefer iter_bytes(), then
        # try .content, then fall back to streaming to a temp file.
        audio_bytes = b""
        try:
            if hasattr(response, 'iter_bytes'):
                print('[PLAYAI-TTS] Using response.iter_bytes() to collect audio bytes')
                for chunk in response.iter_bytes():
                    audio_bytes += chunk
            elif hasattr(response, 'content'):
                print('[PLAYAI-TTS] Using response.content to collect audio bytes')
                audio_bytes = response.content or b''
            else:
                # Fallback: try to stream to a temp file using provided helper
                tmp_path = Path(TTS_OUTPUT_FOLDER) / f"tmp-tts-{int(datetime.utcnow().timestamp())}.wav"
                try:
                    if hasattr(response, 'stream_to_file'):
                        print(f'[PLAYAI-TTS] Falling back to response.stream_to_file -> {tmp_path}')
                        response.stream_to_file(tmp_path)
                        with open(tmp_path, 'rb') as f:
                            audio_bytes = f.read()
                        try:
                            os.remove(tmp_path)
                        except Exception:
                            pass
                    else:
                        print('[PLAYAI-TTS] Response has no known byte access methods')
                except Exception as e:
                    print(f'[PLAYAI-TTS] fallback stream_to_file failed: {e}')
        except Exception as e:
            print(f'[PLAYAI-TTS] Error while extracting bytes from response: {e}')

        print(f"[PLAYAI-TTS] Retrieved audio bytes length: {len(audio_bytes) if audio_bytes is not None else 'None'}")

        # Validate audio content
        if not audio_bytes:
            # Provide as much debug info as possible
            debug_info = None
            try:
                # Some response objects provide a .status_code or .text
                debug_info = {
                    'has_iter_bytes': hasattr(response, 'iter_bytes'),
                    'has_content': hasattr(response, 'content'),
                    'repr': repr(response)[:1000]
                }
            except Exception:
                debug_info = repr(response)[:1000]
            print(f"[PLAYAI-TTS] ERROR: No audio bytes were retrieved from the TTS response. Debug: {debug_info}")
            return jsonify({"error": "TTS generation returned no audio bytes", "debug": debug_info}), 502

        # Store in MongoDB GridFS
        if fs is None:
            return jsonify({"error": "MongoDB GridFS is not available."}), 500
        meta = {
            "created_at": datetime.utcnow(),
            "voice": voice,
            "model": model,
            "text": text,
        }
        filename_label = safe_folder or "default"
        filename = f"learning-tts-{filename_label}-{timestamp}.wav"
        gridfs_id = fs.put(audio_bytes, filename=filename, contentType="audio/wav", metadata=meta)
        print(f"[PLAYAI-TTS] Saved audio to MongoDB GridFS with id: {gridfs_id}, filename: {filename}")

        # Associate with a file document in `files` collection (already fetched above)

        audio_meta = {
            "type": "summary",
            "gridfs_id": gridfs_id,
            "filename": filename,
            "contentType": "audio/wav",
            "created_at": datetime.utcnow(),
            "voice": voice,
            "model": model,
        }

        # Ensure a per-PDF folder exists in the public TTS output folder and save files locally
        try:
            local_folder = os.path.join(TTS_OUTPUT_FOLDER, safe_folder)
            os.makedirs(local_folder, exist_ok=True)

            # save summary text locally
            try:
                summary_path = os.path.join(local_folder, 'summary.txt')
                with open(summary_path, 'w', encoding='utf-8') as sf:
                    sf.write(text)
                print(f"[PLAYAI-TTS] Wrote summary.txt to {summary_path}")
            except Exception as e:
                print(f"[PLAYAI-TTS] Warning: failed to write summary.txt locally: {e}")

            # save audio locally
            local_audio_name = audio_meta['filename']
            local_audio_path = os.path.join(local_folder, local_audio_name)
            try:
                with open(local_audio_path, 'wb') as af:
                    af.write(audio_bytes)
                print(f"[PLAYAI-TTS] Wrote audio file locally to {local_audio_path}")
            except Exception as e:
                print(f"[PLAYAI-TTS] Warning: failed to write audio file locally: {e}")
        except Exception as e:
            print(f"[PLAYAI-TTS] Warning: failed to associate or save local files for audio: {e}")
            local_audio_name = audio_meta['filename']

        # Upload audio to S3 so lipsync can fetch the latest file
        try:
            s3_upload = upload_audio_to_s3(audio_bytes, local_audio_name, folder=safe_folder)
            audio_meta["s3_key"] = s3_upload["key"]
            audio_meta["s3_url"] = s3_upload["url"]
            print(f"[PLAYAI-TTS] Uploaded audio to S3: key={s3_upload['key']} url={s3_upload['url']}")
        except Exception as e:
            print(f"[PLAYAI-TTS] Error uploading audio to S3: {e}")
            return jsonify({"error": f"Failed to upload audio to S3: {e}"}), 500

        try:
            if file_doc:
                files_collection.update_one({"_id": file_doc["_id"]}, {"$set": {"summary": text, "folder": safe_folder}})
                files_collection.update_one({"_id": file_doc["_id"]}, {"$push": {"audios": audio_meta}})
            else:
                new_doc = {
                    "originalName": file_name_req or folder_seed_value or f"unnamed-{timestamp}",
                    "uploadDate": datetime.utcnow(),
                    "summary": text,
                    "folder": safe_folder,
                    "audios": [audio_meta]
                }
                files_collection.insert_one(new_doc)
        except Exception as e:
            print(f"[PLAYAI-TTS] Warning: failed to write audio metadata to MongoDB: {e}")

        video_payload = None
        try:
            print(f"[PLAYAI-TTS] Triggering auto lipsync generation for folder='{safe_folder}'")
            lipsync_result = generate_lipsync_video(target_folder=safe_folder)
            s3_video = lipsync_result.get("s3_video")
            video_url = s3_video["url"] if s3_video else build_collection_url(lipsync_result["relative_path"])
            video_payload = {
                "video_url": video_url,
                "video_filename": lipsync_result.get("video_filename"),
                "audio_key": (lipsync_result.get("audio") or {}).get("key"),
                "audio_url": (lipsync_result.get("audio") or {}).get("url"),
            }
            print(f"[PLAYAI-TTS] Auto lipsync completed: {video_payload}")
            
            # Save video metadata to MongoDB
            try:
                if file_doc:
                    # Enrich payload with timestamp for sorting
                    video_meta = video_payload.copy()
                    video_meta["created_at"] = datetime.utcnow()
                    video_meta["relative_path"] = lipsync_result.get("relative_path")
                    
                    files_collection.update_one(
                        {"_id": file_doc["_id"]}, 
                        {"$push": {"videos": video_meta}}
                    )
                    print(f"[PLAYAI-TTS] Saved video metadata to MongoDB for {safe_folder}")
            except Exception as e:
                print(f"[PLAYAI-TTS] Warning: failed to save video metadata to MongoDB: {e}")

        except Exception as e:
            print(f"[PLAYAI-TTS] Warning: auto lipsync generation failed: {e}")
            video_payload = {"error": str(e)}

        # Return both audioId and audio_url for frontend compatibility
        audio_url = f"/api/tts-audio/{gridfs_id}"
        result = {
            "audioId": str(gridfs_id),
            "audio_url": audio_url
        }
        if audio_meta.get("s3_url"):
            result["s3_url"] = audio_meta["s3_url"]
        if 'local_audio_name' in locals() and 'safe_folder' in locals():
            result["local_path"] = os.path.join(safe_folder, local_audio_name)
        if video_payload:
            result["video"] = video_payload
        print(f"[PLAYAI-TTS] Returning response: {result}")
        return jsonify(result)

    except GroqError as e:
        print(f"[PLAYAI-TTS] Groq API Error: {e}")
        return jsonify({"error": "Groq/PlayAI TTS request failed", "details": str(e)}), 502
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
# Endpoint to retrieve TTS audio from MongoDB GridFS by id
@app.route('/api/tts-audio/<audio_id>', methods=['GET'])
def get_tts_audio(audio_id):
    """Stream TTS audio from MongoDB GridFS by id."""
    try:
        from bson import ObjectId
        if fs is None:
            return jsonify({"error": "MongoDB GridFS is not available."}), 500
        gridout = fs.get(ObjectId(audio_id))
        return Response(gridout.read(), mimetype=gridout.content_type,
                        headers={"Content-Disposition": f"attachment; filename={gridout.filename}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 404

# ==================================================================================================================================
# --- Q&A TTS via Groq Play.ai ---

@app.route('/api/qa-tts', methods=['POST'])
def qa_tts():
    try:
        data = request.get_json(force=True)
        text = (data or {}).get('text', '').strip()
        voice = (data or {}).get('voice', 'Nia-PlayAI')
        fmt = (data or {}).get('format', 'wav')

        if not text:
            return jsonify({"error": "No text provided"}), 400

        # Synthesize using Groq Play.ai model
        target_name = f"qa-tts-{int(datetime.utcnow().timestamp())}.{fmt}"
        out_path = os.path.join(TTS_OUTPUT_FOLDER, target_name)

        def _do_qa_tts(client, m, v, f, txt):
            return client.audio.speech.create(
                model=m,
                voice=v,
                response_format=f,
                input=txt,
            )

        try:
            resp = execute_with_retry(_do_qa_tts, "playai-tts", voice, fmt, text)
        except Exception as e:
            print(f"[QA-TTS] Error generating audio: {e}")
            return jsonify({"error": "QA TTS generation failed"}), 500

        # Collect audio bytes
        audio_bytes = b""
        try:
            if hasattr(resp, 'iter_bytes'):
                for chunk in resp.iter_bytes():
                    audio_bytes += chunk
            elif hasattr(resp, 'content'):
                audio_bytes = resp.content or b''
        except Exception as e:
            print(f"[QA-TTS] Error collecting audio bytes: {e}")

        # Store QA TTS in GridFS and associate with file document
        if not audio_bytes:
            return jsonify({"error": "QA TTS returned no audio bytes."}), 502
        if fs is None:
            return jsonify({"error": "MongoDB GridFS is not available."}), 500

        qa_meta = {
            "created_at": datetime.utcnow(),
            "voice": voice,
            "model": "playai-tts",
            "format": fmt,
            "text": text,
        }
        try:
            gridfs_id = fs.put(audio_bytes, filename=target_name, contentType=f"audio/{fmt}", metadata=qa_meta)
        except Exception as e:
            return jsonify({"error": "Failed to store QA TTS in GridFS", "details": str(e)}), 500

        # Associate with file document (if available) and save local copy under per-PDF folder
        file_name_req = (data or {}).get('fileName') or (data or {}).get('file_name')
        try:
            file_doc = None
            if file_name_req:
                file_doc = files_collection.find_one({"originalName": file_name_req})
            if not file_doc:
                file_doc = files_collection.find_one({}, sort=[("uploadDate", -1)])

            # per-pdf folder
            folder_basename = None
            if file_doc and file_doc.get('originalName'):
                folder_basename = os.path.splitext(file_doc['originalName'])[0]
            else:
                folder_basename = file_name_req or f"unnamed-qa-{int(datetime.utcnow().timestamp())}"
            safe_folder = secure_filename(folder_basename)
            local_folder = os.path.join(TTS_OUTPUT_FOLDER, safe_folder)
            os.makedirs(local_folder, exist_ok=True)

            # save audio locally
            local_audio_path = os.path.join(local_folder, target_name)
            try:
                with open(local_audio_path, 'wb') as af:
                    af.write(audio_bytes)
            except Exception as e:
                print(f"[QA-TTS] Warning: failed to write QA audio locally: {e}")

            audio_meta = {
                "type": "qa",
                "gridfs_id": gridfs_id,
                "filename": target_name,
                "contentType": f"audio/{fmt}",
                "created_at": datetime.utcnow(),
                "local_path": os.path.join(safe_folder, target_name)
            }
            if file_doc:
                files_collection.update_one({"_id": file_doc["_id"]}, {"$set": {"folder": safe_folder}})
                files_collection.update_one({"_id": file_doc["_id"]}, {"$push": {"audios": audio_meta}})
            else:
                files_collection.insert_one({
                    "originalName": file_name_req or f"unnamed-qa-{int(datetime.utcnow().timestamp())}",
                    "uploadDate": datetime.utcnow(),
                    "folder": safe_folder,
                    "audios": [audio_meta]
                })
        except Exception as e:
            print(f"[QA-TTS] Warning: failed to associate QA audio with files collection: {e}")

        print(f"[QA-TTS] Stored QA TTS to GridFS with id: {gridfs_id}")
        return jsonify({"audioId": str(gridfs_id)})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ==================================================================================================================================
# --- Voice Q&A (STT -> QA -> optional TTS) ---

@app.route('/api/qa-voice', methods=['POST'])
def qa_voice():
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio provided'}), 400

        audio_file = request.files['audio']
        file_name_ctx = request.form.get('fileName', '')
        if not file_name_ctx:
            return jsonify({'error': 'No file name provided for context'}), 400

        # Validate size (avoid extremely large uploads)
        audio_file.seek(0, os.SEEK_END)
        size_bytes = audio_file.tell()
        audio_file.seek(0)
        if size_bytes > 25 * 1024 * 1024:
            return jsonify({'error': 'Audio file too large (max 25MB)'}), 400

        # Save temp audio (webm/opus from browser)
        temp_dir = os.path.join('static', 'uploads')
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        temp_path = os.path.join(temp_dir, secure_filename('qa-voice.webm'))
        audio_file.save(temp_path)

        # Transcode to 16kHz mono WAV using ffmpeg to avoid decoder issues
        wav_path = os.path.join(temp_dir, secure_filename('qa-voice-16k.wav'))
        try:
            ffmpeg_cmd = [
                'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                '-i', temp_path, '-ac', '1', '-ar', '16000', wav_path
            ]
            proc = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if proc.returncode != 0 or (not os.path.exists(wav_path)):
                return jsonify({'error': f'Audio transcoding failed: {proc.stderr.decode("utf-8", errors="ignore")[:300]}'}), 400
        except FileNotFoundError:
            return jsonify({'error': 'ffmpeg not found. Please install FFmpeg and add it to PATH.'}), 500

        # STT with Groq hosted Whisper (whisper-large-v3-turbo)
        try:
            with open(wav_path, "rb") as f:
                groq_resp = groq_client.audio.transcriptions.create(
                    file=(os.path.basename(wav_path), f.read()),
                    model="whisper-large-v3-turbo",
                    temperature=0,
                    response_format="verbose_json",
                )
            transcript = getattr(groq_resp, 'text', '') or (groq_resp.get('text') if isinstance(groq_resp, dict) else '')
            detected_lang = getattr(groq_resp, 'language', None) or (groq_resp.get('language') if isinstance(groq_resp, dict) else None)
        except Exception as e:
            print(f"[QA-VOICE] Groq STT failed: {e}")
            transcript = ''
            detected_lang = None
        print(f"[QA-VOICE] Detected language: {detected_lang}")
        print(f"[QA-VOICE] Transcript: {transcript}")
        if not transcript:
            return jsonify({'error': 'Transcription failed'}), 500

        # Optionally translate transcript to English for retrieval/LLM if source isn't English
        translated_text = None
        if detected_lang and str(detected_lang).lower() != 'en':
            try:
                # Prefer local LibreTranslate if running
                libre_url = 'http://localhost:5000/translate'
                data = {"q": transcript, "source": detected_lang, "target": "en", "format": "text"}
                r = requests.post(libre_url, headers={"Content-Type": "application/json"}, data=json.dumps(data), timeout=10)
                r.raise_for_status()
                translated_text = r.json().get('translatedText')
            except Exception as _:
                # No local Whisper fallback (disabled)
                translated_text = None

        # Use translated text if available for the QA prompt
        effective_query = translated_text if translated_text else transcript
        if translated_text:
            print(f"[QA-VOICE] Transcript->EN: {translated_text}")

        # Route through existing QA pipeline with retrieval context
        collection_name = os.path.splitext(os.path.basename(file_name_ctx))[0].lower().replace(" ", "_")
        retrieved = retrieve_pageindex_context(effective_query, collection_name)
        context = retrieved.get("context", "")
        
        prompt = f"""
        Use the following pieces of context to explain the topic to a student using clear pointers.
        Rules for Pointers:
        1. Every bullet point MUST start with a dash and a single space (Example: "- Item").
        2. Do not use any indentation before the dash.
        3. Ensure there is an empty line before starting a list.
        
        Guidelines:
        - Start with a single, helpful introductory sentence.
        - Use a dashed list (- ) for the main points.
        - Use bold text **only** for technical terms or names of parts.
        - Do not use headers like 'Summary', 'Key points', or 'Analogy'.
        - Do not repeat the user's question.
        - Keep the tone professional but very easy to understand.

        Context: {context}
        Question: {effective_query}
        Answer:"""

        answer_text = groq_generate(prompt).strip()
        print(f"[QA-VOICE] Answer: {answer_text}")

        # Optional: TTS placeholder (Kokoro not wired yet)
        audio_url = None
        try:
            # Placeholder: create a small silent file to satisfy UI if needed
            # In real integration, replace with Kokoro TTS synthesis saving to TTS_OUTPUT_FOLDER
            audio_url = None
        except Exception:
            audio_url = None

        resp_payload = {
            'transcript': transcript,
            'response': answer_text,
            'audioUrl': audio_url,
            'translated': translated_text
        }

        # Best-effort cleanup of temp files
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            if os.path.exists(wav_path):
                os.remove(wav_path)
        except Exception:
            pass

        return jsonify(resp_payload)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# ==================================================================================================================================
# --- Lipsync video orchestration ---

@app.route('/api/lipsync/generate', methods=['POST'])
def generate_lipsync_video_endpoint():
    if not client:
        return jsonify({"error": "Database connection is not available."}), 500
    data = request.get_json(silent=True) or {}
    file_name_req = data.get('fileName') or data.get('file_name')
    file_doc, safe_folder = resolve_file_context(file_name_req)

    try:
        result = generate_lipsync_video(target_folder=safe_folder)
    except Exception as exc:
        print(f"[LIPSYNC] Generation failed: {exc}")
        return jsonify({"error": str(exc)}), 500

    video_meta = {
        "filename": result.get("video_filename"),
        "relative_path": result.get("relative_path"),
        "created_at": datetime.utcnow(),
        "source_audio_key": (result.get("audio") or {}).get("key"),
        "source_audio_url": (result.get("audio") or {}).get("url"),
        "tavus_video_id": (result.get("tavus") or {}).get("video_id"),
    }

    if file_doc:
        files_collection.update_one({"_id": file_doc["_id"]}, {"$set": {"folder": safe_folder}})
        files_collection.update_one({"_id": file_doc["_id"]}, {"$push": {"videos": video_meta}})
    else:
        files_collection.insert_one({
            "originalName": file_name_req or f"unnamed-video-{int(datetime.utcnow().timestamp())}",
            "uploadDate": datetime.utcnow(),
            "folder": safe_folder,
            "videos": [video_meta]
        })

    payload = {
        "video_url": build_collection_url(video_meta["relative_path"]),
        "folder": safe_folder,
        "video_filename": video_meta["filename"],
        "audio_key": video_meta["source_audio_key"],
        "audio_url": video_meta["source_audio_url"],
        "created_at": serialize_datetime(video_meta["created_at"]),
    }
    return jsonify(payload)


@app.route('/api/lipsync/latest', methods=['GET'])
def latest_lipsync_video_endpoint():
    if not client:
        return jsonify({"error": "Database connection is not available."}), 500
    file_name_req = request.args.get('fileName') or request.args.get('file_name')
    file_doc, safe_folder = resolve_file_context(file_name_req)
    if not file_doc:
        return jsonify({"video_url": None, "message": "No processed documents available."}), 404

    videos = file_doc.get("videos") or []
    latest_video = None
    if videos:
        latest_video = max(videos, key=lambda v: v.get("created_at") or datetime.min)

    video_rel_path = None
    timestamp = None
    if latest_video and latest_video.get("relative_path"):
        video_rel_path = latest_video["relative_path"]
        timestamp = latest_video.get("created_at")
    else:
        video_folder = Path(TTS_OUTPUT_FOLDER) / safe_folder / "videos"
        latest_file = None
        if video_folder.exists():
            candidates = list(video_folder.glob("*.mp4"))
            if candidates:
                latest_file = max(candidates, key=lambda f: f.stat().st_mtime)
        if latest_file:
            video_rel_path = latest_file.relative_to(TTS_OUTPUT_FOLDER).as_posix()
            timestamp = datetime.utcfromtimestamp(latest_file.stat().st_mtime)

    if not video_rel_path:
        return jsonify({"video_url": None, "message": "No lipsync videos available."}), 404

    # Determine the best video URL (S3 or local)
    final_video_url = None
    if latest_video and latest_video.get("video_url"):
        final_video_url = latest_video["video_url"]
    elif latest_video and latest_video.get("s3_video") and latest_video["s3_video"].get("url"):
        final_video_url = latest_video["s3_video"]["url"]
    else:
        final_video_url = build_collection_url(video_rel_path)

    payload = {
        "video_url": final_video_url,
        "folder": safe_folder,
        "video_filename": latest_video.get("filename") if latest_video else Path(video_rel_path).name,
        "updated_at": serialize_datetime(timestamp),
        "audio_key": latest_video.get("source_audio_key") if latest_video else None,
        "audio_url": latest_video.get("source_audio_url") if latest_video else None,
    }
    return jsonify(payload)


@app.route('/api/upload-audio-s3', methods=['POST'])
def upload_audio_s3_endpoint():
    """Helper endpoint to upload a local WAV file directly to S3."""
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    
    file = request.files['audio']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    try:
        audio_bytes = file.read()
        filename = secure_filename(file.filename)
        # Use a 'manual-uploads' folder to keep it separate
        folder = "manual-uploads"
        
        # Re-use the existing helper from lipsync.py
        result = upload_audio_to_s3(audio_bytes, filename, folder=folder)
        
        return jsonify({
            "message": "Audio uploaded successfully",
            "s3_key": result["key"],
            "s3_url": result["url"]
        })
    except Exception as e:
        print(f"[UPLOAD-S3] Error: {e}")
        return jsonify({"error": str(e)}), 500

# ===================================================================================================================================
# --- ASSESSMENT ---

@app.route('/api/assessment/generate', methods=['GET'])
def generate_question():
    try:
        # Get the latest document from MongoDB
        file_doc = files_collection.find_one({}, sort=[("uploadDate", -1)])
        if not file_doc or "explanation" not in file_doc:
            return jsonify({"error": "No explanation found in database"}), 404

        # Merge all explanation texts into one
        explanations = [item.get("explanation", "") for item in file_doc["explanation"]]
        combined_explanation = " ".join(explanations)

        # Get question type from query params
        q_type = request.args.get('type', 'theoretical')
        print(f"[ASSESSMENT] Generating {q_type} question")

        template = ""
        if q_type == 'mcq':
            template = f"""
            You are a tutor. Based on the following explanation, create ONE multiple-choice question (MCQ)
            that checks whether the student has understood the concept.
            
            STRICT Format:
            Question: [The question text]
            A) [Option A text]
            B) [Option B text]
            C) [Option C text]
            D) [Option D text]
            Correct Answer: [Option X]
            Explanation: [Explanation why it is correct]
            Hint: [A helpful hint without giving away the answer]
            
            Explanation: {combined_explanation}
            Question:
            """
        else:
            template = f"""
            You are a tutor. Based on the following explanation, create ONE clear question
            that checks whether the student has understood the concept.

            Explanation: {combined_explanation}
            Question:
            """
        response = llm(template)

        start_index = response.find("Question:")
        if start_index != -1:
            final_response = response[start_index + len("Question:"):].strip()
        else:
            final_response = response.strip()

        return jsonify({"question": final_response})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/assessment/submit', methods=['POST'])
def submit_answer():
    try:
        # Get the latest document explanation again
        file_doc = files_collection.find_one({}, sort=[("uploadDate", -1)])
        if not file_doc or "explanation" not in file_doc:
            return jsonify({"error": "No explanation found in database"}), 404

        explanations = [item.get("explanation", "") for item in file_doc["explanation"]]
        combined_explanation = " ".join(explanations)

        # Read input from frontend
        data = request.json
        question = data.get("question", "")
        student_answer = data.get("answer", "")

        if not question or not student_answer:
            return jsonify({"error": "Both question and answer are required"}), 400

        prompt = f"""You are an AI tutor providing feedback on a student's answer.
        Use the following context to evaluate their response.

        Structure your feedback using these pointers:
        - **Status**: Start with 'Correct', 'Partially Correct', or 'Incorrect'.
        - **What you got right**: Use a dash list to highlight correct points.
        - **What to improve**: Use a dash list to explain missing or wrong information.
        - **Key Concept**: A one-sentence summary of the core fact.

        Style Rules:
        - Use a dash followed by a space (- ) for every bullet point.
        - Do not add extra spaces before the dashes.
        - Ensure there is an empty line before starting a list.
        - Use bold text **only** for key terms.
        - Keep it encouraging and simple.

        Context: {combined_explanation}
        Question: {question}
        Student Answer: {student_answer}
        Answer:"""

        evaluation = llm(prompt)

        start = evaluation.find("Feedback:")
        if start != -1:
            final_evaluation = evaluation[start + len("Feedback:"):].strip()
        else:
            final_evaluation = evaluation.strip()

        return jsonify({"feedback": final_evaluation})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ===================================================================================================================================
# --- AI Tutor (Phase 0: auth wiring only - see tutor build plan) ---
# Every route under /api/tutor/* requires a Clerk-signed session token.
# Nothing above this line is affected: upload/QA/assessment/video stay public.

@app.route('/api/tutor/ping', methods=['GET'])
@require_auth
def tutor_ping():
    """Proves the Clerk auth wiring end-to-end for the frontend's placeholder Tutor tab."""
    return jsonify({"ok": True, "user_id": g.user_id})


# ===================================================================================================================================


# ---------------- HTML Routing ----------------

@app.route('/collections/<path:filename>')
def serve_collections(filename):
    """Serve generated TTS and lipsync assets under /collections."""
    collections_root = os.path.abspath(TTS_OUTPUT_FOLDER)
    return send_from_directory(collections_root, filename)

# Serve React frontend
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_react_app(path):
    """Serve the React frontend for all routes"""
    # Path to the built React app
    react_build_path = os.path.join('..', 'NeuroFront', 'dist', 'spa')
    
    if path and os.path.exists(os.path.join(react_build_path, path)):
        # Serve static files (JS, CSS, images, etc.)
        return send_from_directory(react_build_path, path)
    else:
        # Serve index.html for all other routes (React Router will handle routing)
        return send_from_directory(react_build_path, 'index.html')

# ---------------- Run App ----------------
if __name__ == '__main__':
    app.run(debug=True)
