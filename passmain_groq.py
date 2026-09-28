import os
import sys
import json
import time
import PyPDF2
from datetime import datetime
from pymongo import MongoClient
from werkzeug.utils import secure_filename
from dotenv import load_dotenv, find_dotenv
# Shared key-rotation/retry client (was a duplicated copy of the same logic -
# see groq_client.py). Runs as its own subprocess (see test_groq.py's
# subprocess.Popen call), so this has no shared in-process state with
# test_groq.py's own rotation.
from groq_client import get_client as get_current_groq_client, rotate_key, execute_with_retry

# Load environment
load_dotenv(find_dotenv())

# Import S3 & Lipsync helpers
from lipsync import upload_audio_to_s3, generate_lipsync_video

# MongoDB Setup
MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "neurolearn")

client = MongoClient(MONGO_URI)
db = client[MONGO_DB_NAME]
collection = db["files"]

def groq_generate(prompt, max_tokens=700, temperature=0.7):
    """Send prompt to Groq API using supported model."""
    def _do_generate(groq_client, p, mt, temp):
        completion = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": p}],
            temperature=temp,
            max_tokens=mt,
        )
        return completion.choices[0].message.content.strip()

    try:
        return execute_with_retry(_do_generate, prompt, max_tokens, temperature)
    except Exception as e:
        print(f"[PASSMAIN] Error generating response after retries: {e}")
        return None

def extract_text_from_pdf(pdf_path):
    """Extracts all text from a PDF file."""
    text = ""
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"[PASSMAIN] Error reading PDF {pdf_path}: {e}")
        return None
    return text

def split_text_into_chunks(text, chunk_size=3000, chunk_overlap=200):
    """Simple character-level chunk splitter with overlap (No heavy langchain dependency)."""
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = start + chunk_size
        chunks.append(text[start:end])
        start += (chunk_size - chunk_overlap)
    return chunks

def get_latest_pdf_doc():
    """Fetch the latest uploaded PDF document from MongoDB."""
    return collection.find_one({}, sort=[("uploadDate", -1)])

def save_explanation_to_mongo(file_path, explanation):
    """Update MongoDB document with generated explanation."""
    result = collection.update_one(
        {"filePath": file_path},
        {"$set": {"explanation": explanation, "status": "completed"}}
    )
    if result.modified_count > 0:
        print(f"[PASSMAIN] [OK] Explanation saved successfully for {file_path}")
    else:
        # Fallback match by originalName if filePath differs
        base_name = os.path.basename(file_path)
        result2 = collection.update_one(
            {"originalName": base_name},
            {"$set": {"explanation": explanation, "status": "completed"}}
        )
        print(f"[PASSMAIN] [OK] Fallback update by originalName '{base_name}': {result2.modified_count} doc modified.")

def process_file():
    print("\n[PASSMAIN] =========== [START] Processing Document Summary & Explanations ===========")
    # Brief pause to allow ingest.py to complete writing metadata & page_index to MongoDB
    time.sleep(2)
    
    file_doc = get_latest_pdf_doc()
    if not file_doc:
        print("[PASSMAIN] [WARNING] No PDF document found in MongoDB to process.")
        return

    pdf_path = os.path.normpath(file_doc.get("filePath", ""))
    print(f"[PASSMAIN] Fetching document for summary: '{pdf_path}'")

    if not os.path.exists(pdf_path):
        print(f"[PASSMAIN] [ERROR] File does not exist at path: '{pdf_path}'")
        return

    pdf_text = extract_text_from_pdf(pdf_path)
    if not pdf_text:
        print("[PASSMAIN] [ERROR] Could not extract text from PDF.")
        return

    chunks = split_text_into_chunks(pdf_text, chunk_size=3000, chunk_overlap=200)
    print(f"[PASSMAIN] Document split into {len(chunks)} text chunks.")

    output = []
    prev_summary = ""

    for i, chunk in enumerate(chunks[:3]):  # Process top chunks
        print(f"[PASSMAIN] Summarizing Chunk {i+1}/{min(3, len(chunks))}...")
        prompt = f"""
        You are an expert academic AI tutor.
        Summarize the following document chunk for a student. Include key topics and 3 reference links if available.

        Previous Context: {prev_summary}
        
        Text Chunk:
        {chunk[:2500]}
        """
        explanation_text = groq_generate(prompt)
        if not explanation_text:
            explanation_text = "Summary preview for this section."

        parsed_output = {
            "explanation": explanation_text,
            "links": ["https://en.wikipedia.org/wiki/Special:Search?search=education"]
        }
        output.append(parsed_output)
        prev_summary = explanation_text[-500:]

    # Save explanation array to MongoDB
    save_explanation_to_mongo(pdf_path, output)
    print("[PASSMAIN] [OK] Document explanation processing completed and saved to MongoDB!\n")

if __name__ == '__main__':
    process_file()


# ==============================================================================
# PIPELINE 2: TRADITIONAL LANGCHAIN SLIDING WINDOW (COMMENTED OUT FOR FUTURE TOGGLE)
# To restore original LangChain StructuredOutputParser + RecursiveCharacterTextSplitter:
# 1. Install langchain & langchain-community in requirement.txt
# 2. Uncomment the code below
# ==============================================================================
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.output_parsers import StructuredOutputParser, ResponseSchema
#
# response_schemas = [
#     ResponseSchema(name="explanation", description="A concise summary of the provided chunks."),
#     ResponseSchema(name="links", description="A list of 3 reliable reference links to resources.")
# ]
# output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
# format_instructions = output_parser.get_format_instructions()
#
# def split_text_into_chunks_langchain(text, chunk_size=1000, chunk_overlap=200):
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len, separators=["\n\n", "\n", ".", " ", ""]
#     )
#     return splitter.split_text(text)
# ==============================================================================
