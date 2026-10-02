import os
import sys
import json
import time
from datetime import datetime
from bson import ObjectId
from bson.errors import InvalidId
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

# Import S3 & Lipsync helpers - audio/video only; PDFs are never stored in
# S3 (see the note above process_file() for why).
from lipsync import upload_audio_to_s3
# from lipsync import generate_lipsync_video  # Tavus video generation disabled by user request
from gcp_tts import synthesize_speech
from reference_links import fetch_reference_links

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

def get_target_doc(target_id: str | None):
    """
    Resolves the document THIS run should process. test_groq.py's
    /api/upload always passes the exact Mongo _id as argv[1] (see the
    comment at its subprocess.Popen call) - "most recently uploaded" is
    only a fallback for a manual/no-argument run. Blindly using "most
    recent" as the primary lookup used to mean a re-upload of an
    already-existing file (whose uploadDate never changes) could grab a
    completely different, unrelated document's summary, while the file
    that was actually just uploaded sat stuck "processing" forever - a
    real, observed failure, not a hypothetical one.
    """
    if target_id:
        try:
            doc = collection.find_one({"_id": ObjectId(target_id)})
            if doc:
                return doc
            print(f"[PASSMAIN] [WARNING] No document found for id '{target_id}' - falling back to most recent upload.")
        except InvalidId:
            print(f"[PASSMAIN] [WARNING] '{target_id}' isn't a valid document id - falling back to most recent upload.")
    return collection.find_one({}, sort=[("uploadDate", -1)])

def save_explanation_to_mongo(doc_id, explanation):
    """Update MongoDB document with generated explanation, by its exact _id - no string-matching guesswork."""
    result = collection.update_one(
        {"_id": doc_id},
        {"$set": {"explanation": explanation, "status": "completed"}}
    )
    if result.modified_count > 0:
        print(f"[PASSMAIN] [OK] Explanation saved successfully for document _id={doc_id}")
    else:
        print(f"[PASSMAIN] [WARNING] No document matched _id={doc_id} when saving the explanation (deleted since?).")


def save_failure_status_to_mongo(doc_id, error_message):
    """Marks the document as failed instead of leaving it silently stuck at
    status: "processing" forever - every early-return failure path below
    used to just exit without touching MongoDB at all, so a genuine
    ingestion failure (ingest.py crashed, PageIndex timed out, empty PDF
    text, ...) looked IDENTICAL to "still working on it" from the frontend's
    perspective: /api/processing-status only ever distinguished "has an
    explanation" (completed) from "doesn't yet" (processing), with no way to
    tell those two situations apart."""
    try:
        collection.update_one(
            {"_id": doc_id},
            {"$set": {"status": "failed", "error": error_message}}
        )
        print(f"[PASSMAIN] Marked document _id={doc_id} as failed: {error_message}")
    except Exception as e:
        print(f"[PASSMAIN] [WARNING] Could not mark document _id={doc_id} as failed: {e}")

def process_file():
    print("\n[PASSMAIN] =========== [START] Processing Document Summary & Explanations ===========")

    target_id = sys.argv[1] if len(sys.argv) > 1 else None

    # Uses the page text ingest.py already extracted and stored in Mongo
    # (files.pages_text), instead of re-opening the original PDF file at
    # all. This used to independently re-read the PDF from its local disk
    # path, which is exactly what failed for a real upload ('ir_unit_1.pdf'
    # no longer on disk) - PDFs aren't kept in S3 (only TTS audio/video are,
    # per this app's storage design), so local disk is genuinely just a
    # same-request scratch copy; nothing should depend on it existing later.
    #
    # pages_text is written by ingest.py, which can easily take longer than
    # a couple of seconds (PageIndex Cloud polling, or a topic-mode Groq
    # curriculum-generation call) - a single fixed sleep(2)-then-give-up
    # used to fail immediately and permanently in that case (confirmed: a
    # topic-mode upload got stuck "processing" forever from exactly this).
    # Poll instead, same pattern as ingest.py's own PageIndex Cloud wait.
    max_wait_seconds = 90
    poll_interval = 3
    waited = 0
    file_doc = None
    while waited <= max_wait_seconds:
        file_doc = get_target_doc(target_id)
        if file_doc and file_doc.get("pages_text"):
            break
        time.sleep(poll_interval)
        waited += poll_interval

    if not file_doc:
        print("[PASSMAIN] [WARNING] No document found in MongoDB to process.")
        return

    pdf_path = file_doc.get("filePath", "")
    pages_text = file_doc.get("pages_text") or []
    if not pages_text:
        error_message = f"No pages_text appeared in MongoDB after waiting {max_wait_seconds}s - ingestion likely failed."
        print(f"[PASSMAIN] [ERROR] {error_message} ('{pdf_path}')")
        save_failure_status_to_mongo(file_doc["_id"], error_message)
        return
    print(f"[PASSMAIN] Using {len(pages_text)} page(s) of text already stored in MongoDB for '{pdf_path}' (waited {waited}s)")

    pdf_text = "\n".join(p.get("text", "") for p in pages_text)
    if not pdf_text.strip():
        error_message = "Stored pages_text was empty for this document."
        print(f"[PASSMAIN] [ERROR] {error_message}")
        save_failure_status_to_mongo(file_doc["_id"], error_message)
        return

    chunks = split_text_into_chunks(pdf_text, chunk_size=3000, chunk_overlap=200)
    print(f"[PASSMAIN] Document split into {len(chunks)} text chunks.")

    output = []
    prev_summary = ""

    for i, chunk in enumerate(chunks[:3]):  # Process top chunks
        print(f"[PASSMAIN] Summarizing Chunk {i+1}/{min(3, len(chunks))}...")
        prompt = f"""
        You are an expert academic AI tutor.
        Summarize the following document chunk for a student, highlighting the key topics covered.

        Previous Context: {prev_summary}

        Text Chunk:
        {chunk[:2500]}
        """
        explanation_text = groq_generate(prompt)
        if not explanation_text:
            explanation_text = "Summary preview for this section."

        parsed_output = {"explanation": explanation_text}
        output.append(parsed_output)
        prev_summary = explanation_text[-500:]

    # Real reference links (was a single hardcoded Wikipedia search URL
    # returned for every document, every time - the prompt above used to
    # ask the LLM for "3 reference links" but the code never actually
    # parsed any out of its response, so the placeholder was all anyone
    # ever saw). One search covering the whole document is enough - the
    # frontend already flattens every chunk's links into one combined list,
    # so there was never a reason to search per-chunk.
    fallback_topic = os.path.splitext(os.path.basename(pdf_path))[0].replace('_', ' ').replace('-', ' ').strip()
    topic_prompt = (
        "In 3-6 words, what is the main topic of this text? "
        "Reply with ONLY the topic phrase - no punctuation, no explanation.\n\n"
        f"Text: {output[0]['explanation'][:800]}"
    )
    search_topic = (groq_generate(topic_prompt, max_tokens=20, temperature=0.2) or "").strip().strip('"\'')
    real_links = fetch_reference_links(search_topic or fallback_topic)
    for item in output:
        item["links"] = real_links

    # Save explanation array to MongoDB, against this exact document's _id
    save_explanation_to_mongo(file_doc["_id"], output)
    print("[PASSMAIN] [OK] Document explanation processing completed and saved to MongoDB!\n")

    # Auto-generate summary speech audio & Tavus summary video
    try:
        summary_text = " ".join(item.get("explanation", "") for item in output if isinstance(item, dict))
        if summary_text.strip():
            safe_folder = file_doc.get("folder") or secure_filename(os.path.basename(pdf_path))
            print(f"[PASSMAIN] Synthesizing summary audio for document '{safe_folder}' using GCP TTS...")
            audio_bytes = synthesize_speech(summary_text[:3500])
            timestamp = int(time.time())
            filename = f"summary-{timestamp}.wav"
            s3_upload = upload_audio_to_s3(audio_bytes, filename, folder=safe_folder)
            print(f"[PASSMAIN] Uploaded summary audio to S3: {s3_upload['url']}")

            # Tavus video generation commented out per user request
            print(f"[PASSMAIN] Tavus summary video generation skipped (disabled). Storing summary audio metadata in MongoDB.")
            collection.update_one({"_id": file_doc["_id"]}, {"$push": {"audios": {"s3_url": s3_upload["url"], "filename": filename, "created_at": datetime.utcnow()}}})
            print(f"💾 [PASSMAIN] Saved summary audio metadata to MongoDB for document _id={file_doc['_id']}")
    except Exception as exc:
        print(f"[PASSMAIN] Warning: Auto summary audio/video generation failed: {exc}")

if __name__ == '__main__':
    # Any uncaught exception here used to crash this subprocess silently -
    # the document stayed at status: "processing" forever with nothing in
    # MongoDB to tell the frontend (or a human debugging it) that this run
    # actually died, vs. was still genuinely working.
    try:
        process_file()
    except Exception as exc:
        print(f"[PASSMAIN] [FATAL] Unhandled exception during processing: {exc}")
        import traceback
        traceback.print_exc()
        target_id = sys.argv[1] if len(sys.argv) > 1 else None
        if target_id:
            try:
                collection.update_one(
                    {"_id": ObjectId(target_id)},
                    {"$set": {"status": "failed", "error": str(exc)}}
                )
            except Exception:
                pass


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
