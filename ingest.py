import os
import sys
import json
import re
import requests
from pypdf import PdfReader
from pymongo import MongoClient
from dotenv import load_dotenv, find_dotenv
from groq import Groq

# Load environment variables
load_dotenv(find_dotenv())

# Configure MongoDB connection
MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "neurolearn")

# API Keys
GROQ_API_KEYS_STR = os.getenv("GROQ_API_KEY", "")
GROQ_KEYS = [k.strip() for k in GROQ_API_KEYS_STR.split(",") if k.strip()]
PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "").strip()

def get_groq_client():
    if not GROQ_KEYS:
        raise ValueError("[ERROR] GROQ_API_KEY is not configured in .env file.")
    return Groq(api_key=GROQ_KEYS[0])

def check_native_toc(reader):
    """
    Inspects PDF metadata to check if a native Table of Contents (Outline) is present.
    Returns: (toc_entries, True) if TOC found, else (None, False).
    """
    try:
        outline = reader.outline
        if not outline:
            return None, False
        
        toc_entries = []
        def extract_items(items):
            for item in items:
                if isinstance(item, list):
                    extract_items(item)
                else:
                    title = getattr(item, 'title', None)
                    page_num = None
                    try:
                        if hasattr(item, 'page_number') and item.page_number is not None:
                            page_num = item.page_number + 1
                        elif hasattr(reader, 'get_destination_page_number'):
                            page_num = reader.get_destination_page_number(item) + 1
                    except Exception:
                        pass
                    if title:
                        toc_entries.append({
                            "title": title.strip(),
                            "page_num": page_num if page_num else 1
                        })
        
        extract_items(outline)
        if len(toc_entries) > 0:
            return toc_entries, True
        return None, False
    except Exception as e:
        print(f"[INGEST] [DEBUG] Outline inspection notice: {e}")
        return None, False

def parse_with_pageindex_sdk(toc_entries, total_pages, file_path=None):
    """
    PageIndex SDK Engine (Local Fast-Path):
    Constructs a hierarchical page-tree locally from native PDF TOC outlines.
    Zero cloud API calls required.
    """
    structure = []
    for i, entry in enumerate(toc_entries):
        start_p = entry['page_num']
        if i < len(toc_entries) - 1:
            end_p = max(start_p, toc_entries[i+1]['page_num'] - 1)
        else:
            end_p = total_pages
            
        structure.append({
            "section": str(i + 1),
            "title": entry['title'],
            "page_range": [start_p, end_p],
            "summary": f"Native TOC Section covering pages {start_p}-{end_p}"
        })
        
    return {
        "engine": "PageIndex SDK (Local Fast-Path)",
        "has_native_toc": True,
        "doc_id": None,
        "document_title": toc_entries[0]['title'] if toc_entries else "Textbook Document",
        "structure": structure
    }

def parse_with_pageindex_api_or_synthetic(pages_text, file_path):
    """
    PageIndex Cloud API / Synthetic Vision-TOC Engine (Deep-Path):
    Used when NO native TOC is detected.
    Calls PageIndex Cloud API if key available, submitting document so it appears on PageIndex Cloud dashboard.
    """
    if PAGEINDEX_API_KEY:
        try:
            print(f"[INGEST] Submitting document to PageIndex Cloud API with key ending in '...{PAGEINDEX_API_KEY[-6:]}'...")
            from pageindex import PageIndexClient
            pi_client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
            submit_res = pi_client.submit_document(file_path)
            doc_id = submit_res.get("doc_id")
            print(f"[INGEST] [OK] Successfully registered document on PageIndex Cloud dashboard! Doc ID: '{doc_id}'")
            
            # Fetch tree structure if immediately ready
            try:
                tree_res = pi_client.get_tree(doc_id)
                structure = tree_res.get("structure") or tree_res.get("result") or []
                if structure:
                    print(f"[INGEST] [OK] Retrieved cloud-generated PageIndex tree with {len(structure)} nodes!")
                    return {
                        "engine": "PageIndex Cloud API (Vision Engine)",
                        "has_native_toc": False,
                        "doc_id": doc_id,
                        "structure": structure
                    }
            except Exception as tree_err:
                print(f"[INGEST] [NOTICE] Tree generation processing asynchronously on PageIndex Cloud: {tree_err}")
                
            return {
                "engine": "PageIndex Cloud API (Vision Engine)",
                "has_native_toc": False,
                "doc_id": doc_id,
                "structure": []
            }
        except Exception as e:
            print(f"[INGEST] [NOTICE] PageIndex Cloud API submission failed: {e}. Falling back to Groq Synthetic Engine.")

    # Fallback to Groq LLM Synthetic TOC generator
    client = get_groq_client()
    page_samples = []
    for p in pages_text[:30]:
        text_preview = p['text'][:400].replace('\n', ' ')
        page_samples.append(f"Page {p['page_num']}: {text_preview}")
    
    sample_context = "\n".join(page_samples)
    
    prompt = f"""
    Analyze the following page previews from a document without a native Table of Contents.
    Generate a structured subtopic tree in valid JSON format.
    
    Format requirements:
    Return ONLY a valid JSON object with schema:
    {{
      "is_synthetic": true,
      "document_title": "Estimated Document Title",
      "structure": [
        {{
          "section": "1",
          "title": "Section Title",
          "page_range": [start_page, end_page],
          "summary": "1-sentence section summary"
        }}
      ]
    }}
    
    Page Previews:
    {sample_context[:6000]}
    """
    
    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        data = json.loads(response.choices[0].message.content)
        data["engine"] = "PageIndex Vision-TOC Engine (Groq)"
        data["has_native_toc"] = False
        return data
    except Exception as e:
        print(f"[INGEST] Synthetic TOC fallback warning: {e}")
        total_p = len(pages_text)
        chunk_size = max(1, total_p // 5)
        structure = []
        for i in range(0, total_p, chunk_size):
            end_p = min(i + chunk_size, total_p)
            structure.append({
                "section": str(len(structure) + 1),
                "title": f"Part {len(structure) + 1} (Pages {i+1}-{end_p})",
                "page_range": [i + 1, end_p],
                "summary": f"Content covering pages {i+1} to {end_p}"
            })
        return {
            "engine": "PageIndex Fallback Sectioner",
            "has_native_toc": False,
            "document_title": "Uploaded Document",
            "structure": structure
        }

def ingest_document(file_path):
    """
    Smart Combined PageIndex Architecture (SDK + API):
    1. Check PDF for native Table of Contents (TOC).
    2. Route to PageIndex SDK (Local Fast-Path) if TOC present.
    3. Route to PageIndex Cloud API / Synthetic Engine if NO TOC present.
    4. Store output subtopic tree in MongoDB Atlas.
    5. Signal start of summary & downstream processing.
    """
    print(f"\n[INGEST] ==================== [START] Document Ingestion ====================")
    print(f"[INGEST] Processing file: '{file_path}'")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"[INGEST] [ERROR] File not found at path: {file_path}")

    # Step 1: Read PDF pages
    reader = PdfReader(file_path)
    total_pages = len(reader.pages)
    print(f"[INGEST] PDF loaded with {total_pages} total pages.")

    pages_text = []
    for idx, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages_text.append({
            "page_num": idx + 1,
            "text": text
        })

    # Step 2: Check TOC & Route
    native_toc, has_toc = check_native_toc(reader)
    
    if has_toc:
        print("[INGEST] [TOC PRESENT] Document contains native Table of Contents -> Routing via PageIndex SDK (Local Fast-Path)")
        page_index_tree = parse_with_pageindex_sdk(native_toc, total_pages, file_path)
    else:
        print("[INGEST] [NO TOC DETECTED] Document lacks native Table of Contents -> Routing via PageIndex Cloud API / Synthetic Vision-TOC Engine")
        page_index_tree = parse_with_pageindex_api_or_synthetic(pages_text, file_path)

    # Step 3: Store tree in MongoDB
    base_filename = os.path.basename(file_path)
    collection_name = os.path.splitext(base_filename)[0].lower().replace(" ", "_")
    collection_name = re.sub(r'[^a-z0-9_]', '', collection_name)

    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB_NAME]
    files_col = db["files"]

    doc_payload = {
        "collection_name": collection_name,
        "filename": base_filename,
        "file_path": os.path.abspath(file_path),
        "total_pages": total_pages,
        "page_index": page_index_tree,
        "pages_text": pages_text,
        "ingested_at": os.getenv("INGEST_TIMESTAMP", "2026-09-28")
    }

    files_col.update_one(
        {"filePath": file_path},
        {"$set": doc_payload},
        upsert=True
    )

    # Fallback update by originalName
    files_col.update_one(
        {"originalName": base_filename},
        {"$set": doc_payload},
        upsert=True
    )

    print(f"[INGEST] [OK] Subtopic tree stored in MongoDB successfully! (Collection: '{collection_name}')")
    print(f"[INGEST] [START] Document ingestion complete. Initiating summary generation & passmain background worker...\n")

    return {
        "status": "success",
        "collection_name": collection_name,
        "total_pages": total_pages,
        "page_index": page_index_tree
    }

if __name__ == '__main__':
    if len(sys.argv) > 1:
        ingest_document(sys.argv[1])
    else:
        print("Usage: python ingest.py <path_to_your_file.pdf>")