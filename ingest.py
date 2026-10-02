
import os
import sys
import json
import re
import time
import requests
from pypdf import PdfReader
from pymongo import MongoClient
from bson import ObjectId
from dotenv import load_dotenv, find_dotenv
# Shared key-rotation/retry client (was a duplicated single-key copy here -
# see groq_client.py). Behavior preserved: this module still just grabs
# whichever key is currently active, no retry loop, same as before.
from groq_client import get_client as get_groq_client

# Load environment variables
load_dotenv(find_dotenv())

# Configure MongoDB connection
MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "neurolearn")

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "").strip()

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

def _flatten_pageindex_cloud_tree(nodes: list, total_pages: int) -> list:
    """
    PageIndex Cloud's tree is nested - {title, page_index (a single page,
    not a range), nodes: [...children...]} - a completely different shape
    from the flat {section, title, page_range: [start, end], summary} list
    every other engine here produces (and everything downstream - Phase 1's
    knowledge_graph.py, the diagnostic, the planner, the lesson generator -
    assumes). Confirmed by inspecting a real response: previously this was
    returned to callers untouched, which is how a real document ended up
    with a single "section" with page_range=None and all its actual content
    silently discarded in unread child nodes.

    Only LEAF nodes become sections (a node with children is just a
    chapter-level grouping - its content already lives in its children).
    page_range is derived the same way parse_with_pageindex_sdk derives it
    from a flat TOC: each section runs up to just before the next one
    starts, in document order.
    """
    leaves = []

    def collect(node_list):
        for node in node_list:
            children = node.get("nodes") or []
            if children:
                collect(children)
            else:
                leaves.append(node)

    collect(nodes)
    leaves.sort(key=lambda n: n.get("page_index") or 0)

    structure = []
    for i, node in enumerate(leaves):
        start_p = node.get("page_index") or 1
        if i < len(leaves) - 1:
            next_start = leaves[i + 1].get("page_index") or start_p
            end_p = max(start_p, next_start - 1)
        else:
            end_p = max(start_p, total_pages)
        preview = (node.get("text") or "").strip().replace("\n", " ")[:300]
        structure.append({
            "section": str(i + 1),
            "title": node.get("title") or f"Section {i + 1}",
            "page_range": [start_p, end_p],
            "summary": preview or f"Covers page {start_p}.",
        })
    return structure


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
            from pageindex.errors import PageIndexAPIError
            pi_client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
            try:
                submit_res = pi_client.submit_document(file_path)
            except PageIndexAPIError as e:
                if e.status_code == 429:
                    print(f"[QUOTA LIMIT HIT] PageIndex Cloud key '...{PAGEINDEX_API_KEY[-6:]}' hit its rate/usage limit (429). You need a fresh PAGEINDEX_API_KEY - falling back to Groq Synthetic Engine for now.")
                elif e.status_code in (401, 403):
                    print(f"[INVALID KEY] PageIndex Cloud key '...{PAGEINDEX_API_KEY[-6:]}' was rejected ({e.status_code}) - it's likely invalid/expired. Replace PAGEINDEX_API_KEY in .env - falling back to Groq Synthetic Engine for now.")
                raise
            doc_id = submit_res.get("doc_id")
            print(f"[INGEST] [OK] Successfully registered document on PageIndex Cloud dashboard! Doc ID: '{doc_id}'")

            # PageIndex Cloud builds the tree asynchronously. This used to
            # check get_tree() exactly once, immediately after submission -
            # which meant it essentially always returned an empty structure
            # (confirmed: every document that reached this branch ended up
            # stuck with no topic graph, no diagnostic, nothing). Poll for a
            # while instead; this runs inside ingest.py's own background
            # subprocess (see test_groq.py's /api/upload), so waiting here
            # doesn't block the upload response.
            max_wait_seconds = 60
            poll_interval = 5
            waited = 0
            structure = []
            while waited <= max_wait_seconds:
                try:
                    tree_res = pi_client.get_tree(doc_id)
                    structure = tree_res.get("structure") or tree_res.get("result") or []
                    status = str(tree_res.get("status", "")).lower()
                    if structure:
                        break
                    if status == "failed":
                        print(f"[INGEST] [NOTICE] PageIndex Cloud tree generation failed for doc '{doc_id}'.")
                        break
                except PageIndexAPIError as tree_err:
                    if tree_err.status_code == 429:
                        print(f"[QUOTA LIMIT HIT] PageIndex Cloud key '...{PAGEINDEX_API_KEY[-6:]}' hit its rate/usage limit (429) while polling for the tree. You need a fresh PAGEINDEX_API_KEY.")
                    elif tree_err.status_code in (401, 403):
                        print(f"[INVALID KEY] PageIndex Cloud key '...{PAGEINDEX_API_KEY[-6:]}' was rejected ({tree_err.status_code}) while polling. Replace PAGEINDEX_API_KEY in .env.")
                    else:
                        print(f"[INGEST] [NOTICE] Tree poll error (will retry): {tree_err}")
                except Exception as tree_err:
                    print(f"[INGEST] [NOTICE] Tree poll error (will retry): {tree_err}")
                time.sleep(poll_interval)
                waited += poll_interval

            if structure:
                flat_structure = _flatten_pageindex_cloud_tree(structure, total_pages=len(pages_text))
                print(f"[INGEST] [OK] Retrieved cloud-generated PageIndex tree with {len(flat_structure)} sections (from {len(structure)} top-level nodes) after {waited}s!")
                return {
                    "engine": "PageIndex Cloud API (Vision Engine)",
                    "has_native_toc": False,
                    "doc_id": doc_id,
                    "structure": flat_structure
                }

            print(f"[INGEST] [NOTICE] PageIndex Cloud tree not ready after {max_wait_seconds}s - falling back to Groq Synthetic Engine so this document isn't left with no structure at all.")
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

def _extract_topic_name(base_filename: str, pages_text: list) -> str:
    """
    The frontend's "type a topic" flow (Study.tsx's createMinimalPdfBlob)
    embeds "Topic: <name>" as literal text in its synthetic one-page PDF -
    pull the exact original name/casing from there rather than reversing
    the lossy filename slug (topicName.toLowerCase()...replace(/[^a-z0-9]/, '_')).
    """
    if pages_text:
        first_page_text = pages_text[0].get("text", "")
        match = re.search(r'Topic:\s*(.+)', first_page_text)
        if match:
            return match.group(1).strip().split('\n')[0]
    # Fallback if the embedded text couldn't be parsed (still recoverable,
    # just lossier): derive a readable name from the filename slug.
    stem = base_filename[:-len('_topic.pdf')] if base_filename.endswith('_topic.pdf') else base_filename
    return stem.replace('_', ' ').strip().title() or "Untitled Topic"

def generate_topic_curriculum(topic_name: str, num_sections: int = 6) -> dict:
    """
    Topic-only mode (the "type a topic name" flow, not a real PDF upload):
    there is no source document, so routing this through PageIndex SDK/API
    used to submit a meaningless one-line placeholder PDF and get back an
    empty tree (has_native_toc is always False here, has_toc is always
    False, and the "content" is just "Topic: X" - PageIndex Cloud has
    nothing real to vision-parse). Instead the LLM builds the curriculum
    directly from its own knowledge of the subject. PageIndex SDK/API stay
    exactly as before for real PDF uploads (see ingest_document).
    """
    client = get_groq_client()
    prompt = f"""You are an expert curriculum designer. A student wants to learn about:
"{topic_name}"

Design a structured learning curriculum for this topic, broken into
{num_sections} logically ordered sections a student should study in
sequence - earlier sections should be prerequisites for later ones
(foundational concepts first, advanced/applied concepts last).

Return ONLY a valid JSON object with this exact schema:
{{
  "document_title": "A clear title for this learning module",
  "structure": [
    {{
      "section": "1",
      "title": "Section title",
      "page_range": [1, 1],
      "summary": "1-2 sentence summary of what this section covers"
    }}
  ]
}}

Number page_range sequentially starting at 1, one number per section
(these are module numbers for internal ordering - there is no source PDF,
so they are not real page numbers).
"""
    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        data = json.loads(response.choices[0].message.content)
        if not isinstance(data.get("structure"), list) or not data["structure"]:
            raise ValueError("Groq returned an empty/invalid structure")
        data["engine"] = "LLM Topic Curriculum Generator (no source document)"
        data["has_native_toc"] = False
        data["is_topic_mode"] = True
        data["doc_id"] = None
        return data
    except Exception as e:
        print(f"[INGEST] [ERROR] Topic curriculum generation failed: {e}")
        # Minimal single-section fallback so ingestion doesn't hard-fail.
        return {
            "engine": "LLM Topic Curriculum Generator (fallback)",
            "has_native_toc": False,
            "is_topic_mode": True,
            "doc_id": None,
            "document_title": topic_name,
            "structure": [{
                "section": "1",
                "title": topic_name,
                "page_range": [1, 1],
                "summary": f"Overview of {topic_name}."
            }]
        }

def ingest_document(file_path, file_id=None):
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

    # Step 2: Route
    base_filename = os.path.basename(file_path)
    # The frontend's "type a topic name" flow (Study.tsx's createMinimalPdfBlob)
    # uploads a synthetic one-page PDF named "<slug>_topic.pdf" instead of a
    # real document. There's no real content for PageIndex SDK/API to parse
    # in that case, so it's routed separately - PageIndex SDK/API below are
    # otherwise completely unchanged for real PDF uploads.
    is_topic_mode = base_filename.endswith('_topic.pdf')

    if is_topic_mode:
        topic_name = _extract_topic_name(base_filename, pages_text)
        print(f"[INGEST] [TOPIC MODE] No source document - user typed a topic ('{topic_name}') -> Routing via LLM Topic Curriculum Generator (PageIndex skipped)")
        page_index_tree = generate_topic_curriculum(topic_name)
    else:
        native_toc, has_toc = check_native_toc(reader)
        if has_toc:
            print("[INGEST] [TOC PRESENT] Document contains native Table of Contents -> Routing via PageIndex SDK (Local Fast-Path)")
            page_index_tree = parse_with_pageindex_sdk(native_toc, total_pages, file_path)
        else:
            print("[INGEST] [NO TOC DETECTED] Document lacks native Table of Contents -> Routing via PageIndex Cloud API / Synthetic Vision-TOC Engine")
            page_index_tree = parse_with_pageindex_api_or_synthetic(pages_text, file_path)

    # Step 3: Store tree in MongoDB
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

    # Update by the exact document _id when known (passed from test_groq.py's
    # /api/upload, which already created this document) - matching by _id is
    # unambiguous, unlike the string-matching this replaced. The old code
    # matched by filePath (the SANITIZED local disk path - fine) and then,
    # as a "fallback", matched by originalName against base_filename, which
    # is ALSO the sanitized name (derived from the sanitized file_path, via
    # secure_filename() in /api/upload). For any filename secure_filename()
    # changes (most commonly: spaces -> underscores), that fallback's query
    # never matched the real document (whose originalName keeps the
    # UNSANITIZED name), so upsert=True silently created a second, ghost
    # document with the sanitized originalName instead of updating the real
    # one. passmain_groq.py later writes the `explanation` field onto the
    # REAL document (it updates by _id), so the ghost document - the one the
    # frontend's polling actually finds, since it polls by this same
    # sanitized name - never gets an explanation and "Processing..." never
    # resolves. (See /api/upload and /api/processing-status in test_groq.py.)
    if file_id:
        result = files_col.update_one(
            {"_id": ObjectId(file_id)},
            {"$set": doc_payload},
        )
        if result.matched_count == 0:
            print(f"[INGEST] [WARNING] No document matched _id={file_id} - falling back to filePath match.")
            files_col.update_one({"filePath": file_path}, {"$set": doc_payload}, upsert=True)
    else:
        files_col.update_one({"filePath": file_path}, {"$set": doc_payload}, upsert=True)

    print(f"[INGEST] [OK] Subtopic tree stored in MongoDB successfully! (Collection: '{collection_name}')")
    print(f"[INGEST] [START] Document ingestion complete. Initiating summary generation & passmain background worker...\n")

    # The PDF's full text (pages_text) and structure (page_index) are now
    # durable in MongoDB - nothing downstream reads this local disk copy
    # again (passmain_groq.py reads pages_text from Mongo, image_finder.py
    # only uses file_path as a string key). Deleting it here is the only
    # place that's safe to do so: this function is the sole reader of the
    # actual file bytes, and we've just finished using them. Per the "nothing
    # stored locally, only AWS/MongoDB" requirement - uploads/ would
    # otherwise grow forever.
    try:
        os.remove(file_path)
        print(f"[INGEST] Removed local upload copy: '{file_path}'")
    except OSError as e:
        print(f"[INGEST] [WARNING] Could not remove local upload copy '{file_path}': {e}")

    return {
        "status": "success",
        "collection_name": collection_name,
        "total_pages": total_pages,
        "page_index": page_index_tree
    }

if __name__ == '__main__':
    if len(sys.argv) > 1:
        file_id_arg = sys.argv[2] if len(sys.argv) > 2 else None
        ingest_document(sys.argv[1], file_id_arg)
    else:
        print("Usage: python ingest.py <path_to_your_file.pdf> [file_id]")