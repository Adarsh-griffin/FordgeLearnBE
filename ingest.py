import os
import sys
import json
import re
from pypdf import PdfReader
from pymongo import MongoClient
from dotenv import load_dotenv, find_dotenv
from groq import Groq

# Load environment variables
load_dotenv(find_dotenv())

# Configure MongoDB connection
MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "neurolearn")

# Configure Groq client for Synthetic TOC generation & reasoning
GROQ_API_KEYS_STR = os.getenv("GROQ_API_KEY", "")
GROQ_KEYS = [k.strip() for k in GROQ_API_KEYS_STR.split(",") if k.strip()]

def get_groq_client():
    if not GROQ_KEYS:
        raise ValueError("GROQ_API_KEY is not configured in .env file.")
    return Groq(api_key=GROQ_KEYS[0])

def generate_synthetic_toc(pages_text):
    """
    Generates a Synthetic Table of Contents (TOC) JSON using Groq API
    for PDFs that lack an explicit table of contents (e.g., research papers, long notes).
    """
    client = get_groq_client()
    
    # Create a truncated sample of page headings and initial content
    page_samples = []
    for p in pages_text[:30]:  # Sample up to first 30 pages for TOC structure
        text_preview = p['text'][:400].replace('\n', ' ')
        page_samples.append(f"Page {p['page_num']}: {text_preview}")
    
    sample_context = "\n".join(page_samples)
    
    prompt = f"""
    Analyze the following page previews from a document and generate a structured Table of Contents (TOC) in valid JSON format.
    
    Format requirements:
    Return ONLY a valid JSON object with the following schema:
    {{
      "is_synthetic": true,
      "document_title": "Estimated Document Title",
      "structure": [
        {{
          "chapter_or_section": "1",
          "title": "Section or Chapter Title",
          "page_range": [start_page, end_page],
          "summary": "1-sentence summary of this section"
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
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"[INGEST] Synthetic TOC generation warning: {e}. Falling back to default sectioning.")
        total_p = len(pages_text)
        chunk_size = max(1, total_p // 5)
        structure = []
        for i in range(0, total_p, chunk_size):
            end_p = min(i + chunk_size, total_p)
            structure.append({
                "chapter_or_section": str(len(structure) + 1),
                "title": f"Part {len(structure) + 1} (Pages {i+1}-{end_p})",
                "page_range": [i + 1, end_p],
                "summary": f"Content covering pages {i+1} to {end_p}"
            })
        return {"is_synthetic": True, "document_title": "Uploaded Document", "structure": structure}

def generate_page_block_index(pages_text):
    """
    Generates a Page-Block Index for short PDFs (< 15 pages) or Question Banks / Worksheets.
    Creates a simple summary per page for fine-grained retrieval.
    """
    structure = []
    for p in pages_text:
        text_snippet = p['text'][:300].replace('\n', ' ')
        structure.append({
            "page_num": p['page_num'],
            "title": f"Page {p['page_num']}",
            "summary": text_snippet[:150] if text_snippet else f"Content on page {p['page_num']}"
        })
    return {
        "is_page_block": True,
        "document_title": "Short Document / Question Bank",
        "pages": structure
    }

def ingest_document(file_path):
    """
    Vectorless RAG Ingestion Pipeline:
    1. Loads PDF pages and extracts full text per page.
    2. Determines indexing strategy (Book TOC vs. Synthetic TOC vs. Page-Block).
    3. Persists document metadata and page_index JSON in MongoDB.
    """
    print(f"[START] Starting Vectorless PageIndex ingestion for: {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Step 1: Read PDF pages
    reader = PdfReader(file_path)
    total_pages = len(reader.pages)
    print(f"[INFO] Read PDF with {total_pages} total pages.")

    pages_text = []
    for idx, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages_text.append({
            "page_num": idx + 1,
            "text": text
        })

    # Step 2: Determine Ingestion Strategy
    base_filename = os.path.basename(file_path)
    collection_name = os.path.splitext(base_filename)[0].lower().replace(" ", "_")
    collection_name = re.sub(r'[^a-z0-9_]', '', collection_name)

    print(f"[INFO] Collection name: '{collection_name}'")

    if total_pages <= 15:
        print("[INFO] Applying Strategy 3: Page-Block Indexing (Short Document / Worksheet / Question Bank)")
        page_index_tree = generate_page_block_index(pages_text)
    else:
        print("[INFO] Applying Strategy 2: Synthetic TOC Generation via Groq API (Long Document / Paper / Textbook)")
        page_index_tree = generate_synthetic_toc(pages_text)

    # Step 3: Persist into MongoDB
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
        "ingested_at": os.getenv("INGEST_TIMESTAMP", "2026-09-27")
    }

    files_col.update_one(
        {"collection_name": collection_name},
        {"$set": doc_payload},
        upsert=True
    )

    print(f"[OK] Vectorless PageIndex tree saved to MongoDB database '{MONGO_DB_NAME}', collection '{collection_name}'.")
    return {
        "status": "success",
        "collection_name": collection_name,
        "total_pages": total_pages,
        "page_index": page_index_tree
    }


# ==============================================================================
# PIPELINE 2: TRADITIONAL VECTOR RAG (COMMENTED OUT FOR FUTURE TOGGLE)
# To switch to Traditional Vector RAG:
# 1. Uncomment the block below
# 2. Rename ingest_document_vector_rag -> ingest_document
# ==============================================================================
# from langchain.vectorstores import Qdrant
# from langchain.embeddings import HuggingFaceBgeEmbeddings
# from langchain.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
#
# def ingest_document_vector_rag(file_path):
#     """Loads PDF, splits into chunks, creates BAAI/bge-base-en-v1.5 PyTorch embeddings & stores in local Qdrant Vector DB."""
#     print(f"Starting traditional vector ingestion for: {file_path}")
#     try:
#         loader = PyPDFLoader(file_path)
#         documents = loader.load()
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#         texts = text_splitter.split_documents(documents)
#         print(f"Loaded and split {len(documents)} document pages into {len(texts)} chunks.")
#
#         model_name = "BAAI/bge-base-en-v1.5"
#         embeddings = HuggingFaceBgeEmbeddings(
#             model_name=model_name,
#             model_kwargs={'device': 'cpu'},
#             encode_kwargs={'normalize_embeddings': False}
#         )
#
#         qdrant_cloud_url = "http://localhost:6333"
#         base_filename = os.path.basename(file_path)
#         collection_name = os.path.splitext(base_filename)[0].lower().replace(" ", "_")
#
#         Qdrant.from_documents(
#             texts,
#             embeddings,
#             url=qdrant_cloud_url,
#             collection_name=collection_name,
#             prefer_grpc=False
#         )
#         print(f"Vector database updated in Qdrant for collection '{collection_name}'.")
#     except Exception as e:
#         print(f"An error occurred during vector ingestion: {e}")
# ==============================================================================

if __name__ == '__main__':
    if len(sys.argv) > 1:
        ingest_document(sys.argv[1])
    else:
        print("Usage: python ingest.py <path_to_your_file.pdf>")