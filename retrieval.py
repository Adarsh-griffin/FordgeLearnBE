import os
import json
from pymongo import MongoClient
from dotenv import load_dotenv, find_dotenv
from groq import Groq

# Load environment variables
load_dotenv(find_dotenv())

# MongoDB configuration
MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "neurolearn")

# Groq client for reasoning-based retrieval
GROQ_API_KEYS_STR = os.getenv("GROQ_API_KEY", "")
GROQ_KEYS = [k.strip() for k in GROQ_API_KEYS_STR.split(",") if k.strip()]

def get_groq_client():
    if not GROQ_KEYS:
        raise ValueError("GROQ_API_KEY is not configured in .env file.")
    return Groq(api_key=GROQ_KEYS[0])

def get_pageindex_document(collection_name):
    """Fetches the document payload (page_index tree and page texts) from MongoDB."""
    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB_NAME]
    files_col = db["files"]
    
    # Try exact match or regex case-insensitive match
    doc = files_col.find_one({"collection_name": collection_name})
    if not doc:
        doc = files_col.find_one({"collection_name": {"$regex": f"^{collection_name}$", "$options": "i"}})
    
    return doc

def retrieve_pageindex_context(query, collection_name, max_pages=3):
    """
    Vectorless RAG Tree Reasoning Retrieval:
    1. Fetches the page_index tree for collection_name from MongoDB.
    2. Asks Groq LLM to reason over the index tree and pick the best target page numbers.
    3. Extracts and returns text from target pages with exact page citations.
    """
    doc = get_pageindex_document(collection_name)
    if not doc:
        print(f"⚠️ Document collection '{collection_name}' not found in MongoDB.")
        return {
            "context": f"No document found for collection '{collection_name}'.",
            "page_numbers": [],
            "citations": []
        }
    
    page_index = doc.get("page_index", {})
    pages_text = doc.get("pages_text", [])
    
    # Convert index tree to compact string for Groq reasoning
    tree_summary = json.dumps(page_index, indent=2)[:4000]
    
    groq_client = get_groq_client()
    prompt = f"""
    You are an expert academic document navigator.
    Analyze the student's question and the document index tree below.
    Determine which specific page numbers (up to {max_pages} pages) are most likely to contain the answer.

    Question: "{query}"

    Document Index Tree:
    {tree_summary}

    Return ONLY a JSON object with this exact schema:
    {{
      "target_pages": [page_num_1, page_num_2],
      "reasoning": "Short explanation of why these pages were selected",
      "section_title": "Title of selected section/chapter"
    }}
    """
    
    target_pages = []
    section_title = "Document Context"
    for model_name in ["openai/gpt-oss-20b"]:
        try:
            response = groq_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            result = json.loads(response.choices[0].message.content)
            target_pages = result.get("target_pages", [])
            section_title = result.get("section_title", "Selected Section")
            break
        except Exception as ex:
            print(f"[RETRIEVAL] Reasoning retrieval model error on {model_name}: {ex}")
            target_pages = [1, 2] if len(pages_text) >= 2 else [1]
            break

    # Extract text for selected pages
    extracted_contexts = []
    citations = []
    
    for p in pages_text:
        if p["page_num"] in target_pages:
            citation_str = f"[Source: Page {p['page_num']} | {section_title}]"
            extracted_contexts.append(f"{citation_str}\n{p['text']}")
            citations.append(f"Page {p['page_num']}")

    # If no pages matched, default to first page
    if not extracted_contexts and pages_text:
        first_page = pages_text[0]
        extracted_contexts.append(f"[Source: Page 1 | Document Overview]\n{first_page['text']}")
        citations.append("Page 1")

    full_context = "\n\n---\n\n".join(extracted_contexts)
    
    return {
        "context": full_context,
        "page_numbers": target_pages,
        "section_title": section_title,
        "citations": citations
    }

def get_relevant_docs(query, collection_name):
    """
    Compatibility wrapper matching previous function signature.
    Returns structured context from PageIndex vectorless retrieval.
    """
    retrieved = retrieve_pageindex_context(query, collection_name)
    return retrieved["context"]

print("[OK] PageIndex Vectorless Retriever initialized successfully!")


# ==============================================================================
# PIPELINE 2: TRADITIONAL VECTOR RAG (COMMENTED OUT FOR FUTURE TOGGLE)
# To switch to Traditional Vector RAG:
# 1. Uncomment the block below
# 2. Use get_relevant_docs_vector_rag in your Flask endpoints
# ==============================================================================
# from langchain_qdrant.vectorstores import Qdrant
# from langchain_community.embeddings import HuggingFaceBgeEmbeddings
# from qdrant_client import QdrantClient
#
# local_model_path = "C:/techai/program/model/bge-base-en-v1.5"
# if not os.path.exists(local_model_path):
#     model_name = "BAAI/bge-base-en-v1.5"
# else:
#     model_name = local_model_path
#
# embeddings_vector_rag = HuggingFaceBgeEmbeddings(
#     model_name=model_name,
#     model_kwargs={'device': 'cpu'},
#     encode_kwargs={'normalize_embeddings': False}
# )
#
# Qdrant_url = "http://localhost:6333"
# client_qdrant = QdrantClient(url=Qdrant_url, prefer_grpc=False)
#
# def get_relevant_docs_vector_rag(query, collection_name):
#     """Returns LangChain Qdrant retriever for traditional vector similarity search."""
#     db = Qdrant(client=client_qdrant, embeddings=embeddings_vector_rag, collection_name=collection_name)
#     return db.as_retriever(search_type="similarity", search_kwargs={"k": 2})
# ==============================================================================