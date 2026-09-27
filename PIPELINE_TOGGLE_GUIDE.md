# Pipeline Toggle & Startup Guide: Vectorless RAG vs. Traditional Vector RAG

## 🚀 Quick Startup Commands (For Anyone Cloning This Repo)

```bash
# 1. Navigate to LearnBack
cd LearnBack

# 2. Create & Activate Virtual Environment
python -m venv .venv
# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# On macOS / Linux / Git Bash:
source .venv/bin/activate

# 3. Install Dependencies
pip install -r requirement.txt

# 4. Configure .env file (add GROQ_API_KEY, TAVUS_API_KEY, MONGODB_URI)

# 5. Start Backend Server
python test_groq.py
```

---

## 🟢 Default Active Mode: Pipeline 1 (Vectorless RAG / PageIndex)

* **Memory Usage:** < 80 MB RAM (Render 500MB Free Tier compatible).
* **Dependencies:** `pageindex`, `groq`, `pypdf`, `pymongo`.
* **Database:** MongoDB Atlas / Local MongoDB (No vector DB required).

---

## 🔄 How to Switch to Pipeline 2 (Traditional Vector RAG)

If you decide to deploy to a server with GPU/higher RAM (e.g. AWS EC2, GCP, or a 2GB+ container) and want to use local Qdrant + PyTorch embeddings:

### Step 1: Update `requirement.txt`
In `LearnBack/requirement.txt`:
1. Comment out `pageindex`.
2. Uncomment the traditional vector libraries:
   ```txt
   torch
   transformers
   sentence-transformers
   langchain==0.2.16
   langchain-community==0.2.16
   langchain-core==0.2.38
   langchain-huggingface
   langchain-qdrant
   qdrant-client
   ```
3. Run `pip install -r requirement.txt`.

### Step 2: Update `ingest.py`
In `LearnBack/ingest.py`:
1. Comment out the active `ingest_document(file_path)` function.
2. At the bottom of `ingest.py`, uncomment the `ingest_document_vector_rag(file_path)` block and rename it to `ingest_document(file_path)`.

### Step 3: Update `retrieval.py`
In `LearnBack/retrieval.py`:
1. Comment out `retrieve_pageindex_context(...)`.
2. At the bottom of `retrieval.py`, uncomment `get_relevant_docs_vector_rag(...)` and `embeddings_vector_rag`.

### Step 4: Update `test_groq.py`
In `LearnBack/test_groq.py`:
1. Uncomment the LangChain imports (`RetrievalQA`, `PromptTemplate`) and the `GroqLLM` class.
2. In `/api/qa`, switch from `retrieve_pageindex_context` to `RetrievalQA.from_chain_type`.

### Step 5: Start Qdrant Docker
Run Qdrant in Docker:
```bash
docker run -p 6333:6333 -v .:/qdrant/storage qdrant/qdrant
```

---

## 🔄 How to Switch Back to Pipeline 1 (Vectorless RAG)

1. Ensure `pageindex` is uncommented in `requirement.txt`.
2. Ensure active functions in `ingest.py` and `retrieval.py` are using `pageindex` and MongoDB.
3. Keep PyTorch & Qdrant blocks commented out.
