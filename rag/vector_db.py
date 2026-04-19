import os
import json
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_PATH = os.path.join(BASE_DIR, "rag", "documents.json")
DB_PATH = os.path.join(BASE_DIR, "rag", "faiss_index")

def setup_vector_db():
    if not os.path.exists(DOCS_PATH):
        return None
    with open(DOCS_PATH, 'r') as f:
        data = json.load(f)
    
    docs = [Document(page_content=item["content"], metadata=item["metadata"]) for item in data]
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vecdb = FAISS.from_documents(docs, embeddings)
    vecdb.save_local(DB_PATH)

def get_retriever():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    if not os.path.exists(DB_PATH):
        setup_vector_db()
    if os.path.exists(DB_PATH):
        vecdb = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
        return vecdb.as_retriever(search_kwargs={"k": 2})
    return None
