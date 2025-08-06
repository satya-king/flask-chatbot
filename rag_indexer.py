import os
import faiss
from sentence_transformers import SentenceTransformer
import pickle

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
DATA_DIR = "data"
INDEX_PATH = "faiss_index/index.faiss"
DOCS_PATH = "faiss_index/docs.pkl"


# ✅ Step 1: Recursively load all text files
def load_all_documents():
    documents = []
    for root, _, files in os.walk(DATA_DIR):
        for file in files:
            if file.endswith(".txt"):
                path = os.path.join(root, file)
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        documents.append(content)
    return documents


# ✅ Step 2: Build FAISS index
def build_index():
    documents = load_all_documents()
    if not documents:
        raise ValueError("❌ No documents found in data/")

    embeddings = embedding_model.encode(documents, show_progress_bar=True)

    dimension = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)

    faiss.write_index(index, INDEX_PATH)
    with open(DOCS_PATH, "wb") as f:
        pickle.dump(documents, f)

    print("✅ FAISS index built successfully.")


# ✅ Step 3: Load index and docs
def load_index():
    if not os.path.exists(INDEX_PATH) or not os.path.exists(DOCS_PATH):
        build_index()

    index = faiss.read_index(INDEX_PATH)
    with open(DOCS_PATH, "rb") as f:
        documents = pickle.load(f)

    return index, documents
