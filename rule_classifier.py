from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from utils.abbreviation_utils import expand_abbreviations, abbreviation_map
from rag_indexer import load_index

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS and docs for RAG comparison
faiss_index, rag_docs = load_index()
rag_doc_embeddings = embedding_model.encode(rag_docs)


db_examples = [
    "what is the status of my loan",
    "is my application approved or pending",
    "who is the checker assigned to this request",
    "loan id 12345 is stuck"
]

llm_examples = [
    "generate an agreement letter",
    "draft an official message",
    "write a sample circular for sanction",
    "compose a reply mail"
]

db_embeddings = embedding_model.encode(db_examples)
llm_embeddings = embedding_model.encode(llm_examples)


def classify_query(question: str) -> str:
    expanded_question = expand_abbreviations(question, abbreviation_map)
    query_embedding = embedding_model.encode([expanded_question])[0]

    # 🔍 Compare with RAG docs
    rag_sim = cosine_similarity([query_embedding], rag_doc_embeddings).max()

    # 🔍 Compare with db and llm
    db_sim = cosine_similarity([query_embedding], db_embeddings).max()
    llm_sim = cosine_similarity([query_embedding], llm_embeddings).max()

    # 🧠 Choose best match
    sims = {
        "rag": rag_sim,
        "db": db_sim,
        "llm": llm_sim
    }
    best_match = max(sims, key=sims.get)

    # ✅ Threshold logic
    if sims[best_match] < 0.55:
        return "llm"  # Fallback to LLM if nothing is similar enough

    return best_match
