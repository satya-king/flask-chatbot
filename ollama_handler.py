import os
import requests
from dotenv import load_dotenv
from rag_indexer import load_index
from sentence_transformers import SentenceTransformer

from rag_search import get_top_k_documents
from rule_classifier import classify_query
from utils.abbreviation_utils import expand_abbreviations, abbreviation_map

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "satyabot")

# Load embedding model and vector index once
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
faiss_index, rag_docs = load_index()


def generate_response(messages):
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model": MODEL_NAME,
            "messages": messages,
            "stream": False
        }
    )

    if response.status_code == 200:
        return response.json()["message"]["content"].strip()
    else:
        return "⚠️ Failed to generate response."


def generate_chat_response(user_query):
    category = classify_query(user_query)

    if category == "rag":
        docs = get_top_k_documents(user_query, embedding_model, faiss_index, rag_docs)
        context = "\n".join(docs)

        messages = [
            {"role": "system", "content": f"You are a helpful assistant. Use this context to answer:\n{context}"},
            {"role": "user", "content": user_query}
        ]
        reply = generate_response(messages)
        return {
            "reply": reply,
            "source": "📁 RAG Documents"
        }

    elif category == "db":
        reply = "🔧 Live database query not implemented yet."
        return {
            "reply": reply,
            "source": "DB"
        }

    else:
        messages = [
            {"role": "user", "content": user_query}
        ]
        reply = generate_response(messages)
        return {
            "reply": reply,
            "source": "LLM"
        }
