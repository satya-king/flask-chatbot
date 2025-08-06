def get_top_k_documents(query, embedding_model, faiss_index, rag_docs, k=3):
    query_embedding = embedding_model.encode([query])
    distances, indices = faiss_index.search(query_embedding, k)

    results = []
    for i in indices[0]:
        if 0 <= i < len(rag_docs):
            results.append(rag_docs[i])
    return results