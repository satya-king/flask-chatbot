from rag_indexer import build_index, save_index

documents = [
    "Works is a module at APCFSS under NIDHI project.",
    "Andhra Pradesh Centre for Financial Systems and Services",
    "Develops and maintains web applications",
    "Handles financial systems and services",
    "Collaborates with cross-functional teams",
    "Ensures high performance and responsiveness",
    "Participates in code reviews",
]

index, embeddings = build_index(documents)
save_index(index, documents)