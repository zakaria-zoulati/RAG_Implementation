from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import json 
import os


def embed_and_store(chunks_dir="chunks", persist_directory="vectordb"):
    embedder = OllamaEmbeddings(model="nomic-embed-text:latest")
    documents = []
    for fn in os.listdir(chunks_dir):
        if not fn.endswith(".json"):
            continue
        with open(os.path.join(chunks_dir, fn), "r", encoding="utf8") as f:
            items = json.load(f)
        for it in items:
            documents.append(
                Document(
                    page_content=it["page_content"], metadata=it.get("metadata", {})
                )
            )
    vectordb = Chroma.from_documents(
        documents, embedding=embedder, persist_directory=persist_directory
    )
    return vectordb


if __name__ == "__main__":
    embed_and_store()
    print("Embeddings created and stored in vectordb/")