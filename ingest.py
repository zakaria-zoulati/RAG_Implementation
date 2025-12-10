from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import glob 
import json

def load_and_split(pdf_path, chunk_size=1000, chunk_overlap=200):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_docs = splitter.split_documents(docs)
    return split_docs

def main():
    os.makedirs("chunks", exist_ok=True)
    for p in glob.glob("data/*.pdf"):
        print("Processing", p)
        docs = load_and_split(p)
        out = [{"page_content": d.page_content, "metadata": d.metadata} for d in docs]
        fn = os.path.join("chunks", os.path.basename(p) + ".json")
        with open(fn, "w", encoding="utf8") as f:
            json.dump(out, f)
        print("Saved", fn)

main()