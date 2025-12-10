# app.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# LangChain Imports
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.exceptions import OutputParserException


from fastapi.middleware.cors import CORSMiddleware


# Standard Library
import os
import sys


# --- Configuration ---
# Your database path and model names
VECTOR_DB_PATH = "vectordb"
EMBEDDING_MODEL = "nomic-embed-text:latest"
# Recommended lightweight generative model
LLM_MODEL = "phi3:mini"

# --- FastAPI Initialization ---
app = FastAPI(
    title="Local RAG API with Ollama and ChromaDB",
    description=f"RAG endpoint using {LLM_MODEL} for generation and {EMBEDDING_MODEL} for embeddings.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your Streamlit URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for request/response
class QueryIn(BaseModel):
    """Input query structure."""

    question: str


class RAGResponse(BaseModel):
    """Output response structure including answer and sources."""

    answer: str
    sources: list[dict]  # List of document metadata dictionaries


# Global variable for the RAG chain
rag_chain_from_source = None


# --- LangChain Setup: This runs once when the FastAPI application starts ---


# Function to format the retrieved documents into a single string for the prompt context
def format_docs(docs):
    """Concatenates document page content into a single string."""
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


try:
    print(f"Initializing LangChain components...")

    # 1. Embeddings
    embedding_fn = OllamaEmbeddings(model=EMBEDDING_MODEL)

    # 2. Vector DB (Chroma)
    if not os.path.exists(VECTOR_DB_PATH):
        # This branch handles the scenario where the vector DB hasn't been created yet.
        print(
            f"WARNING: Vector DB directory '{VECTOR_DB_PATH}' not found. Initializing empty DB for now."
        )
        vectordb = Chroma.from_texts(
            texts=["The vector store is currently empty."],
            embedding=embedding_fn,
            persist_directory=VECTOR_DB_PATH,
        )
    else:
        # Load the existing vector store
        vectordb = Chroma(
            persist_directory=VECTOR_DB_PATH, embedding_function=embedding_fn
        )

    # Configure the retriever (k=4 is typical)
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    # 3. Generative LLM
    llm = ChatOllama(model=LLM_MODEL, temperature=0.0)

    # 4. RAG prompt template
    template = """Use the following context to answer the question. If the answer is not in the context, say 'I don't know.'

Context: {context}

Question: {question}

Answer:"""
    prompt = ChatPromptTemplate.from_template(template)

    # 5. The Core RAG Chain Structure
    context_chain = retriever | format_docs

    rag_chain_from_source = RunnableParallel(
        # 'context': Retrieves and formats documents
        context=context_chain,
        # 'question': Passes the original input question through
        question=RunnablePassthrough(),
        # 'source_documents': Retrieves the raw documents for source metadata
        source_documents=retriever,
    ) | {
        # 'answer': The generation chain (prompt -> LLM -> parser)
        "answer": prompt | llm | StrOutputParser(),
        # 'sources': Extracts the metadata from the raw documents
        "sources": lambda x: [doc.metadata for doc in x["source_documents"]],
    }
    print("LangChain components initialized successfully. App is ready.")

except Exception as e:
    # A failure here means the app cannot connect to Ollama or load the DB.
    print(
        f"\n--- FATAL STARTUP ERROR ---\nCould not initialize LangChain components: {e}",
        file=sys.stderr,
    )
    print(
        f"**Action required:** Ensure Ollama is running and models ({EMBEDDING_MODEL} and {LLM_MODEL}) are pulled.",
        file=sys.stderr,
    )
    # The global chain remains None, which will trigger the 503 error in the endpoint.


# --- FastAPI Endpoint ---


@app.post("/ask", response_model=RAGResponse)
async def ask(q: QueryIn):
    """
    Handles a user query by invoking the RAG chain asynchronously.
    """
    global rag_chain_from_source

    # Check if initialization failed at startup
    if rag_chain_from_source is None:
        raise HTTPException(
            status_code=503,
            detail="Service Not Ready: LangChain/Ollama failed to initialize at startup. Check server logs.",
        )

    try:
        # Use .ainvoke() for non-blocking asynchronous execution
        result = await rag_chain_from_source.ainvoke(q.question)

        return result

    except OutputParserException as e:
        # This handles cases where the LLM might output something unparseable
        raise HTTPException(
            status_code=500,
            detail=f"LLM Output Error: The model output could not be parsed. This sometimes happens with context-heavy questions. Details: {e}",
        )

    except Exception as e:
        # Catch all other runtime errors (most likely Ollama connection failure)
        error_msg = str(e)

        detail = (
            f"Ollama/RAG Execution Error: Failed to process query. Ensure Ollama is still running "
            f"and the LLM model ({LLM_MODEL}) is available. Details: {error_msg.splitlines()[0]}"
        )

        raise HTTPException(status_code=500, detail=detail)
