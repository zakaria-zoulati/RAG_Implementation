# ui_streamlit.py
import streamlit as st
import requests

# Page configuration
st.set_page_config(page_title="RAG Demo - IPCC AR6", page_icon="🌍", layout="wide")

# Title
st.title("RAG Demo — IPCC AR6 (Ollama + LangChain)")

st.markdown("---")

# Input field
q = st.text_input(
    "Ask a question about the IPCC reports:",
    placeholder="e.g., What are the main causes of climate change?",
)

# Button and query handling
if st.button("Ask") and q:
    with st.spinner("Searching for answer..."):
        try:
            # Make POST request to FastAPI backend
            resp = requests.post("http://localhost:8000/ask", json={"question": q})

            if resp.ok:
                data = resp.json()

                # Display answer
                st.subheader("Answer")
                st.write(data["answer"])

                # Display sources if available
                if data.get("sources"):
                    st.markdown("---")
                    st.subheader("Sources")
                    for idx, source in enumerate(data["sources"], 1):
                        with st.expander(f"Source {idx}"):
                            st.json(source)
            else:
                st.error(f"API error: {resp.status_code}")
                st.write(resp.text)

        except requests.exceptions.ConnectionError:
            st.error(
                "❌ Could not connect to the backend API. Make sure the FastAPI server is running on http://localhost:8000"
            )
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Instructions in sidebar
with st.sidebar:
    st.header("ℹ️ Instructions")
    st.markdown("""
    1. Make sure your FastAPI backend is running:
       ```
       uvicorn app:app --reload
       ```
    
    2. Enter your question in the text box
    
    3. Click the "Ask" button
    
    4. View the answer and sources
    """)

    st.markdown("---")
    st.caption("Powered by Ollama & LangChain")
