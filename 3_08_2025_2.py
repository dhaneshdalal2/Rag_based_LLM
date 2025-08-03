!pip install streamlit pandas chromadb sentence-transformers requests numpy

import streamlit as st
import pandas as pd
import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
import requests
import os
from sentence_transformers import CrossEncoder
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import time
from requests.exceptions import Timeout


# --- Configuration ---
OLLAMA_EMBED_MODEL = "nomic-embed-text"
OLLAMA_LLM_MODEL = "llama2:latest"
OLLAMA_API_URL = "http://localhost:11434/api"
CHROMA_COLLECTION_NAME = "bns_documents"
BATCH_SIZE = 100
RERANK_LIMIT = 10  # Number of candidates to rerank per query

# HNSW params (speed tuned)
hnsw_params = {
    "hnsw:space": "cosine",
    "hnsw:construction_ef": 90,
    "hnsw:M": 16,
    "hnsw:search_ef": 50
}


# ---- Streamlit Page Setup ----
st.set_page_config(
    page_title="🔍 Legal Search with Explanations",
    page_icon="⚖️",
    layout="wide"
)


st.markdown("""
    <style>
        .main {background-color: #f9fafb;}
        div.stButton>button {width: 100%;}
        .result-header {
            background: #4a90e2;
            color: white;
            padding: 8px 16px;
            border-radius: 8px;
        }
        .small-label {color: #505050; font-size:0.95em;}
        .result-container {
            background: #fff;
            padding: 18px 24px;
            border-radius: 10px;
            margin-bottom: 16px;
            box-shadow: 0 0 4px #e8e8e8;
        }
    </style>
""", unsafe_allow_html=True)


# --- Sidebar Info ---
with st.sidebar:
    if os.path.exists("logo.jpg"):
        st.image("logo.jpg", width=64)
    st.header("Legal Search")
    st.write("""
    Search legal documents using:
    - Ollama Embeddings
    - ChromaDB (ANN)
    - Cross-Encoder Reranking
    - LLama2 LLM Explanations
    """)
    st.markdown("---")
    st.write("Developed with ❤️")


# Cache reranker loading once
@st.cache_resource(show_spinner=False)
def load_reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)


# Cache chroma collection loading/creation once, with progress UI during creation
@st.cache_resource(show_spinner=False)
def load_chroma_collection():
    """Load or create ChromaDB collection with Ollama embeddings."""
    try:
        ollama_ef = OllamaEmbeddingFunction(
            model_name=OLLAMA_EMBED_MODEL,
            url=f"{OLLAMA_API_URL}/embeddings"
        )
    except Exception as e:
        st.error(f"Error initializing Ollama embedding function: {e}")
        return None

    client = chromadb.Client()

    # Try to get existing collection first
    try:
        collection = client.get_collection(CHROMA_COLLECTION_NAME)
        return collection
    except Exception:
        # Collection does not exist, proceed to create
        pass

    # Load CSV data
    try:
        df = pd.read_csv("bns.csv")
        df['description'] = df['description'].fillna('')
    except FileNotFoundError:
        st.error("Local file 'bns.csv' not found in current directory.")
        return None

    # If collection exists, delete prior to recreate
    try:
        client.delete_collection(CHROMA_COLLECTION_NAME)
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        embedding_function=ollama_ef,
        metadata=hnsw_params
    )

    total_docs = len(df)
    total_batches = (total_docs + BATCH_SIZE - 1) // BATCH_SIZE

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i in range(0, total_docs, BATCH_SIZE):
        batch_number = (i // BATCH_SIZE) + 1
        batch_df = df.iloc[i:i+BATCH_SIZE]
        batch_ids = [f"doc_{idx}" for idx in batch_df.index]
        batch_docs = batch_df['description'].tolist()
        batch_metas = batch_df[['act', 'section']].to_dict('records')

        status_text.text(f"Adding batch {batch_number} of {total_batches}...")

        try:
            collection.add(
                documents=batch_docs,
                metadatas=batch_metas,
                ids=batch_ids
            )
        except Exception as e:
            st.error(f"Error adding batch {batch_number}: {e}")
            status_text.empty()
            progress_bar.empty()
            return None

        progress = batch_number / total_batches
        progress_bar.progress(progress)

    status_text.text("Completed embedding and indexing.")
    progress_bar.empty()

    return collection


def rerank_results(query, docs, metas, distances, reranker_model):
    rerank_n = min(len(docs), RERANK_LIMIT)
    pairs = [(query, doc) for doc in docs[:rerank_n]]
    scores = reranker_model.predict(pairs)
    ranked_indices = np.argsort(scores)[::-1]
    reranked_docs = [docs[i] for i in ranked_indices]
    reranked_metas = [metas[i] for i in ranked_indices]
    reranked_distances = [distances[i] for i in ranked_indices]
    reranked_scores = [scores[i] for i in ranked_indices]
    return reranked_docs, reranked_metas, reranked_distances, reranked_scores


def generate_relevance_explanation_llm(query_text: str, doc_text: str) -> str:
    prompt = (
        f"User query: \"{query_text}\"\n"
        f"Document excerpt: \"{doc_text}\"\n"
        "Explain concisely why this document excerpt is relevant to the user query."
    )
    try:
        try:
            response = requests.post(
                f"{OLLAMA_API_URL}/generate",
                json={
                    "model": OLLAMA_LLM_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 100
                    }
                },
                timeout=120  # Increased timeout to 120 seconds
            )
            response.raise_for_status()
        except Timeout:
            # Retry once after wait
            time.sleep(5)
            response = requests.post(
                f"{OLLAMA_API_URL}/generate",
                json={
                    "model": OLLAMA_LLM_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 100
                    }
                },
                timeout=120
            )
            response.raise_for_status()

        output = response.json()
        explanation = output.get("response", "").strip()
        return explanation if explanation else "No explanation generated."
    except Exception as e:
        return f"Error generating explanation: {str(e)}"


def parallel_generate_explanations(query_text, docs):
    # Limit to 2 parallel requests to avoid overloading Ollama
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(generate_relevance_explanation_llm, query_text, doc)
            for doc in docs
        ]
        explanations = [f.result() for f in futures]
    return explanations


def query_and_search_with_explanations(collection, query_text: str, k: int, reranker_model):
    try:
        results = collection.query(query_texts=[query_text], n_results=max(k, RERANK_LIMIT))
    except Exception as e:
        st.error(f"Error querying collection: {e}")
        return None

    if not results or not results.get("ids") or not results["ids"][0]:
        return None

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    rerank_n = min(len(docs), RERANK_LIMIT, k)
    if reranker_model:
        docs_rerank, metas_rerank, distances_rerank, scores_rerank = rerank_results(
            query_text, docs, metas, distances, reranker_model
        )
    else:
        docs_rerank = docs[:rerank_n]
        metas_rerank = metas[:rerank_n]
        distances_rerank = distances[:rerank_n]
        scores_rerank = [None] * rerank_n

    docs_final = docs_rerank[:k]
    metas_final = metas_rerank[:k]
    distances_final = distances_rerank[:k]
    rerank_scores_final = scores_rerank[:k]

    explanations = parallel_generate_explanations(query_text, docs_final)

    detailed_results = []
    for i in range(len(docs_final)):
        meta = metas_final[i]
        detailed_results.append({
            "act": meta.get("act", ""),
            "section": meta.get("section", ""),
            "description": docs_final[i],
            "distance": distances_final[i],
            "rerank_score": rerank_scores_final[i],
            "explanation": explanations[i]
        })
    return detailed_results


# ==== Main UI ====
st.markdown("<div class='result-header'><h2>🔍 Legal Search with Explanations</h2></div>", unsafe_allow_html=True)
st.write("Efficiently locate relevant legal documents and understand why they match your query.")

c1, c2 = st.columns([2, 1])
with c1:
    query_text = st.text_area(
        "Enter your legal search query:",
        height=120,
        placeholder="e.g., a person has committed suicide after murdering a person"
    )
with c2:
    top_k = st.slider(
        "Number of top results",
        min_value=1, max_value=10,
        value=5
    )
    st.markdown("<span class='small-label'>Adjust to control how many reranked results you see.</span>", unsafe_allow_html=True)

with st.container():
    st.markdown(" ")
    search_clicked = st.button("🔍 Run Semantic Search", use_container_width=True)

# Wrap chroma collection load in spinner to indicate loading process
with st.spinner("Loading embeddings and building index, please wait..."):
    collection = load_chroma_collection()

reranker_model = load_reranker()

if collection is None:
    st.stop()

if search_clicked:
    if not query_text.strip():
        st.warning("Please enter a non-empty query.")
    else:
        with st.spinner("Searching, reranking, and generating explanations..."):
            detailed_results = query_and_search_with_explanations(collection, query_text.strip(), top_k, reranker_model)

        if detailed_results:
            st.subheader(f"🎯 Top {top_k} Re-Ranked Results:")
            for idx, res in enumerate(detailed_results):
                with st.container():
                    st.markdown(
                        f"<div class='result-container'>"
                        f"<span style='font-size:1.05em;'><strong>Result #{idx + 1}</strong></span><br>"
                        f"<span style='color:#666;'>Cosine Distance:</span> <b>{res['distance']:.4f}</b> &nbsp; | &nbsp; "
                        + (f"<span style='color:#666;'>Rerank Score:</span> <b>{res['rerank_score']:.4f}</b>" if res['rerank_score'] is not None else "")
                        + f"<br><span style='color:#38916e;'><b>Act:</b></span> {res['act']}  "
                        f"&nbsp;&nbsp;<span style='color:#8c6feb;'><b>Section:</b></span> {res['section']}<br><br>"
                        f"<b>Description:</b><br>{res['description']}<br><hr style='margin:8px 0;'>"
                        f"<b>Explanation:</b> <br><span style='color:#3c3c3c'>{res['explanation']}</span>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
        else:
            st.warning("❌ No relevant results found for your query.")


st.markdown(
    """
    <hr>
    <div style='text-align:center; color:#888'>
        Powered by <b>Ollama</b>, <b>ChromaDB</b>, <b>Transformers</b>, and <b>Streamlit</b>
    </div>
    """,
    unsafe_allow_html=True
)

