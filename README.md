# Rag_based_LLM

# 🔍 Legal Search with Explanations

A powerful Streamlit web application for semantic search over legal documents with AI-generated explanations. This app combines state-of-the-art embedding models, approximate nearest neighbor (ANN) search, neural reranking, and LLM-based explanations to help users efficiently find and understand relevant legal provisions.

---

## 🚀 Overview

This project implements a semantic legal document search system featuring:

- **Ollama Embeddings:** Uses Ollama's `"nomic-embed-text"` model to convert legal text into semantic vectors.
- **ChromaDB ANN Search:** Employs ChromaDB for fast approximate nearest neighbor retrieval with optimized HNSW indexing.
- **Cross-Encoder Reranking:** Applies a dense cross-encoder (`ms-marco-MiniLM-L-6-v2`) for refining initial results.
- **LLaMA 2 LLM Explanation Generation:** Calls Ollama's LLM (`llama2:latest`) to generate concise, human-readable explanations for each search result.
- **Streamlit UI:** Interactive, clean interface with query input, dynamic result counts, rerank scoring, and explanation display.

---

## 🔎 Features

- **Efficient Embedding & Index Caching:** Uses Streamlit resource caching to avoid repeated embedding computation and index rebuilding.
- **Batch Processing with Progress Bar:** Displays progress indicators during initial embedding creation to improve user experience.
- **Configurable Top-K Results:** Easily adjust how many top results to display and rerank.
- **Concurrent Explanation Generation:** Runs explanation requests in parallel to speed up response time.
- **Robust Timeout and Retry Handling:** Increased HTTP timeout and retry logic to improve reliability when communicating with Ollama server.
- **Styled Result Display:** Shows cosine distances, rerank scores, legal metadata (`act` and `section`), document excerpts, and AI explanations in a visually appealing manner.

---

## 🎯 Use Cases

- Legal researchers and practitioners seeking semantically relevant legal text excerpts.
- Developers exploring integrations of embeddings, ANN search, cross-encoders, and LLM explanations.
- Organizations aiming to add explainable AI-powered legal search capabilities to their workflows.

---

## ⚙️ How It Works

1. **Data Loading:** Reads legal documents from `bns.csv` (must include `description`, `act`, `section` columns).
2. **Embedding Generation:** Generates semantic vectors for each document using Ollama's embedding model.
3. **Indexing:** Builds an ANN index with ChromaDB optimized via HNSW parameters for fast similarity search.
4. **Query Processing:** Accepts user search text and retrieves candidate documents via vector similarity.
5. **Reranking:** Applies a neural cross-encoder to reorder candidates based on refined relevance scoring.
6. **Explanation Generation:** For each top reranked candidate, generates a concise explanation of relevance using an LLM.
7. **Display:** Presents results with metadata, similarity measures, rerank scores, document excerpts, and explanations.

---

## 📋 Requirements

- Python 3.8 or higher
- Python packages:
  - `streamlit`
  - `pandas`
  - `chromadb`
  - `sentence-transformers`
  - `requests`
  - `numpy`
- Ollama server running locally or remotely with:
  - `"nomic-embed-text"` embedding model
  - `"llama2:latest"` LLM model
- A CSV file (`bns.csv`) containing legal documents with necessary columns.

---

## 🛠️ Installation & Setup

1. Clone this repository:
    ```
    git clone https://github.com/yourusername/your-repo-name.git
    cd your-repo-name
    ```

2. Install required Python packages:
    ```
    pip install streamlit pandas chromadb sentence-transformers requests numpy
    ```

3. Ensure Ollama server is running at the configured API URL (`http://localhost:11434/api` by default) with required models loaded.

4. Place your `bns.csv` file in the root directory. This file should contain legal document descriptions and metadata.

---

## 🚀 Running the App

Launch the Streamlit app with:

streamlit run 3_08_2025_2.py


- Enter your legal query in the input area.
- Adjust the "Number of top results" slider.
- Click "Run Semantic Search" to retrieve and rerank relevant legal documents with explanations.

---

## 📝 Project Structure

- `app.py`: Main Streamlit application implementing the full search pipeline.
- `bns.csv`: Legal documents dataset (user-provided).
- `logo.jpg`: Optional sidebar logo image.
- `README.md`: This documentation file.

---

## ⚙️ Configuration

Adjustable parameters within `app.py`:

- **OLLAMA_EMBED_MODEL:** Model name for embedding (default: `"nomic-embed-text"`).
- **OLLAMA_LLM_MODEL:** LLM for explanation generation (default: `"llama2:latest"`).
- **OLLAMA_API_URL:** Base URL for Ollama API.
- **CHROMA_COLLECTION_NAME:** ChromaDB collection identifier.
- **BATCH_SIZE:** Number of documents processed per batch during indexing.
- **RERANK_LIMIT:** Maximum number of results to rerank per query.
- **HNSW parameters:** Tuning parameters for ANN indexing performance.

---

## 🛡️ Error Handling & Reliability

- Embedding and reranker loading use Streamlit caching to enhance performance.
- Explanation generation includes a 120-second timeout and retry logic on failure.
- User-friendly error messages through Streamlit UI for missing files or API issues.

---

## 💡 Acknowledgements

This project leverages:

- [Ollama](https://ollama.com) for embeddings & language models
- [ChromaDB](https://chroma.com) for vector similarity search
- [Sentence Transformers](https://www.sbert.net) cross-encoder models for reranking
- [Streamlit](https://streamlit.io) for creating the web app interface

---

## 📄 License

This project is licensed under the MIT License. See `LICENSE` for details.

---

## 🤝 Contribution

Contributions, issues, and feature requests are welcome! Feel free to open an issue or submit a pull request.

---

## 🔗 Contact

For questions or support, please contact [your-email@example.com].

---

Enjoy exploring and extending your legal search with AI-powered explanations! 🚀


