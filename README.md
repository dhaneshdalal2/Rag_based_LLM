# Rag_based_LLM

Legal Search with AI-Powered Explanations
This is a Streamlit-based web application for semantic search over legal documents, enhanced with AI-driven explanations to improve transparency and user understanding.

Key Features
Semantic Embeddings with Ollama: Utilizes the "nomic-embed-text" model to generate meaningful vector representations of legal documents for more accurate search.

Fast Vector Search via ChromaDB: Leverages an approximate nearest neighbor (ANN) index with optimized HNSW parameters for high-speed similarity search on large corpora.

Neural Re-ranking: Applies a Cross-Encoder model (ms-marco-MiniLM-L-6-v2) to rerank the top candidates from the initial search, boosting result relevance.

LLM-Generated Explanations: Uses Ollama's LLaMA 2 model (llama2:latest) to generate concise, natural language explanations clarifying why each result matches the user query.

User-Friendly UI: Interactive query input, adjustable result count slider, progress indicators during embedding creation, and visually rich display of search results with scores and explanations.

Caching & Efficiency: Implements Streamlit caching for embeddings and model loading to reduce latency and avoid redundant computations.

Robustness: Built-in HTTP timeout and retry logic to handle slow LLM responses gracefully.

Intended Use
Ideal for legal professionals, researchers, and developers looking for an AI-augmented legal research assistant that not only finds relevant documents but also explains their relevance in human terms.

How It Works
At first launch, the app reads legal documents from bns.csv, builds semantic embeddings with Ollama, and indexes them in ChromaDB.

The user inputs a free-text legal query.

ChromaDB retrieves the closest matches based on embeddings.

The Cross-Encoder reranks these matches for better accuracy.

For each top result, an LLM generates an explanation describing why it is relevant.

Results with detailed metadata, scores, and AI-generated explanations are displayed.

Requirements
Ollama API server running locally with required embedding and LLM models.

CSV file (bns.csv) containing legal data with fields like description, act, and section.

Python environment with necessary packages (streamlit, chromadb, sentence-transformers, requests, pandas, etc.)
