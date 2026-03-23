<h1 align="center">Document-grounded Q&A Chatbot for Study Materials</h1>

<p align="center">
  <strong>A cost-optimized and accurate Retrieval-Augmented Generation (RAG) chatbot for grounded Q&A on study materials.</strong>
</p>

## Project Overview

This chatbot is designed to read, understand, and answer questions grounded in your own study materials (PDF and Word documents).

Built with scalability, high precision, and cost optimization in mind, this project avoids common pitfalls of naive RAG architectures by implementing Hybrid Search, Semantic Chunking, Query Rewriting, and Cross-Encoder Reranking, while shifting heavy generative operations to local LLMs.

The result is a practical, production-minded document Q&A assistant with strong retrieval quality and transparent references.

## Key Engineering Features
### 1. Advanced Document Ingestion

- Docling Parser:
Replaces naive PDF loading by preserving richer document structure such as tables, lists, and layout.

- Semantic Chunking:
Uses SemanticChunker with local HuggingFace embeddings (all-MiniLM-L6-v2) to split by meaning instead of fixed character windows.

### 2. Industry-Standard Retrieval (Hybrid Search + Reranker)

- BM25 (Sparse) + Cosine Vector (Dense) Search:
Combines keyword precision (BM25) with semantic matching (Cohere embeddings), retrieving top 15 from each branch.

- Strict Deduplication and Reranking:
Merges up to 30 chunks, removes duplicates, then reranks using Cohere rerank-multilingual-v3.0.

- Relevance Threshold:
Applies a minimum rerank score threshold (> 0.35) before passing chunks to the generator, reducing hallucinated outputs.

### 3. Context-Aware Conversational Memory

- Query Rewriting:
Uses the last 10 messages as memory buffer to convert follow-up questions into standalone retrieval-ready queries.

### 4. Cloud-Ready LLM Architecture

- Gemini API Models:
Runs generation and query rewrite via Gemini API to support server deployment (Streamlit Cloud).

- Rate-Limit Throttling:
Embeds in controlled batches (25 chunks per request) with a short delay between batches to stay stable on free-tier limits.

## Tech Stack

- UI Framework:
Streamlit (interactive chat interface with expandable reference citations)

- Framework:
LangChain, LangChain Core, LangChain Experimental

- Vector Store:
ChromaDB (local SQLite-based)

- Embeddings and Reranking:
Cohere (embed-multilingual-v3.0, rerank-multilingual-v3.0), HuggingFace (all-MiniLM-L6-v2)

- LLM Engine:
Gemini API (default model: gemini-1.5-flash)

- Document Parsing:
Docling

## Environment Requirements

- Python:
3.13 (recommended and currently used in this project)

- API Keys:
`COHERE_API_KEY` and `GEMINI_API_KEY` are required.

## Project Structure

```text
Academic_Chatbot/
|- .env
|- app.py
|- README.md
|- requirements.txt
|- vector_store/
|- temp_uploads/
|- .streamlit/
|  |- config.toml
|  |- secrets.toml.example
|- ingestion_pipeline/
|  |- __init__.py
|  |- chunker.py
|  |- embedder.py
|  |- loader.py
|  |- orchestrator.py
|- retrieval_pipeline/
|  |- generator.py
|  |- query_rewriter.py
|  |- retriever.py
|- utils/
|  |- __init__.py
|  |- config.py
|  |- file_handler.py
|  |- session_manager.py
```

## Quick Start

1. Install dependencies

```bash
pip install -r requirements.txt
```

2. Configure environment file

```env
COHERE_API_KEY=your_cohere_api_key
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-1.5-flash
CHROMA_DB_PATH=./vector_store
```

3. Start the app

```bash
streamlit run app.py
```

## Deploy on Streamlit Cloud

1. Push project to GitHub (do not commit `.env`).
2. Open Streamlit Cloud and create app from your GitHub repo.
3. Set `Main file path` to `app.py`.
4. In Streamlit app settings, open `Secrets` and paste:

```toml
COHERE_API_KEY = "your_cohere_api_key"
GEMINI_API_KEY = "your_gemini_api_key"
GEMINI_MODEL = "gemini-1.5-flash"
CHROMA_DB_PATH = "./vector_store"
```

5. Click Deploy. Streamlit will return a public URL you can share with recruiters.

## Why This Project Stands Out

- Uses modern retrieval design beyond basic vector top-k.
- Balances quality, latency, and cost with practical engineering tradeoffs.
- Provides transparent answer references in UI for trust and verification.
- Focuses on grounded answers from uploaded materials instead of generic open-domain responses.
