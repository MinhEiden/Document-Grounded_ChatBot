<h1 align="center">Academic Research Chatbot (Advanced RAG)</h1>

<p align="center">
  <strong>An elegant, cost-optimized, and highly accurate Retrieval-Augmented Generation (RAG) system for academic research and document Q&A.</strong>
</p>

## Project Overview

Academic Chatbot is a specialized conversational AI system designed to read, comprehend, and answer questions based on academic papers (PDF and Word documents).

Built with scalability, high precision, and cost optimization in mind, this project avoids common pitfalls of naive RAG architectures by implementing Hybrid Search, Semantic Chunking, Query Rewriting, and Cross-Encoder Reranking, while shifting heavy generative operations to local LLMs.

The result is a practical, production-minded RAG assistant with strong retrieval quality and transparent references.

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

### 4. Cost-Optimized and Local-First Architecture

- Local Generative Models:
Runs generation and rewrite on Ollama models (llama3.2, qwen, gemma2) to minimize token cost.

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
Ollama (local Llama 3.2 by default)

- Document Parsing:
Docling

## Environment Requirements

- Python:
3.13 (recommended and currently used in this project)

- Ollama:
Required for local generation model (default: llama3.2)

## Project Structure

```text
Academic_Chatbot/
|- .env
|- app.py
|- README.md
|- requirements.txt
|- chroma_db/
|- ingestion_pipeline/
|  |- __init__.py
|  |- chunker.py
|  |- embedder.py
|  |- loader.py
|- retrieval_pipeline/
|  |- generator.py
|  |- query_rewriter.py
|  |- retriever.py
|- TrainData/
```

## Quick Start

1. Install dependencies

```bash
pip install -r requirements.txt
```

2. Configure environment file

```env
COHERE_API_KEY=your_cohere_api_key
CHROMA_DB_PATH=./chroma_db
TRAIN_DATA_PATH=./TrainData
```

3. Prepare local model

```bash
ollama pull llama3.2
```

4. Run ingestion

```bash
python -c 'from ingestion_pipeline import run_ingestion_pipeline; run_ingestion_pipeline()'
```

5. Start the app

```bash
streamlit run app.py
```

## Why This Project Stands Out

- Uses modern retrieval design beyond basic vector top-k.
- Balances quality, latency, and cost with practical engineering tradeoffs.
- Provides transparent answer references in UI for trust and verification.
