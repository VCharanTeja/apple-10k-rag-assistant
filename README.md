# Apple 10-K Q&A — RAG Assistant

A Retrieval-Augmented Generation (RAG) system that answers natural-language questions about Apple's SEC 10-K filings (fiscal years 2023–2025), with citations back to the exact source document and page.

## Problem

Business stakeholders often need specific facts from long financial filings (100+ pages each) without manually searching through them. This tool lets a user ask a question in plain English and get a grounded, cited answer in seconds.

## Live Demo

[Try it here](https://apple-10k-rag-assistant-pnwb4dsgpzgttawjxec7yo.streamlit.app/)

## Approach

- **Ingestion:** Extracted text from Apple's 10-K PDFs (fiscal 2023, 2024, 2025) using `pypdf`
- **Chunking:** Token-based chunking (500 tokens, 50-token overlap) using `tiktoken`, preserving page-level source metadata for citations
- **Embeddings:** OpenAI `text-embedding-3-small`
- **Vector store:** ChromaDB (persistent vector database)
- **Retrieval:** Hybrid search combining semantic similarity (vector search) with keyword matching (BM25), followed by a cross-encoder reranking pass to improve relevance ordering before generation
- **Generation:** GPT-4o-mini generates answers strictly grounded in retrieved context, with explicit source/page citations and instructions to say "not found" rather than hallucinate
- **Evaluation:** Built a 5-question test set with known source documents; measured **100% Recall@5** — retrieval correctly found the right source document for every test question
- **Monitoring:** Every query is logged (question, sources retrieved, response latency) to a local database, powering a live, self-updating usage dashboard within the app — showing total queries, average response time, response time per query, and query volume over time
- **Interface:** Streamlit web app with interactive Q&A, clickable example questions, clean citation display, and a live usage stats panel

## Tech Stack

Python · OpenAI API · ChromaDB · Streamlit · pypdf · tiktoken · sentence-transformers (cross-encoder reranking) · rank-bm25 (hybrid search) · SQLite (query logging) · pandas

## Key Design Decisions

- **Token-based chunking with overlap** avoids splitting ideas across chunk boundaries
- **Page-level metadata** attached to every chunk enables accurate citations, not just source-file attribution
- **Hybrid retrieval (vector + BM25)** catches cases where pure semantic search underweights exact keyword matches (e.g., specific financial terms)
- **Cross-encoder reranking** re-scores the top retrieved candidates with a model trained specifically for query-document relevance, improving on similarity search alone
- **Explicit anti-hallucination prompting** (temperature=0, "say so if the answer isn't in the context") reduces the risk of fabricated answers
- **A dedicated evaluation step** measures retrieval accuracy quantitatively rather than relying on manual spot-checks
- **Query logging + a live usage dashboard** apply the same monitoring instinct as a BI/reporting tool to the AI system itself — tracking usage patterns and performance over time

## Known Limitation

Broad, cross-document comparison questions (e.g., comparing narrative sections like risk factors across multiple years) can occasionally underperform relative to direct factual lookups, since retrieval sometimes favors numeric/tabular content that also matches the query's keywords. A production system would likely address this with query decomposition (breaking the question into per-year sub-queries) or section-aware chunking (tagging chunks by document section during ingestion).

## Data Source

Apple Inc. 10-K filings (fiscal years 2023, 2024, 2025), sourced from [SEC EDGAR](https://www.sec.gov/edgar/search). PDFs are not included in this repo (see `.gitignore`) — download the filings directly from EDGAR to reproduce.

## Running Locally

1. Clone this repo
2. Install dependencies: `pip install -r requirements.txt`
3. Add a `.env` file with `OPENAI_API_KEY=your_key_here`
4. Download Apple's 10-K PDFs from SEC EDGAR into a `documents/` folder
5. Run the ingestion/embedding steps in `rag_pipeline.ipynb`
6. Run the app: `streamlit run app.py`
