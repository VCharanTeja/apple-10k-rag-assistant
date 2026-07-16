# Apple 10-K Q&A — RAG Assistant

A Retrieval-Augmented Generation (RAG) system that answers natural-language questions about Apple's SEC 10-K filings (fiscal years 2023–2025), with citations back to the exact source document and page.

## Problem

Business stakeholders often need specific facts from long financial filings (100+ pages each) without manually searching through them. This tool lets a user ask a question in plain English and get a grounded, cited answer in seconds.

## Live Demo

https://apple-10k-rag-assistant-pnwb4dsgpzgttawjxec7yo.streamlit.app
## Approach

- **Ingestion:** Extracted text from Apple's 10-K PDFs (fiscal 2023, 2024, 2025) using `pypdf`
- **Chunking:** Token-based chunking (500 tokens, 50-token overlap) using `tiktoken`, preserving page-level source metadata for citations
- **Embeddings:** OpenAI `text-embedding-3-small`
- **Vector store:** ChromaDB (persistent local vector database)
- **Retrieval:** Semantic similarity search over chunk embeddings
- **Generation:** GPT-4o-mini generates answers strictly grounded in retrieved context, with explicit source/page citations and instructions to say "not found" rather than hallucinate
- **Evaluation:** Built a 5-question test set with known source documents; measured **100% Recall@5** — retrieval correctly found the right source document for every test question
- **Interface:** Streamlit web app with interactive Q&A, clickable example questions, and clean citation display for verifying answers against source pages

## Tech Stack

Python · OpenAI API · ChromaDB · Streamlit · pypdf · tiktoken

## Key Design Decisions

- **Token-based chunking with overlap** avoids splitting ideas across chunk boundaries
- **Page-level metadata** attached to every chunk enables accurate citations, not just source-file attribution
- **Explicit anti-hallucination prompting** (temperature=0, "say so if the answer isn't in the context") reduces the risk of fabricated answers
- **A dedicated evaluation step** measures retrieval accuracy quantitatively rather than relying on manual spot-checks

## Data Source

Apple Inc. 10-K filings (fiscal years 2023, 2024, 2025), sourced from [SEC EDGAR](https://www.sec.gov/edgar/search). PDFs are not included in this repo (see `.gitignore`) — download the filings directly from EDGAR to reproduce.

## Running Locally

1. Clone this repo
2. Install dependencies: `pip install chromadb openai pypdf tiktoken streamlit pandas python-dotenv`
3. Add a `.env` file with `OPENAI_API_KEY=your_key_here`
4. Download Apple's 10-K PDFs from SEC EDGAR into a `documents/` folder
5. Run the ingestion/embedding steps in `rag_pipeline.ipynb`
6. Run the app: `streamlit run app.py`
