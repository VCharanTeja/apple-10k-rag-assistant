import streamlit as st
from openai import OpenAI
import chromadb
import os
from dotenv import load_dotenv
load_dotenv()
from sentence_transformers import CrossEncoder
import sqlite3
from datetime import datetime
import time
import pandas as pd

st.set_page_config(page_title="Apple 10-K Q&A", layout="centered")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
client = OpenAI(api_key=OPENAI_API_KEY)

@st.cache_resource
def get_chroma_collection():
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    return chroma_client.get_collection("apple-10K")

collection = get_chroma_collection()

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

from rank_bm25 import BM25Okapi

@st.cache_resource
def build_bm25_index():
    all_data = collection.get(include=["documents", "metadatas"])
    docs = all_data["documents"]
    metas = all_data["metadatas"]
    tokenized = [d.lower().split() for d in docs]
    bm25 = BM25Okapi(tokenized)
    return bm25, docs, metas

bm25_index, bm25_docs, bm25_metas = build_bm25_index()

def embed_texts(texts, model="text-embedding-3-small"):
    response = client.embeddings.create(input=texts, model=model)
    return [item.embedding for item in response.data]

def search(question, n_results=8, initial_pool=30):
    query_embedding = embed_texts([question])[0]
    vector_results = collection.query(query_embeddings=[query_embedding], n_results=initial_pool)
    vector_candidates = []
    for doc, meta in zip(vector_results["documents"][0], vector_results["metadatas"][0]):
        vector_candidates.append({"text": doc, "source": meta["source"], "page": meta["page"]})

    tokenized_query = question.lower().split()
    bm25_scores = bm25_index.get_scores(tokenized_query)
    top_bm25_idx = sorted(range(len(bm25_scores)), key=lambda i: -bm25_scores[i])[:initial_pool]
    bm25_candidates = [
        {"text": bm25_docs[i], "source": bm25_metas[i]["source"], "page": bm25_metas[i]["page"]}
        for i in top_bm25_idx
    ]

    combined = {c["text"]: c for c in vector_candidates + bm25_candidates}
    candidates = list(combined.values())

    pairs = [[question, c["text"]] for c in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])
    return [c for c, s in ranked[:n_results]]

def generate_answer(question, n_results=15):
    retrieved = search(question, n_results=n_results)
    context_blocks = []
    for r in retrieved:
        context_blocks.append(f"[Source: {r['source']}, Page {r['page']}]\n{r['text']}")
    context = "\n\n---\n\n".join(context_blocks)

    prompt = f"""You are a financial analyst assistant. Answer the question using ONLY
the context below. If the answer isn't in the context, say so clearly.

For every claim, cite the source and page in brackets, e.g. [apple_10k_2024.pdf, p.33].

Context:
{context}

Question: {question}

Answer:"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content, retrieved

def init_log_db():
    conn = sqlite3.connect("query_logs.db")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            question TEXT,
            sources TEXT,
            latency_seconds REAL
        )
    """)
    conn.commit()
    return conn

def log_query(conn, question, sources, latency):
    source_list = ", ".join([f"{r['source']} p.{r['page']}" for r in sources])
    conn.execute(
        "INSERT INTO logs (timestamp, question, sources, latency_seconds) VALUES (?, ?, ?, ?)",
        (datetime.now().isoformat(), question, source_list, latency)
    )
    conn.commit()

st.title("📊 Apple 10-K Q&A")
st.caption("Ask a question about Apple's 2023–2025 annual reports. Answers are grounded with citations.")

st.markdown("**Try an example:**")
example_questions = [
    "What are Apple's main risk factors related to supply chain and manufacturing?",
    "Summarize total net sales by product category for fiscal year 2025.",
    "How did legal proceedings affect Apple's 2024 financial notes?",
]

if "question_box" not in st.session_state:
    st.session_state.question_box = ""

cols = st.columns(len(example_questions))
for i, q in enumerate(example_questions):
    if cols[i].button(q, use_container_width=True):
        st.session_state.question_box = q
        st.rerun()

question = st.text_input("Ask a question:", key="question_box")

if question:
    with st.spinner("Searching filings and generating answer..."):
        start_time = time.time()
        answer, sources = generate_answer(question)
        latency = time.time() - start_time
        conn = init_log_db()
        log_query(conn, question, sources, latency)

    st.markdown("### Answer")
    st.markdown(answer.replace("$", "\\$"))

    st.markdown("**Sources:**")
    badge_row = " ".join([f"`{r['source']}, p.{r['page']}`" for r in sources])
    st.markdown(badge_row)

    with st.expander("View full retrieved passages"):
        for r in sources:
            st.markdown(f"**{r['source']}, page {r['page']}**")
            st.text(r["text"][:300] + "...")
            st.divider()

st.divider()
with st.expander("📊 View usage stats"):
    try:
        conn = init_log_db()
        df = pd.read_sql_query("SELECT * FROM logs ORDER BY timestamp DESC", conn)
        if len(df) > 0:
            df["timestamp"] = pd.to_datetime(df["timestamp"])

            col1, col2 = st.columns(2)
            col1.metric("Total queries", len(df))
            col2.metric("Avg response time (sec)", round(df["latency_seconds"].mean(), 2))

            st.markdown("**Response time per query**")
            st.bar_chart(df.sort_values("timestamp").reset_index(drop=True)["latency_seconds"])

            st.markdown("**Queries per day**")
            daily_counts = df.groupby(df["timestamp"].dt.date).size()
            st.bar_chart(daily_counts)

            st.markdown("**Recent queries**")
            st.dataframe(df[["timestamp", "question", "latency_seconds"]], use_container_width=True)
        else:
            st.write("No queries logged yet.")
    except Exception as e:
        st.write("No logs yet.")