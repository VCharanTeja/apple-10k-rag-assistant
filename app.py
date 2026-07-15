import streamlit as st
from openai import OpenAI
import chromadb
import os
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="Apple 10-K Q&A", layout="centered")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
client = OpenAI(api_key=OPENAI_API_KEY)

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection("apple-10K")

def embed_texts(texts, model="text-embedding-3-small"):
    response = client.embeddings.create(input=texts, model=model)
    return [item.embedding for item in response.data]

def search(question, n_results=5):
    query_embedding = embed_texts([question])[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=n_results)
    retrieved = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        retrieved.append({"text": doc, "source": meta["source"], "page": meta["page"]})
    return retrieved

def generate_answer(question, n_results=5):
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

st.title("📊 Apple 10-K Q&A")
st.caption("Ask a question about Apple's 2023–2025 annual reports. Answers are grounded with citations.")

st.markdown("**Try an example:**")
example_questions = [
    "What were Apple's primary risk factors in 2024 compared to 2023?",
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
        answer, sources = generate_answer(question)

    st.markdown("### Answer")
    st.write(answer)

    st.markdown("**Sources:**")
    badge_row = " ".join([f"`{r['source']}, p.{r['page']}`" for r in sources])
    st.markdown(badge_row)

    with st.expander("View full retrieved passages"):
        for r in sources:
            st.markdown(f"**{r['source']}, page {r['page']}**")
            st.text(r["text"][:300] + "...")
            st.divider()