import streamlit as st
import os
import re
import numpy as np
import faiss

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ================= CONFIG =================
TEXT_FOLDER = "txt_file"

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GEN_MODEL = "google/flan-t5-base"

TOP_K = 8
FINAL_K = 4

# ================= MODELS =================
embedder = SentenceTransformer(EMBED_MODEL)
reranker = CrossEncoder(RERANK_MODEL)

tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL)

# ================= TEXT =================
def clean_text(t):
    return re.sub(r"\s+", " ", t.replace("\n", " ")).strip()

def chunk_text(text, size=700, overlap=150):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, cur = [], ""

    for s in sentences:
        if len(cur) + len(s) < size:
            cur += " " + s
        else:
            chunks.append(cur.strip())
            cur = cur[-overlap:] + " " + s

    if cur:
        chunks.append(cur.strip())

    return chunks

# ================= LOAD DATA =================
texts, metadata = [], []

for f in os.listdir(TEXT_FOLDER):
    if f.endswith(".txt"):
        with open(os.path.join(TEXT_FOLDER, f), "r", encoding="utf-8") as file:
            raw = clean_text(file.read())
            chunks = chunk_text(raw)

            for i, c in enumerate(chunks):
                texts.append(c)
                metadata.append({"file": f, "chunk": i})

# ================= INDEX =================
embeddings = embedder.encode(texts, convert_to_numpy=True).astype("float32")
faiss.normalize_L2(embeddings)

index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

bm25 = BM25Okapi([t.lower().split() for t in texts])

# ================= RETRIEVE =================
def retrieve(query):
    scores_map = {}

    scores = bm25.get_scores(query.lower().split())
    for i, s in enumerate(scores):
        scores_map[i] = float(s)

    q_emb = embedder.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_emb)

    _, idxs = index.search(q_emb, TOP_K)

    for r, idx in enumerate(idxs[0]):
        scores_map[idx] = scores_map.get(idx, 0) + (1 / (r + 1))

    ranked = sorted(scores_map.items(), key=lambda x: x[1], reverse=True)

    return [{"idx": i, "text": texts[i], "meta": metadata[i]} for i, _ in ranked[:TOP_K]]

# ================= RERANK =================
def rerank(query, results):
    pairs = [(query, r["text"]) for r in results]
    scores = reranker.predict(pairs)

    for r, s in zip(results, scores):
        r["score"] = float(s)

    return sorted(results, key=lambda x: x["score"], reverse=True)[:FINAL_K]

# ================= CONTEXT =================
def build_context(results):
    return "\n\n---\n\n".join([r["text"] for r in results])[:4000]

# ================= GENERATE =================
def generate(context, query):

    prompt = f"""
Use ONLY context.

Context:
{context}

Question:
{query}

Answer:
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_new_tokens=200)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


import streamlit as st

st.set_page_config(layout="wide")

st.title("🚀 RAG Chat App")

query = st.text_input("Ask something:")

if query:

    with st.spinner("Thinking..."):

        results = retrieve(query)
        results = rerank(query, results)
        context = build_context(results)
        answer = generate(context, query)

    st.subheader("🤖 Answer")

    # BIGGER OUTPUT (no small box)
    st.markdown(f"""
    <div style="
        background:#111827;
        color:white;
        padding:18px;
        border-radius:10px;
        font-size:18px;
        line-height:1.6;
    ">
        {answer}
    </div>
    """, unsafe_allow_html=True)

    st.subheader("📄 Retrieved Chunks")

    for r in results:

        st.markdown(f"""
        <div style="
            background:#0f172a;
            padding:12px;
            border-radius:10px;
            margin-bottom:10px;
            color:#e5e7eb;
        ">
            <b>📄 {r['meta']['file']} | chunk {r['meta']['chunk']}</b>
            <br><br>
            {r['text']}
        </div>
        """, unsafe_allow_html=True)
