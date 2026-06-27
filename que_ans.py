import os
import re
import numpy as np
import faiss

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import pipeline

TEXT_FOLDER = "txt_file"

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GEN_MODEL = "google/flan-t5-base"

TOP_K = 8
FINAL_K = 4

print("Loading models...")

embedder = SentenceTransformer(EMBED_MODEL)
reranker = CrossEncoder(RERANK_MODEL)

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def generate(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_new_tokens=200)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def clean_text(t):
    t = t.replace("\n", " ")
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def chunk_text(text, size=700, overlap=150):
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks, current = [], ""

    for s in sentences:
        if len(current) + len(s) < size:
            current += " " + s
        else:
            chunks.append(current.strip())
            current = current[-overlap:] + " " + s

    if current:
        chunks.append(current.strip())

    return chunks

texts, metadata = [], []

print("Loading documents...")

for f in os.listdir(TEXT_FOLDER):
    if f.endswith(".txt"):
        path = os.path.join(TEXT_FOLDER, f)

        with open(path, "r", encoding="utf-8") as file:
            raw = clean_text(file.read())
            chunks = chunk_text(raw)

            for i, c in enumerate(chunks):
                texts.append(c)
                metadata.append({"file": f, "chunk": i})

print("Chunks:", len(texts))


print("Building vector index...")

embeddings = embedder.encode(texts, convert_to_numpy=True).astype("float32")
faiss.normalize_L2(embeddings)

index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)


tokenized = [t.lower().split() for t in texts]
bm25 = BM25Okapi(tokenized)


def rewrite_query(q):
    return [
        q,
        f"Explain: {q}",
        f"Details about {q}"
    ]


def retrieve(query):

    queries = rewrite_query(query)

    candidate_scores = {}

  
    for q in queries:
        tokens = q.lower().split()
        scores = bm25.get_scores(tokens)

        for i, s in enumerate(scores):
            candidate_scores[i] = candidate_scores.get(i, 0) + float(s)

    q_emb = embedder.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_emb)

    _, idxs = index.search(q_emb, TOP_K)

    for rank, idx in enumerate(idxs[0]):
        candidate_scores[idx] = candidate_scores.get(idx, 0) + (1.0 / (rank + 1))

    ranked = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)

    top = ranked[:TOP_K]

    results = []
    for idx, score in top:
        results.append({
            "idx": idx,
            "text": texts[idx],
            "score": score,
            "meta": metadata[idx]
        })

    return results


def rerank(query, results):
    pairs = [(query, r["text"]) for r in results]
    scores = reranker.predict(pairs)

    for r, s in zip(results, scores):
        r["rerank_score"] = float(s)

    results.sort(key=lambda x: x["rerank_score"], reverse=True)
    return results[:FINAL_K]


def expand(results):
    seen = set()
    expanded = []

    for r in results:
        i = r["idx"]

        for j in [i-1, i, i+1]:
            if 0 <= j < len(texts) and j not in seen:
                seen.add(j)
                expanded.append(texts[j])

    return expanded


def generate(context, query):

    prompt = f"""
You are a precise assistant.

Rules:
- Use ONLY the context.
- If missing, say "I don't know based on context".

Context:
{context}

Question:
{query}

Answer:
"""

  
def generate(context, query):

    prompt = f"""
You are a precise assistant.

Rules:
- Use ONLY the context.
- If missing, say "I don't know based on context".

Context:
{context}

Question:
{query}

Answer:
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=False
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

    

def ask(query):

    print("\n====================")
    print("QUERY:", query)
    print("====================\n")
    
    results = retrieve(query)
    results = rerank(query, results)

    print("TOP DOCS:")
    for r in results:
        print(f"{r['meta']['file']} | chunk {r['meta']['chunk']} | score {r['rerank_score']:.4f}")
    context = expand(results)
    context = "\n\n---\n\n".join(context)

    context = context[:4000]  # safety
    print("\nANSWER:\n")
    print(generate(context, query))

while True:
    q = input("\nAsk (exit to stop): ")
    if q.lower() == "exit":
        break
    ask(q)
    