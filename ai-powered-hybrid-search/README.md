# 🚀 AI-Powered Hybrid Search: Merging Elasticsearch with FAISS & LLMs for Smarter Search  

Search engines drive **product discovery in e-commerce**, but traditional **keyword-based search** often fails to understand **context, synonyms, and user intent**. While working on a **product search system for an e-commerce Virtual Assistant (VA) services provider**, we faced challenges where users couldn't find relevant products due to limitations in **exact-match keyword searches**. To solve this, we built a **Hybrid Search solution** that combines **Lexical Search (Elasticsearch) and Semantic Search (FAISS) with AI-powered ranking (LLMs)**, delivering **faster, more accurate, and context-aware search results**.

This article explains how we **implemented Hybrid Search** and how the code works.

---

## 🔹 How the Code Works  

### 1️⃣ **Loading Pre-trained Models**  
We use **`multi-qa-mpnet-base-dot-v1`** for generating **vector embeddings** and **`cross-encoder/ms-marco-MiniLM-L-12-v2`** for **AI-powered re-ranking**.  

```python
embedding_model = SentenceTransformer("multi-qa-mpnet-base-dot-v1")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")
```

These models help in **semantic understanding and ranking** of search results.

---

### 2️⃣ **Elasticsearch for Keyword-Based Lexical Search (BM25)**  
We use **Elasticsearch BM25** to perform **traditional keyword-based search**.

```python
query_body = {"query": {"multi_match": {"query": query, "fields": ["title", "description"]}}}
es_response = es_client.search(index=index_name, body=query_body, size=top_k)
lexical_results = [{**hit["_source"], "score": hit["_score"]} for hit in es_response["hits"]["hits"]]
```

✔️ **Pros**: Fast and efficient for exact keyword matches  
❌ **Cons**: Fails to match synonyms or understand search intent  

---

### 3️⃣ **FAISS for Semantic Search (Vector Similarity Matching)**  
We store product descriptions as **vector embeddings** and use **FAISS** for **Approximate Nearest Neighbor (ANN) search**.

```python
query_vector = embedding_model.encode([query], normalize_embeddings=True)
D, I = index.search(np.array(query_vector), k=top_k)
semantic_results = [{"title": products[i]["title"], "description": products[i]["description"], "score": float(D[0][j])} for j, i in enumerate(I[0])]
```

✔️ **Pros**: Finds similar results even if words differ  
❌ **Cons**: Can return **slightly irrelevant** results without re-ranking  

---

### 4️⃣ **Merging Lexical & Semantic Search (Weighted Scoring)**  
We dynamically balance **keyword-based (BM25) and semantic (FAISS) search results**.

```python
alpha = 0.5 if len(query.split()) > 2 else 0.7

results = {}

for doc in semantic_results:
    results[doc["title"]] = {"title": doc["title"], "description": doc["description"], "score": (1 - alpha) * doc["score"]}

for doc in lexical_results:
    if doc["title"] in results:
        results[doc["title"]]["score"] += alpha * doc["score"]
    else:
        results[doc["title"]] = {"title": doc["title"], "description": doc["description"], "score": alpha * doc["score"]}
```

✔️ **Combines strengths of both methods**  
✔️ **Prioritizes results that appear in both searches**  

---

### 5️⃣ **AI-Powered Re-Ranking with Cross-Encoders (LLMs)**  
Even after merging, **ranking may still be imperfect**, so we use **LLMs for re-ranking**.

```python
query_doc_pairs = [(query, doc["title"] + " " + doc["description"]) for doc in initial_results]
reranking_scores = reranker.predict(query_doc_pairs)
```

To normalize the scores:

```python
min_score = min(reranking_scores)
max_score = max(reranking_scores)
normalized_scores = [(score - min_score) / (max_score - min_score + 1e-6) * 10 for score in reranking_scores]

for i, doc in enumerate(initial_results):
    doc["score"] = float(normalized_scores[i])
```

✔️ **Ensures most relevant products rank highest**  
✔️ **Filters out irrelevant results dynamically**  

---

## 🚀 Test Drive: Running the Hybrid Search  
We run our **Hybrid Search with Re-Ranking** for the query:

```python
query = "Bluetooth headphones"
results = hybrid_search_with_reranking(query, top_k=5, score_threshold=1.5)

for res in results:
    print(f"Title: {res['title']}, Score: {res['score']}")
```

### **✅ Search Results**
```
Title: Wireless Headphones, Score: 9.999999444802226
Title: Bluetooth Speaker, Score: 6.483402361453368
```

✔️ **Wireless Headphones ranked highest as it matches the query exactly**  
✔️ **Bluetooth Speaker ranked lower but still relevant**  

---

## 🚀 Further Improvements and Optimizations  

To improve this **Hybrid Search system**, we can integrate **more advanced AI models**:  

- **Use Larger Embedding Models** → Replace `multi-qa-mpnet-base-dot-v1` with `bge-base-en-v1.5` or `text-embedding-ada-002` for better semantic search.  
- **Use a Stronger Re-Ranking Model** → Upgrade from `cross-encoder/ms-marco-MiniLM-L-12-v2` to `cross-encoder/ms-marco-MultiBERT-L-12-v2` for improved ranking.  
- **Use FAISS HNSW (Hierarchical Navigable Small World)** → Instead of `IndexIVFFlat`, use `IndexHNSWFlat` for **faster retrieval with high accuracy**.  
- **Deploy as a Real-Time API** → Wrap in **FastAPI** for seamless integration with e-commerce platforms.  

By incorporating **these optimizations**, we can make **AI-powered search even smarter and more efficient**! 🚀🔥  

---

## 📌 Source Code  
The complete working **Hybrid Search implementation** can be found on **GitHub**:  
🔗 [GitHub Repository: fistix/ai](https://github.com/fistix/ai/tree/main/ai-powered-hybrid-search) 

Clone the repository using:  
```sh
git clone https://github.com/fistix/ai.git
```

Explore the code, contribute, and experiment with improvements! 🚀🔥  

---

#AI #HybridSearch #Ecommerce #Elasticsearch #FAISS #MachineLearning #LLMs
