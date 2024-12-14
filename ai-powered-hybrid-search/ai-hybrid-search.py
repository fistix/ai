# AI-Powered Hybrid Search: Merging Elasticsearch with FAISS & LLMs for Smarter Search

import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
from elasticsearch import Elasticsearch


# Sample product data
products = [
    {
        "title": "Wireless Headphones",
        "description": "Noise-canceling Bluetooth headphones",
        "category": "Electronics",
    },
    {
        "title": "Running Shoes",
        "description": "Lightweight sports shoes for jogging",
        "category": "Footwear",
    },
    {
        "title": "Smartphone",
        "description": "Latest Android phone with OLED display",
        "category": "Electronics",
    },
    {
        "title": "Gaming Laptop",
        "description": "Powerful gaming laptop with high-end GPU",
        "category": "Computers",
    },
    {
        "title": "Fitness Tracker",
        "description": "Smart fitness band with heart rate monitor",
        "category": "Wearables",
    },
    {
        "title": "4K TV",
        "description": "55-inch Ultra HD Smart TV with HDR",
        "category": "Home Appliances",
    },
    {
        "title": "Bluetooth Speaker",
        "description": "Portable wireless speaker with deep bass",
        "category": "Electronics",
    },
    {
        "title": "Mechanical Keyboard",
        "description": "RGB backlit gaming keyboard",
        "category": "Computers",
    },
    {
        "title": "Electric Toothbrush",
        "description": "Rechargeable electric toothbrush with smart timer",
        "category": "Personal Care",
    },
    {
        "title": "Dumbbells Set",
        "description": "Adjustable dumbbells for home workouts",
        "category": "Fitness",
    },
]
# elasticsearch index
index_name = "products"

# Load an embedding model
embedding_model = SentenceTransformer("multi-qa-mpnet-base-dot-v1")

# Load a pre-trained cross-encoder model
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")

# Connect to Elasticsearch (disable SSL verification)
es_client = Elasticsearch(
    hosts=[{"host": "localhost", "port": 9200, "scheme": "https"}],  # Use HTTPS
    basic_auth=("elastic", "PASSWORD"),  # Replace with your Elasticsearch credentials 
    verify_certs=False,  # Ignore self-signed SSL issues
)


FAISS_INDEX_FILE = "faiss_index.bin"


def save_faiss_index(index):
    faiss.write_index(index, FAISS_INDEX_FILE)


def load_faiss_index():
    if os.path.exists(FAISS_INDEX_FILE):
        return faiss.read_index(FAISS_INDEX_FILE)
    return None


def init_faiss():
    index = load_faiss_index()
    if index:
        return index  # Load existing index

    # Generate vector embeddings for products
    product_texts = [p["title"] + " " + p["description"] for p in products]
    product_vectors = embedding_model.encode(product_texts, normalize_embeddings=True)
    product_vectors = np.array(product_vectors)

    dim = product_vectors.shape[1]

    # 🔹 Ensure num_clusters is ≤ number of data points
    num_clusters = min(len(products), 8)  # Reduce to 8 or fewer clusters

    quantizer = faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, num_clusters, faiss.METRIC_L2)

    # 🔹 Train FAISS index only if there are enough samples
    if len(products) >= num_clusters:
        index.train(product_vectors)
    else:
        print(
            f"Warning: Not enough data to train FAISS index with {num_clusters} clusters. Using Flat index."
        )
        index = faiss.IndexFlatL2(dim)  # Fallback to FlatL2

    index.add(product_vectors)

    save_faiss_index(index)
    return index


index = init_faiss()


def init_elasticsearch():

    # Test the connection
    print(es_client.info())

    # Create index if it doesn't exist
    if not es_client.indices.exists(index=index_name):
        es_client.indices.create(
            index=index_name,
            body={
                "mappings": {
                    "properties": {
                        "title": {"type": "text"},
                        "description": {"type": "text"},
                        "category": {"type": "keyword"},
                    }
                }
            },
        )


def hybrid_search(query, top_k=10):
    """
    Improved hybrid search with dynamic weight adjustments.
    - Adjusts lexical and semantic importance dynamically.
    - Returns top `k` merged results with improved ranking.
    """

    # 🔹 Step 1: Lexical Search (Elasticsearch)
    query_body = {
        "query": {"multi_match": {"query": query, "fields": ["title", "description"]}}
    }
    es_response = es_client.search(index=index_name, body=query_body, size=top_k)
    lexical_results = [
        {**hit["_source"], "score": hit["_score"]}
        for hit in es_response["hits"]["hits"]
    ]

    # 🔹 Step 2: Semantic Search (FAISS)
    query_vector = embedding_model.encode([query], normalize_embeddings=True)
    D, I = index.search(np.array(query_vector), k=top_k)
    semantic_results = [
        {
            "title": products[i]["title"],
            "description": products[i]["description"],
            "score": float(D[0][j]),
        }
        for j, i in enumerate(I[0])
    ]

    # 🔹 Step 3: Dynamic Weight Adjustment for Lexical & Semantic Search
    alpha = 0.5 if len(query.split()) > 2 else 0.7

    # 🔹 Step 4: Merge & Score Adjustment
    results = {}

    for doc in semantic_results:
        results[doc["title"]] = {
            "title": doc["title"],
            "description": doc["description"],
            "score": (1 - alpha) * doc["score"],
        }

    for doc in lexical_results:
        if doc["title"] in results:
            results[doc["title"]]["score"] += alpha * doc["score"]
        else:
            results[doc["title"]] = {
                "title": doc["title"],
                "description": doc["description"],
                "score": alpha * doc["score"],
            }

    return sorted(results.values(), key=lambda x: x["score"], reverse=True)[:top_k]


def hybrid_search_with_reranking(query, top_k=5, score_threshold=0.0):
    """
    AI-powered re-ranking after hybrid search.
    - Uses Min-Max normalization for better scoring.
    - Dynamically adjusts score threshold if needed.
    """

    # 🔹 Step 1: Perform Hybrid Search
    initial_results = hybrid_search(query, top_k=top_k + 5)

    if not initial_results:
        print("🔹 No initial search results found.")
        return []

    # 🔹 Step 2: Prepare Query-Document Pairs for AI Re-Ranking
    query_doc_pairs = [
        (query, doc["title"] + " " + doc["description"]) for doc in initial_results
    ]

    # 🔹 Step 3: Compute AI-Based Ranking Scores
    reranking_scores = reranker.predict(query_doc_pairs)

    # 🔹 Step 4: Apply Min-Max Scaling for Normalized Scores
    min_score = min(reranking_scores)
    max_score = max(reranking_scores)
    normalized_scores = [
        (score - min_score) / (max_score - min_score + 1e-6) * 10
        for score in reranking_scores
    ]

    # 🔹 Step 5: Assign Normalized Scores
    for i, doc in enumerate(initial_results):
        doc["score"] = float(normalized_scores[i])

    # 🔹 Step 6: Dynamic Score Thresholding
    max_score = max(normalized_scores)
    min_score = min(normalized_scores)
    score_threshold = max(
        0.3, min(score_threshold, max_score - (max_score - min_score) * 0.2)
    )

    # 🔹 Step 7: Filter and Sort Results
    filtered_results = [
        doc for doc in initial_results if doc["score"] >= score_threshold
    ]

    if not filtered_results:
        print(
            f"⚠️ All results were filtered out! Adjusting threshold to: {score_threshold}"
        )
        filtered_results = initial_results[
            :top_k
        ]  # Return unfiltered results if all were filtered

    ranked_results = sorted(filtered_results, key=lambda x: x["score"], reverse=True)

    return ranked_results[:top_k]


def test_drive():
    query = "Bluetooth headphones"
    results = hybrid_search_with_reranking(query, top_k=5, score_threshold=1.5)
    # results = hybrid_search(query, top_k=5)
    for res in results:
        print(f"Title: {res['title']}, Score: {res['score']}")


test_drive()
