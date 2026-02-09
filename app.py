from fastapi import FastAPI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import OrderedDict
import hashlib
import time

app = FastAPI()

# ---------------------------
# Configuration
# ---------------------------
TTL_SECONDS = 24 * 60 * 60
MAX_CACHE_SIZE = 1500
MODEL_COST_PER_1M = 1.0
AVG_TOKENS = 3000

# ---------------------------
# Cache + analytics
# ---------------------------
cache = OrderedDict()
total_requests = 0
cache_hits = 0
cache_misses = 0
cached_tokens = 0

# ---------------------------
# Embedding setup (offline)
# ---------------------------
vectorizer = TfidfVectorizer(stop_words="english")
embedding_texts = []

def fake_llm_response(query: str) -> str:
    return f"Summary for: {query}"

def now():
    return time.time()

def hash_key(query: str):
    return hashlib.md5(query.encode()).hexdigest()

def cleanup_cache():
    while len(cache) > MAX_CACHE_SIZE:
        cache.popitem(last=False)

def expire_entries():
    expired = [
        k for k, v in cache.items()
        if now() - v["timestamp"] > TTL_SECONDS
    ]
    for k in expired:
        del cache[k]

# ---------------------------
# Main endpoint
# ---------------------------
@app.post("/")
async def query_endpoint(payload: dict):
    global total_requests, cache_hits, cache_misses, cached_tokens

    start = time.time()
    total_requests += 1
    query = payload["query"]

    expire_entries()
    key = hash_key(query)

    # ---------- Exact match ----------
    if key in cache:
        cache_hits += 1
        cache.move_to_end(key)
        cached_tokens += AVG_TOKENS
        return {
            "answer": cache[key]["answer"],
            "cached": True,
            "latency": int((time.time() - start) * 1000),
            "cacheKey": key
        }

    # ---------- Semantic cache ----------
    if embedding_texts:
        q_vec = vectorizer.transform([query])
        sims = cosine_similarity(q_vec, vectorizer.transform(embedding_texts))[0]
        best_idx = sims.argmax()
        if sims[best_idx] > 0.95:
            semantic_key = list(cache.keys())[best_idx]
            cache_hits += 1
            cached_tokens += AVG_TOKENS
            return {
                "answer": cache[semantic_key]["answer"],
                "cached": True,
                "latency": int((time.time() - start) * 1000),
                "cacheKey": semantic_key
            }

    # ---------- Cache miss ----------
    cache_misses += 1
    answer = fake_llm_response(query)

    cache[key] = {
        "answer": answer,
        "timestamp": now()
    }
    embedding_texts.append(query)
    vectorizer.fit(embedding_texts)

    cleanup_cache()

    return {
        "answer": answer,
        "cached": False,
        "latency": int((time.time() - start) * 1000),
        "cacheKey": key
    }

# ---------------------------
# Analytics endpoint
# ---------------------------
@app.get("/analytics")
async def analytics():
    hit_rate = (cache_hits / total_requests) if total_requests else 0
    savings_tokens = cached_tokens
    savings = (savings_tokens / 1_000_000) * MODEL_COST_PER_1M

    return {
        "hitRate": round(hit_rate, 2),
        "totalRequests": total_requests,
        "cacheHits": cache_hits,
        "cacheMisses": cache_misses,
        "cacheSize": len(cache),
        "costSavings": round(savings, 2),
        "savingsPercent": round(hit_rate * 100),
        "strategies": [
            "exact match",
            "semantic similarity",
            "LRU eviction",
            "TTL expiration"
        ]
    }
