from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import OrderedDict
import hashlib
import time

app = FastAPI()

# ---------------------------
# Enable CORS (IMPORTANT)
# ---------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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
# Embedding setup
# ---------------------------
vectorizer = TfidfVectorizer(stop_words="english")

def fake_llm_response(query: str) -> str:
    return f"Summary for: {query}"

def now():
    return time.time()

def hash_key(query: str):
    return hashlib.md5(query.encode()).hexdigest()

def expire_entries():
    expired = [
        k for k, v in cache.items()
        if now() - v["timestamp"] > TTL_SECONDS
    ]
    for k in expired:
        del cache[k]

def cleanup_cache():
    while len(cache) > MAX_CACHE_SIZE:
        cache.popitem(last=False)

# ---------------------------
# Request model (IMPORTANT)
# ---------------------------
class QueryRequest(BaseModel):
    query: str
    application: str

# ---------------------------
# Main endpoint
# ---------------------------
@app.post("/")
async def query_endpoint(req: QueryRequest):
    global total_requests, cache_hits, cache_misses, cached_tokens

    start = time.time()
    total_requests += 1
    query = req.query
    key = hash_key(query)

    expire_entries()

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
    if cache:
        texts = [v["query"] for v in cache.values()]
        vectorizer.fit(texts)
        q_vec = vectorizer.transform([query])
        sims = cosine_similarity(q_vec, vectorizer.transform(texts))[0]
        best_idx = sims.argmax()

        if sims[best_idx] > 0.95:
            semantic_key = list(cache.keys())[best_idx]
            cache_hits += 1
            cached_tokens += AVG_TOKENS
            cache.move_to_end(semantic_key)
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
        "query": query,
        "answer": answer,
        "timestamp": now()
    }

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
    savings = (cached_tokens / 1_000_000) * MODEL_COST_PER_1M

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
