"""
query_pipeline.py — Two-stage query pipeline for movie recommendations.

Stage 1: Intent Extraction (Gemini 2.5 Flash)
  Raw query → Gemini → structured intent + enriched_query

Stage 2: Retrieval + Re-ranking
  enriched_query → sentence-transformers → FAISS top-50 → Gemini re-rank → top 10


"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
import numpy as np
from google import genai

# Load .env file so GOOGLE_API_KEY is available via os.environ
load_dotenv()

# Config
DATA_DIR = Path("data")
EMBEDDING_MODEL = "all-mpnet-base-v2"
GEMINI_MODEL = "gemini-2.5-flash"
FAISS_TOP_K = 50
FINAL_TOP_K = 10


# Prompts
INTENT_EXTRACTION_PROMPT = """\
You are a movie/TV show recommendation assistant. Analyze the user's query and \
extract their intent into structured JSON.

User query: "{query}"

Return ONLY valid JSON with this exact structure (no markdown, no explanation):
{{
  "genres": [],
  "mood": [],
  "themes": [],
  "tone": "",
  "era": null,
  "content_type": "movie" | "show" | "any",
  "enriched_query": ""
}}

Field definitions:
- genres: relevant Netflix genre tags (e.g. "Thrillers", "Comedies", "Documentaries")
- mood: emotional descriptors (e.g. "dark", "uplifting", "tense", "heartwarming")
- themes: story themes (e.g. "revenge", "coming-of-age", "survival", "family bonds")
- tone: overall feel ("light", "dark", "intense", "whimsical", "gritty")
- era: decade preference if mentioned, else null (e.g. "1980s", "2010s")
- content_type: "movie", "show", or "any" based on whether user specified
- enriched_query: a rich, natural-language re-statement of what the user wants, \
expanded with synonyms and related concepts. This will be used for semantic search, \
so make it descriptive and detailed. DO NOT include specific movie/show titles — \
only describe the qualities, themes, and mood the user is looking for.

Examples:
- "something like squid game" → enriched_query: "Intense survival competition \
drama with deadly stakes, social commentary on wealth inequality, psychological \
tension, and morally complex characters forced into desperate choices"
- "feel-good comedy for family night" → enriched_query: "Light-hearted family \
comedy with wholesome humor, positive messages, suitable for all ages, warm and \
uplifting tone with relatable family dynamics"
"""

RERANK_PROMPT = """\
You are a movie/TV show recommendation expert. The user asked:

"{query}"

Below are {count} candidate titles with descriptions. Select the {top_k} BEST \
matches for the user's query. Consider relevance to their mood, themes, genres, \
and specific preferences.

Candidates:
{candidates}

Return ONLY valid JSON — an array of exactly {top_k} objects:
[
  {{
    "index": 0,
    "title": "Movie Title",
    "reason": "One-line explanation of why this matches the query"
  }}
]

Rules:
- "index" must be the candidate number from the list above (0-indexed)
- Order from best match (first) to weakest match (last)
- "reason" should be specific to the user's query, not generic praise
- Return EXACTLY {top_k} results
- DO NOT include any text outside the JSON array
"""


# Gemini client (initialized lazily)
_gemini_client = None

def get_gemini_client() -> genai.Client:
    """Get or create the Gemini client. Reads GOOGLE_API_KEY from env."""
    global _gemini_client
    if _gemini_client is None:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY environment variable not set")
        _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client


# JSON extraction utilities
def extract_json_object(text: str) -> dict | None:
    """Extract the first JSON object from text, handling markdown fences."""
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*$", "", text.strip())

    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def extract_json_array(text: str) -> list | None:
    """Extract the first JSON array from text, handling markdown fences."""
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*$", "", text.strip())

    match = re.search(r"\[[\s\S]*\]", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


# Resource loading — called once at server startup
def load_embedding_model():
    """Load the sentence-transformer model into memory. Call once at startup."""
    from sentence_transformers import SentenceTransformer
    print(f"[startup] Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"[startup] Embedding model loaded")
    return model


def load_faiss_index():
    """Load the FAISS index from disk. Call once at startup."""
    import faiss
    index_path = str(DATA_DIR / "faiss.index")
    print(f"[startup] Loading FAISS index from {index_path}")
    index = faiss.read_index(index_path)
    print(f"[startup] FAISS index loaded: {index.ntotal} vectors")
    return index


def load_metadata() -> list[dict]:
    """Load movie metadata from disk. Call once at startup."""
    meta_path = DATA_DIR / "metadata.json"
    print(f"[startup] Loading metadata from {meta_path}")
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    print(f"[startup] Metadata loaded: {len(metadata)} entries")
    return metadata


# Stage 1: Intent Extraction
def extract_intent(query: str) -> dict:
    """
    Use Gemini to extract structured intent from a raw user query.

    Returns a dict with genres, mood, themes, tone, era, content_type,
    and enriched_query. On any failure, falls back to using the raw query
    as the enriched_query — the pipeline degrades gracefully, never crashes.
    """
    default_intent = {
        "genres": [],
        "mood": [],
        "themes": [],
        "tone": "",
        "era": None,
        "content_type": "any",
        "enriched_query": query,
    }

    try:
        client = get_gemini_client()
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=INTENT_EXTRACTION_PROMPT.format(query=query),
            config=genai.types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=1024,
                thinking_config=genai.types.ThinkingConfig(thinking_budget=0),
            ),
        )

        parsed = extract_json_object(response.text)
        if parsed is None:
            print(f"[intent] WARNING: Could not parse Gemini response as JSON, using raw query")
            return default_intent

        for key in default_intent:
            if key not in parsed:
                parsed[key] = default_intent[key]

        if not parsed.get("enriched_query", "").strip():
            parsed["enriched_query"] = query

        print(f"[intent] Extracted: content_type={parsed['content_type']}, "
              f"genres={parsed['genres']}, mood={parsed['mood']}")
        print(f"[intent] Enriched query: {parsed['enriched_query'][:120]}...")
        return parsed

    except Exception as e:
        print(f"[intent] ERROR: Gemini call failed ({e}), using raw query as fallback")
        return default_intent


# Stage 2a: Embed Query
def embed_query(enriched_query: str, model=None) -> np.ndarray:
    """
    Embed the enriched query using the same model that built the index.

    If model is provided (pre-loaded at server startup), uses it directly.
    Otherwise loads the model fresh (for CLI testing).
    """
    if model is None:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(EMBEDDING_MODEL)

    embedding = model.encode(
        enriched_query,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embedding.reshape(1, -1).astype("float32")


# Stage 2b: FAISS Search
def faiss_search(query_embedding: np.ndarray, index=None, metadata=None,
                 top_k: int = FAISS_TOP_K) -> list[dict]:
    """
    Search the FAISS index for the top-k most similar items.

    If index/metadata are provided (pre-loaded at server startup), uses them
    directly. Otherwise loads from disk (for CLI testing).
    """
    if index is None or metadata is None:
        import faiss
        index = faiss.read_index(str(DATA_DIR / "faiss.index"))
        with open(DATA_DIR / "metadata.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)

    scores, indices = index.search(query_embedding, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        item = metadata[idx].copy()
        item["faiss_score"] = float(score)
        item["faiss_rank"] = len(results)
        results.append(item)

    print(f"[faiss] Retrieved {len(results)} candidates (top score: {results[0]['faiss_score']:.4f})")
    return results


# Stage 2c: Re-rank with Gemini
def rerank(query: str, candidates: list[dict], top_k: int = FINAL_TOP_K) -> list[dict]:
    """
    Use Gemini to re-rank FAISS candidates based on the original query.

    On failure, falls back to the FAISS-ranked top-k.
    """
    candidate_lines = []
    for i, c in enumerate(candidates):
        candidate_lines.append(
            f"{i}. [{c['type']}] {c['title']} ({c['release_year']}) — {c['description']}"
        )
    candidates_text = "\n".join(candidate_lines)

    try:
        client = get_gemini_client()
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=RERANK_PROMPT.format(
                query=query,
                count=len(candidates),
                top_k=top_k,
                candidates=candidates_text,
            ),
            config=genai.types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=4096,
                thinking_config=genai.types.ThinkingConfig(thinking_budget=0),
            ),
        )

        ranked = extract_json_array(response.text)
        if ranked is None:
            print(f"[rerank] WARNING: Could not parse Gemini response, using FAISS order")
            return candidates[:top_k]

        reranked = []
        for item in ranked:
            idx = item.get("index")
            if idx is not None and 0 <= idx < len(candidates):
                result = candidates[idx].copy()
                result["rerank_reason"] = item.get("reason", "")
                result["final_rank"] = len(reranked)
                reranked.append(result)

        if len(reranked) < top_k:
            seen_ids = {r["show_id"] for r in reranked}
            for c in candidates:
                if len(reranked) >= top_k:
                    break
                if c["show_id"] not in seen_ids:
                    c_copy = c.copy()
                    c_copy["rerank_reason"] = "Added from FAISS ranking"
                    c_copy["final_rank"] = len(reranked)
                    reranked.append(c_copy)

        print(f"[rerank] Gemini selected {len(reranked)} results")
        return reranked[:top_k]

    except Exception as e:
        print(f"[rerank] ERROR: Gemini call failed ({e}), using FAISS order")
        return candidates[:top_k]


# Full Pipeline
def recommend(query: str, model=None, index=None, metadata=None) -> dict:
    """
    Full recommendation pipeline: intent → embed → search → re-rank.

    Accepts pre-loaded resources (model, index, metadata) for server use.
    Falls back to loading from disk if not provided (CLI use).
    """
    t_start = time.time()

    t0 = time.time()
    intent = extract_intent(query)
    t_intent = time.time() - t0

    t0 = time.time()
    query_embedding = embed_query(intent["enriched_query"], model=model)
    t_embed = time.time() - t0

    t0 = time.time()
    candidates = faiss_search(query_embedding, index=index, metadata=metadata,
                              top_k=FAISS_TOP_K)
    t_faiss = time.time() - t0

    t0 = time.time()
    results = rerank(query, candidates, FINAL_TOP_K)
    t_rerank = time.time() - t0

    t_total = time.time() - t_start

    return {
        "query": query,
        "intent": intent,
        "results": results,
        "timing": {
            "intent_extraction": round(t_intent, 2),
            "embedding": round(t_embed, 2),
            "faiss_search": round(t_faiss, 2),
            "reranking": round(t_rerank, 2),
            "total": round(t_total, 2),
        },
    }


# CLI for testing
if __name__ == "__main__":
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY not found.")
        print("  Either add it to .env file or: export GOOGLE_API_KEY='your-key-here'")
        exit(1)

    test_queries = [
        "something dark and psychological like inception",
        "feel-good comedy for family night",
        "korean drama with romance",
    ]

    for q in test_queries:
        print(f"\n{'='*70}")
        print(f"Query: {q}")
        print(f"{'='*70}")
        result = recommend(q)

        print(f"\nTiming: {result['timing']}")
        print(f"\nTop {len(result['results'])} Results:")
        for i, r in enumerate(result["results"]):
            reason = r.get("rerank_reason", "")
            print(f"  {i+1}. [{r['type']}] {r['title']} ({r['release_year']}) "
                  f"— score: {r.get('faiss_score', 0):.4f}")
            if reason:
                print(f"     → {reason}")

        print("\n(Skipping remaining test queries to avoid rate limits)")
        break
