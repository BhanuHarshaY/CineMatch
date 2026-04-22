"""
query_pipeline.py — Two-stage query pipeline for movie recommendations.

Stage 1: Intent Extraction (GPT-4o-mini)
  Raw query → OpenAI → structured intent + enriched_query

Stage 2: Retrieval + Re-ranking
  enriched_query → sentence-transformers → FAISS top-50 → GPT-4o-mini re-rank → top 10


"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
import numpy as np
from openai import OpenAI

# Load .env file so OPENAI_API_KEY is available via os.environ
load_dotenv()

# Config
DATA_DIR = Path("data")
EMBEDDING_MODEL = "all-mpnet-base-v2"
OPENAI_MODEL = "gpt-4o-mini"
FAISS_TOP_K = 50
FAISS_SCORE_THRESHOLDS = [0.30, 0.22, 0.15]  # Adaptive: try strict first, relax if too few results
FAISS_MIN_CANDIDATES = 8  # Minimum candidates before relaxing threshold
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

Below are {count} candidate titles with descriptions, ratings, and directors. \
Select the BEST matches for the user's query.

Candidates:
{candidates}

Return ONLY valid JSON with this structure:
{{
  "note": null,
  "results": [
    {{
      "index": 0,
      "title": "Movie Title",
      "reason": "One-line explanation of why this matches the query"
    }}
  ]
}}

Rules:
- "index" must be the candidate number from the list above (0-indexed)
- Order from best match (first) to weakest match (last)
- "reason" should be specific to the user's query, not generic praise
- Return between 5 and {top_k} results — ONLY include genuinely relevant matches
- If fewer than {top_k} strong matches exist, return fewer. Do NOT pad with \
weak or irrelevant results.
- DO NOT include any text outside the JSON object

Content-type filtering:
- Concert films, stand-up comedy specials, and documentaries are NOT matches for \
"comedy", "family comedy", or "family night" queries — exclude them
- Match the actual genre the user wants, not superficially related content types

Rating rules:
- Each candidate includes a content rating (G, PG, PG-13, R, TV-Y, TV-G, TV-PG, \
TV-14, TV-MA, etc.)
- If the query mentions "family", "kids", "children", "family night", or \
"family-friendly", ONLY include G, PG, TV-Y, TV-Y7, TV-G, and TV-PG rated content
- For family/kids queries, EXCLUDE R, TV-MA, TV-14, and PG-13 rated content entirely
- TV-14 or PG-13 may be included ONLY if fewer than 5 family-safe results exist, \
and must include explicit justification in the reason

Director/actor-specific queries:
- Each candidate includes its director. Use ONLY this metadata to verify claims.
- If the user asks for films by a specific director or actor, ONLY count candidates \
whose listed director/cast actually matches as "direct matches"
- Do NOT fabricate connections to a director or actor
- If fewer than 3 direct matches exist, set "note" to a user-friendly message like: \
"Only [N] [Director/Actor] titles found in this dataset. Showing similar films you \
might enjoy."
- For non-direct-match results, prefix the reason with "Fans of [Name] also enjoy: " \
or "Similar themes: " — never claim false attribution
- If 3 or more direct matches exist, set "note" to null
"""


# OpenAI client (initialized lazily)
_openai_client = None

def get_openai_client() -> OpenAI:
    """Get or create the OpenAI client. Reads OPENAI_API_KEY from env."""
    global _openai_client
    if _openai_client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable not set")
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


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
    Use OpenAI GPT-4o-mini to extract structured intent from a raw user query.

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
        client = get_openai_client()
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": INTENT_EXTRACTION_PROMPT.format(query=query)}],
            temperature=0,
            max_tokens=800,
        )

        parsed = extract_json_object(response.choices[0].message.content)
        if parsed is None:
            print(f"[intent] WARNING: Could not parse OpenAI response as JSON, using raw query")
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
        print(f"[intent] ERROR: OpenAI call failed ({e}), using raw query as fallback")
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

    Uses adaptive thresholds: starts strict (0.30), relaxes to 0.22 then 0.15
    if fewer than FAISS_MIN_CANDIDATES pass. This ensures mood/vibe queries
    get enough candidates for the re-ranker to work with.
    """
    if index is None or metadata is None:
        import faiss
        index = faiss.read_index(str(DATA_DIR / "faiss.index"))
        with open(DATA_DIR / "metadata.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)

    scores, indices = index.search(query_embedding, top_k)

    # Build full scored list once, then filter at each threshold
    all_candidates = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        item = metadata[idx].copy()
        item["faiss_score"] = float(score)
        all_candidates.append(item)

    # Try each threshold tier until we have enough candidates
    for threshold in FAISS_SCORE_THRESHOLDS:
        results = [c for c in all_candidates if c["faiss_score"] >= threshold]
        if len(results) >= FAISS_MIN_CANDIDATES:
            for i, r in enumerate(results):
                r["faiss_rank"] = i
            print(f"[faiss] Retrieved {len(results)} candidates at threshold {threshold} "
                  f"(top score: {results[0]['faiss_score']:.4f})")
            return results

    # Even the lowest threshold didn't yield enough — return whatever we have
    results = [c for c in all_candidates if c["faiss_score"] >= FAISS_SCORE_THRESHOLDS[-1]]
    for i, r in enumerate(results):
        r["faiss_rank"] = i
    if not results:
        print(f"[faiss] No candidates above minimum threshold {FAISS_SCORE_THRESHOLDS[-1]}")
        return []
    print(f"[faiss] Retrieved {len(results)} candidates at minimum threshold {FAISS_SCORE_THRESHOLDS[-1]} "
          f"(top score: {results[0]['faiss_score']:.4f})")
    return results


# Stage 2c: Re-rank with OpenAI
def rerank(query: str, candidates: list[dict], top_k: int = FINAL_TOP_K) -> dict:
    """
    Use OpenAI GPT-4o-mini to re-rank FAISS candidates based on the original query.

    Returns a dict with 'results' (list) and 'note' (str or None).
    The note explains limitations (e.g. few director matches in dataset).
    On failure, falls back to the FAISS-ranked top-k.
    """
    if not candidates:
        return {"results": [], "note": "No matching candidates found."}

    candidate_lines = []
    for i, c in enumerate(candidates):
        rating = c.get('rating', 'NR')
        director = c.get('director', 'Unknown')
        candidate_lines.append(
            f"{i}. [{c['type']}] {c['title']} ({c['release_year']}) "
            f"[Rated {rating}] [Dir: {director}] — {c['description']}"
        )
    candidates_text = "\n".join(candidate_lines)

    try:
        client = get_openai_client()
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": RERANK_PROMPT.format(
                query=query,
                count=len(candidates),
                top_k=top_k,
                candidates=candidates_text,
            )}],
            temperature=0.2,
            max_tokens=800,
        )

        # Parse response — expects {"note": ..., "results": [...]}
        raw_text = response.choices[0].message.content
        parsed = extract_json_object(raw_text)
        note = None
        ranked = None

        if parsed and "results" in parsed:
            ranked = parsed["results"]
            note = parsed.get("note")
        else:
            # Fallback: try parsing as a bare array (backward compat)
            ranked = extract_json_array(raw_text)

        if ranked is None:
            print(f"[rerank] WARNING: Could not parse OpenAI response, using FAISS order")
            return {
                "results": candidates[:top_k],
                "note": "AI re-ranking unavailable: showing semantic search results only.",
            }

        reranked = []
        for item in ranked:
            idx = item.get("index")
            if idx is not None and 0 <= idx < len(candidates):
                result = candidates[idx].copy()
                result["rerank_reason"] = item.get("reason", "")
                result["final_rank"] = len(reranked)
                reranked.append(result)

        # No padding — if OpenAI returned fewer, that's intentional quality filtering
        print(f"[rerank] OpenAI selected {len(reranked)} results"
              f"{f' (note: {note[:60]}...)' if note else ''}")
        return {"results": reranked[:top_k], "note": note}

    except Exception as e:
        print(f"[rerank] ERROR: OpenAI call failed ({e}), using FAISS order")
        fallback_note = (
            "AI re-ranking unavailable : showing semantic search results only. "
            "Add OPENAI_API_KEY to .env for full recommendations."
        )
        return {"results": candidates[:top_k], "note": fallback_note}


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
    rerank_result = rerank(query, candidates, FINAL_TOP_K)
    t_rerank = time.time() - t0

    t_total = time.time() - t_start

    return {
        "query": query,
        "intent": intent,
        "results": rerank_result["results"],
        "note": rerank_result.get("note"),
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
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not found.")
        print("  Either add it to .env file or: export OPENAI_API_KEY='your-key-here'")
        exit(1)

    test_queries = [
        "something dark and psychological like Inception",
        "Indian comedy films",
        "European drama with romance",
        "if I liked Parasite what should I watch next",
        "feel-good comedy for family night",
        "documentary about nature",
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
