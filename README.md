# CineMatch -  Movie Recommendation Engine

A content-based movie recommendation system that uses **two-stage retrieval** (FAISS dense retrieval + GPT-4o-mini re-ranking) to answer natural language queries like *"something dark and psychological like Inception"* against 8,807 Netflix titles.

![CineMatch Demo](demo_screenshot.png)

## Architecture

```mermaid
flowchart TD
    A[netflix_data.csv] --> B[build_index.py]
    B --> C[(FAISS Index 8807x768)]
    B --> D[(metadata.json)]

    E[User Query] --> F[GPT-4o-mini Intent Extraction]
    F --> G[Enriched Query]
    G --> H[sentence-transformers all-mpnet-base-v2]
    H --> I[FAISS Search top 50 candidates]
    C -.->|loaded at startup| I
    I --> J[Adaptive Threshold 0.30 to 0.22 to 0.15]
    J --> K[GPT-4o-mini Re-rank and Filter]
    D -.->|loaded at startup| K
    K --> L[5 to 10 Results with reasoning]
    L --> M[HTML Frontend]

```

## AI Setup

| Component | Provider | Model | Runs |
|-----------|----------|-------|------|
| Intent extraction | OpenAI | `gpt-4o-mini` | Per request (~2s) |
| Re-ranking | OpenAI | `gpt-4o-mini` | Per request (~5s) |
| Embeddings | Local | `all-mpnet-base-v2` (768d) | Build time + per request |
| Vector search | Local | FAISS `IndexFlatIP` | Per request (<1ms) |

**Reviewer requirement:** Set `OPENAI_API_KEY` at runtime. The system degrades gracefully without it (FAISS-only results + info banner), but recommendation quality depends on the LLM re-ranking stage.

## Approach

CineMatch uses a two-stage content-based retrieval pipeline. Rather than relying on a single similarity search, it separates retrieval (high recall) from re-ranking (high precision), allowing each stage to optimize for what it does best.

When a user submits a natural language query, GPT-4o-mini first extracts structured intent - genres, mood, themes, content type, era - and generates an enriched query. A raw query like "something cozy for a rainy Sunday" becomes a dense paragraph of semantic anchors ("warm, comforting films with gentle storytelling, themes of home and friendship..."), giving the embedding model 20+ meaningful terms instead of 5 vague words.

The enriched query is embedded using `all-mpnet-base-v2` (768-dimensional, cosine-trained) and searched against the FAISS `IndexFlatIP` index of 8,807 pre-computed title vectors. FAISS returns the top 50 candidates, which are then filtered through an adaptive threshold cascade: strict (0.30) for specific queries that produce high similarity scores, relaxing to 0.22 then 0.15 for abstract mood queries where no single title is a near-exact match. This ensures broad queries get enough candidates without polluting specific ones.

The surviving candidates are passed to GPT-4o-mini for re-ranking against the full query intent. The re-ranker applies constraints that embedding similarity alone cannot capture: rating filters for family queries, content-type exclusions (no stand-up specials for "comedy movie"), director accuracy checks, and thematic relevance scoring. It returns 5-10 results with per-item reasoning, dropping weak matches rather than padding to a fixed count.

Key design choices: duration is excluded from embedding text (it is a filter, not a semantic signal), FAISS similarity scores are passed to the re-ranker as additional context, and the FAISS index is built at Docker build time so container startup takes under 5 seconds instead of regenerating embeddings on every deploy.

## Setup

### Prerequisites

- Docker
- OpenAI API key ([platform.openai.com](https://platform.openai.com))

### Build & Run

```bash
# Build the image (~5 min: downloads model + generates FAISS index)
docker build -t cinematch .

# Run the container
docker run -p 8080:80 -e OPENAI_API_KEY=your-openai-key cinematch
```

Open [http://localhost:8080](http://localhost:8080)

### Without Docker (local development)

```bash
pip install -r requirements.txt
python scripts/build_index.py      # One-time: generates FAISS index
uvicorn src.main:app --host 0.0.0.0 --port 8080
```

> Note: `posters.json` is pre-committed to the repo. Running `scripts/fetch_posters.py` is only needed if you want to refresh poster images (requires `TMDB_API_KEY`).

## Demo

### Sample Queries

| Query | What it tests |
|-------|---------------|
| *"something dark and psychological like Inception"* | Semantic similarity + comparison |
| *"feel-good comedy for family night"* | Mood extraction + rating filtering |
| *"Korean drama with romance"* | Genre + region constraint |
| *"if I liked Parasite what should I watch next"* | Reference-based recommendation |
| *"movies directed by Christopher Nolan"* | Director lookup + honesty (limited data) |
| *"documentary about nature"* | Content-type filtering |

### Features



- **Transparency notes:** Info banner when few direct matches exist (e.g., director queries)
- **Graceful degradation:** Without API key, shows FAISS results with explanation banner



## Project Structure

```
├── src/
│   ├── main.py              # FastAPI server, lifespan resource loading
│   └── query_pipeline.py    # Two-stage pipeline: intent > embed > FAISS > rerank
├── scripts/
│   ├── build_index.py       # Offline: CSV > embeddings > FAISS index
│   └── fetch_posters.py     # Offline: TMDB API > poster URLs (optional, pre-committed)
├── frontend/
│   └── index.html           # Single-page UI with poster collage
├── data/
│   ├── netflix_data.csv     # Source dataset (8,807 titles)
│   ├── posters.json         # Pre-fetched TMDB poster URLs
│   ├── faiss.index          # Generated at build time
│   ├── metadata.json        # Generated at build time
│   └── embeddings.npy       # Generated at build time
├── test/
│   └── qa_test.py           # QA test suite (25 queries)
├── Dockerfile
├── requirements.txt
└── README.md
```