"""
main.py — FastAPI server for the movie recommender.

Loads FAISS index, metadata, and embedding model once at startup.
Serves the HTML frontend at GET / and the recommendation API at POST /recommend.
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from query_pipeline import (
    load_embedding_model,
    load_faiss_index,
    load_metadata,
    recommend,
)

load_dotenv()

# Shared state — loaded once at startup, used by every request
app_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load heavy resources once at startup, clean up on shutdown."""
    print("[server] Starting up...")

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set. Add it to .env or export it.")

    # Load resources into shared state (one-time cost: ~5s)
    app_state["model"] = load_embedding_model()
    app_state["index"] = load_faiss_index()
    app_state["metadata"] = load_metadata()

    print("[server] All resources loaded. Ready to serve requests.")
    yield

    # Cleanup on shutdown
    app_state.clear()
    print("[server] Shutdown complete.")


app = FastAPI(
    title="CineMatch",
    description="AI-powered movie recommendation engine",
    version="1.0.0",
    lifespan=lifespan,
)

# Serve static frontend files
FRONTEND_DIR = Path("frontend")
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


# Request/Response models
class RecommendRequest(BaseModel):
    query: str


class RecommendResponse(BaseModel):
    query: str
    intent: dict
    results: list
    note: Optional[str] = None
    timing: dict


# Routes
@app.get("/")
async def serve_frontend():
    """Serve the single-page HTML frontend."""
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return FileResponse(index_path)


@app.get("/api/posters")
async def get_posters():
    """Return cached poster URLs for the background collage."""
    poster_path = Path("data/posters.json")
    if not poster_path.exists():
        return []
    with open(poster_path, "r") as f:
        return json.load(f)


@app.post("/recommend", response_model=RecommendResponse)
async def get_recommendations(req: RecommendRequest):
    """
    Accept a natural language query and return ranked movie recommendations.

    The pipeline: intent extraction → embedding → FAISS search → re-ranking.
    Uses pre-loaded model, index, and metadata from server startup.
    """
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    if len(query) > 500:
        raise HTTPException(status_code=400, detail="Query too long (max 500 chars)")

    result = recommend(
        query,
        model=app_state["model"],
        index=app_state["index"],
        metadata=app_state["metadata"],
    )

    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=False)
