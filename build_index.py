"""
build_index.py — Run once at Docker build time.

Generates:
  data/embeddings.npy  — 8,807 × 768 L2-normalized float32 matrix
  data/faiss.index     — IndexFlatIP for cosine similarity search
  data/metadata.json   — row metadata for API responses

Key decisions:
  - IndexFlatIP on L2-normalized vectors = cosine similarity
  - Description placed last in text template for maximum embedding weight
  - faiss imported after sentence-transformers finishes encoding
    (avoids OpenMP conflict on Apple Silicon — harmless on Linux)
"""

import csv
import json
import os
import time
from pathlib import Path

import numpy as np


# Config

CSV_PATH = "netflix_data.csv"
OUTPUT_DIR = Path("data")
EMBEDDING_MODEL = "all-mpnet-base-v2"  # 768-dim, best quality/speed tradeoff
BATCH_SIZE = 64
# Rating values that are actually duration data (shifted columns in CSV)
INVALID_RATINGS = {"74 min", "84 min", "66 min"}



# Step 1: Load and clean data

def load_and_clean(csv_path: str) -> list[dict]:
    """Load netflix_data.csv and fix known data quality issues."""
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Fix: 3 rows have duration values leaked into the rating column.
            # We clear the invalid rating rather than guessing the correct one.
            if row["rating"].strip() in INVALID_RATINGS:
                row["rating"] = ""

            rows.append(row)

    print(f"[load] {len(rows)} rows loaded, {sum(1 for r in rows if not r['rating'].strip())} with missing/cleaned ratings")
    return rows



# Step 2: Build rich text representation

def build_text(row: dict) -> str:
    """
    Combine structured metadata + free-text description into a single string
    optimized for semantic embedding.

    Template structure:
      {type}: {title}. [Directed by {director}.] Genres: {listed_in}.
      [Rated {rating}.] [Released in {year}.] [From {country}.]
      [Starring {cast}.] {description}

    - Fields with missing values are silently omitted (no "Unknown" placeholders
      that would pollute the embedding space with false similarity).
    - Description goes last so the model's attention gives it the strongest
      weight — it carries the richest semantic signal.
    """
    parts = [f"{row['type']}: {row['title']}."]

    director = row.get("director", "").strip()
    if director:
        parts.append(f"Directed by {director}.")

    parts.append(f"Genres: {row['listed_in']}.")

    rating = row.get("rating", "").strip()
    if rating:
        parts.append(f"Rated {rating}.")

    release_year = row.get("release_year", "").strip()
    if release_year:
        parts.append(f"Released in {release_year}.")

    country = row.get("country", "").strip()
    if country:
        parts.append(f"From {country}.")

    cast = row.get("cast", "").strip()
    if cast:
        parts.append(f"Starring {cast}.")

    parts.append(row["description"])

    return " ".join(parts)



# Step 3: Generate embeddings

def generate_embeddings(texts: list[str], model_name: str, batch_size: int) -> np.ndarray:
    """
    Encode texts into normalized embeddings using sentence-transformers.

    Returns an (N x dim) float32 matrix where each row is L2-normalized.
    L2 normalization is critical: it makes the FAISS inner product operation
    equivalent to cosine similarity.

    Imports sentence_transformers here (not at top level) to avoid OpenMP
    conflicts with faiss-cpu on macOS/Apple Silicon.
    """
    # Lazy import — keeps PyTorch's OpenMP isolated from faiss's OpenMP
    from sentence_transformers import SentenceTransformer

    print(f"[embed] Loading model: {model_name}")
    model = SentenceTransformer(model_name)

    print(f"[embed] Encoding {len(texts)} texts (batch_size={batch_size})...")
    t0 = time.time()

    # normalize_embeddings=True applies L2 normalization inside the model,
    # which is more numerically stable than normalizing after the fact.
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # Critical for IndexFlatIP = cosine sim
        convert_to_numpy=True,
    )

    elapsed = time.time() - t0
    print(f"[embed] Done in {elapsed:.1f}s — shape: {embeddings.shape}, dtype: {embeddings.dtype}")
    return embeddings



# Step 4: Build FAISS index

def build_faiss_index(embeddings: np.ndarray):
    """
    Build a flat (exact search) FAISS index using inner product.

    Why IndexFlatIP over IndexFlatL2:
      - Our embeddings are L2-normalized, so inner product = cosine similarity.
      - IP scores range from 0 to 1 (higher = more similar), which is intuitive.
      - L2 would give inverted scores (lower = more similar) — harder to reason about.

    Why flat (exact) over approximate (IVF, HNSW):
      - 8,807 vectors x 768 dims = ~27MB. Exact search over this takes <1ms.
      - Approximate indices add complexity and recall loss for zero speed gain
        at this scale. They only matter at 100K+ vectors.

    Imports faiss here (not at top level) to avoid OpenMP conflicts with
    PyTorch on macOS/Apple Silicon.
    """
    # Lazy import — faiss loads AFTER sentence-transformers is done,
    # avoiding the dual-OpenMP segfault on Apple Silicon.
    import faiss

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"[faiss] Built IndexFlatIP with {index.ntotal} vectors, dim={dim}")
    return index



# Step 5: Build metadata for serving

def build_metadata(rows: list[dict]) -> list[dict]:
    """
    Extract the fields the API needs to return in search results.

    We store this separately from the FAISS index because FAISS only stores
    vectors — it returns integer IDs that we map back to metadata.
    """
    metadata = []
    for row in rows:
        metadata.append({
            "show_id": row["show_id"],
            "type": row["type"],
            "title": row["title"],
            "director": row.get("director", "").strip(),
            "cast": row.get("cast", "").strip(),
            "country": row.get("country", "").strip(),
            "release_year": row.get("release_year", "").strip(),
            "rating": row.get("rating", "").strip(),
            "duration": row.get("duration", "").strip(),
            "listed_in": row["listed_in"],
            "description": row["description"],
        })
    return metadata



# Step 6: Save FAISS index to disk

def save_faiss_index(index, idx_path: Path):
    """Save FAISS index using faiss's own serialization (lazy import)."""
    import faiss
    faiss.write_index(index, str(idx_path))



# Main

def main():
    t_start = time.time()

    # 1. Load and clean
    rows = load_and_clean(CSV_PATH)

    # 2. Build text representations
    print(f"[text] Building rich text for {len(rows)} rows...")
    texts = [build_text(row) for row in rows]

    # Sanity check: print a sample
    print(f"\n[text] Sample (row 0):\n  {texts[0][:200]}...\n")

    # 3. Generate embeddings (imports sentence_transformers internally)
    embeddings = generate_embeddings(texts, EMBEDDING_MODEL, BATCH_SIZE)

    # 4. Build FAISS index (imports faiss internally — AFTER encoding is done)
    index = build_faiss_index(embeddings)

    # 5. Build metadata
    metadata = build_metadata(rows)

    # 6. Save everything to disk
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    emb_path = OUTPUT_DIR / "embeddings.npy"
    np.save(emb_path, embeddings)
    print(f"[save] Embeddings → {emb_path} ({os.path.getsize(emb_path) / 1e6:.1f} MB)")

    idx_path = OUTPUT_DIR / "faiss.index"
    save_faiss_index(index, idx_path)
    print(f"[save] FAISS index → {idx_path} ({os.path.getsize(idx_path) / 1e6:.1f} MB)")

    meta_path = OUTPUT_DIR / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False)
    print(f"[save] Metadata → {meta_path} ({os.path.getsize(meta_path) / 1e6:.1f} MB)")

    elapsed = time.time() - t_start
    print(f"\n✅ Pipeline complete in {elapsed:.1f}s")
    print(f"   {len(rows)} rows → {embeddings.shape[1]}-dim embeddings → FAISS IndexFlatIP")


if __name__ == "__main__":
    main()
