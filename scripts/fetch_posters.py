"""
fetch_posters.py — One-time script to fetch movie poster URLs from TMDB.

Reads metadata.json, searches TMDB for each title, and saves
poster URLs to data/posters.json. Run once before deployment.
"""

import json
import os
import time
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

TMDB_API_KEY = os.environ.get("TMDB_API_KEY")
if not TMDB_API_KEY:
    raise RuntimeError("TMDB_API_KEY not set in .env")

TMDB_SEARCH_URL = "https://api.themoviedb.org/3/search/movie"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w342"  # w342 = good quality, reasonable size
DATA_DIR = Path("data")
TARGET_COUNT = 50

from typing import Optional

def fetch_poster(title: str, year: str) -> Optional[str]:
    """Search TMDB for a movie and return its poster URL."""
    try:
        params = {
            "api_key": TMDB_API_KEY,
            "query": title,
            "year": year,
        }
        r = requests.get(TMDB_SEARCH_URL, params=params, timeout=5)
        r.raise_for_status()
        results = r.json().get("results", [])

        if not results:
            # Retry without year (some titles have mismatched years)
            params.pop("year")
            r = requests.get(TMDB_SEARCH_URL, params=params, timeout=5)
            r.raise_for_status()
            results = r.json().get("results", [])

        if results and results[0].get("poster_path"):
            return TMDB_IMAGE_BASE + results[0]["poster_path"]
    except Exception as e:
        print(f"  ERROR fetching '{title}': {e}")
    return None


def main():
    # Load metadata
    with open(DATA_DIR / "metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Filter to movies only (better poster coverage than TV shows)
    movies = [m for m in metadata if m.get("type") == "Movie"]
    print(f"[posters] {len(movies)} movies in dataset, fetching top {TARGET_COUNT} posters")

    # Pick well-known titles first (sort by year descending for recognizable posters)
    movies.sort(key=lambda m: int(m.get("release_year", 0)), reverse=True)

    posters = []
    seen_titles = set()

    for m in movies:
        if len(posters) >= TARGET_COUNT:
            break

        title = m["title"]
        if title in seen_titles:
            continue
        seen_titles.add(title)

        print(f"  [{len(posters)+1}/{TARGET_COUNT}] {title} ({m['release_year']})...", end=" ")
        url = fetch_poster(title, m.get("release_year", ""))

        if url:
            posters.append({"title": title, "url": url})
            print(f"✓")
        else:
            print(f"✗ (no poster)")

        # Rate limit: TMDB free tier allows ~40 req/10s
        time.sleep(0.25)

    # Save results
    output_path = DATA_DIR / "posters.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(posters, f, indent=2)

    print(f"\n[posters] Saved {len(posters)} posters to {output_path}")


if __name__ == "__main__":
    main()
