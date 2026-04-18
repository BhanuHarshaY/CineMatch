"""QA test suite — runs 25 queries and reports results."""
import json
import time
import requests

API = "http://localhost:8080/recommend"

QUERIES = [
    # Mood queries
    ("MOOD", "something cozy for a rainy Sunday"),
    ("MOOD", "happy feel good movies"),
    ("MOOD", "dark and disturbing but can't stop watching"),
    ("MOOD", "light and funny nothing serious"),
    ("MOOD", "movie for when you're sad and alone"),
    # Genre combos
    ("GENRE", "romantic comedy set in Europe"),
    ("GENRE", "sci-fi with strong female lead"),
    ("GENRE", "horror that relies on psychological tension not jump scares"),
    ("GENRE", "historical drama based on true events"),
    ("GENRE", "action movie with a great plot not just explosions"),
    # Comparison
    ("COMP", "something like Black Mirror but less dark"),
    ("COMP", "if I liked Parasite what should I watch next"),
    ("COMP", "movies similar to Squid Game"),
    # Constraint
    ("FILTER", "family friendly animated movie for a 6 year old"),
    ("FILTER", "Korean drama with romance"),
    ("FILTER", "Indian comedy movie"),
    ("FILTER", "documentary about nature"),
    # Actor/Director
    ("PERSON", "movies directed by Christopher Nolan"),
    ("PERSON", "films with Meryl Streep"),
    ("PERSON", "Adam Sandler comedy"),
    # Edge cases
    ("EDGE", ""),
    ("EDGE", "asdfghjkl"),
    ("EDGE", "best movie ever made"),
    ("EDGE", "something I haven't seen before"),
    ("EDGE", "films that changed cinema"),
]

FAMILY_KEYWORDS = ["family", "kid", "child", "6 year old", "animated"]
BAD_FAMILY_RATINGS = {"R", "TV-MA", "NR"}

results_table = []

for i, (cat, query) in enumerate(QUERIES):
    label = f"{i+1}. {query[:45]}" if query else f"{i+1}. (empty)"
    
    if not query.strip():
        # Empty query — should be rejected
        try:
            r = requests.post(API, json={"query": ""}, timeout=15)
            if r.status_code == 400:
                results_table.append((label, cat, "-", "-", "-", "✅ Rejected (400)"))
            else:
                data = r.json()
                results_table.append((label, cat, str(len(data.get("results", []))), "-", "-", "⚠ Not rejected"))
        except Exception as e:
            results_table.append((label, cat, "-", "-", "-", f"❌ {e}"))
        continue

    try:
        t0 = time.time()
        r = requests.post(API, json={"query": query}, timeout=30)
        wall = time.time() - t0
        
        if r.status_code != 200:
            results_table.append((label, cat, "-", "-", f"{wall:.1f}s", f"❌ HTTP {r.status_code}"))
            continue

        data = r.json()
        res = data.get("results", [])
        timing = data.get("timing", {})
        note = data.get("note")
        count = len(res)
        
        top3 = [f"{r['title']}" for r in res[:3]]
        top3_str = " | ".join(top3) if top3 else "(none)"
        
        t_str = f"{timing.get('total','?')}s (i:{timing.get('intent_extraction','?')} r:{timing.get('reranking','?')})"
        
        # Check for issues
        issues = []
        
        # API failure detection
        if note and "unavailable" in str(note).lower():
            issues.append("⚠ FAISS fallback")
        
        # Low count for broad queries
        if count < 5 and cat in ("MOOD", "GENRE", "COMP"):
            issues.append(f"⚠ Low count ({count})")
        
        # Family query with bad ratings
        is_family = any(kw in query.lower() for kw in FAMILY_KEYWORDS)
        if is_family:
            bad = [f"{r['title']}({r.get('rating','?')})" for r in res if r.get("rating") in BAD_FAMILY_RATINGS]
            if bad:
                issues.append(f"❌ Bad ratings: {', '.join(bad)}")
        
        # Duplicates
        titles = [r["title"] for r in res]
        if len(titles) != len(set(titles)):
            issues.append("❌ Duplicates")
        
        # No rerank reasons (FAISS fallback indicator)
        no_reason = sum(1 for r in res if not r.get("rerank_reason"))
        if no_reason > 0 and count > 0:
            issues.append(f"⚠ {no_reason}/{count} no reason")
        
        issue_str = " | ".join(issues) if issues else "✅"
        
        results_table.append((label, cat, str(count), top3_str, t_str, issue_str))
        
        # Rate limit protection — small delay between queries
        time.sleep(0.5)
        
    except Exception as e:
        results_table.append((label, cat, "-", "-", "-", f"❌ {e}"))

# Print results table
print(f"\n{'='*180}")
print(f"{'#':44} | {'Cat':6} | {'Cnt':3} | {'Top 3 Titles':65} | {'Timing':30} | Issues")
print(f"{'-'*44}-+-{'-'*6}-+-{'-'*3}-+-{'-'*65}-+-{'-'*30}-+-{'-'*20}")
for label, cat, count, top3, timing, issues in results_table:
    print(f"{label:44} | {cat:6} | {count:3} | {top3:65.65} | {timing:30} | {issues}")

# Summary stats
total = len(results_table)
ok = sum(1 for r in results_table if "✅" in r[5])
warn = sum(1 for r in results_table if "⚠" in r[5])
fail = sum(1 for r in results_table if "❌" in r[5])
print(f"\n{'='*80}")
print(f"SUMMARY: {total} queries | {ok} clean | {warn} warnings | {fail} failures")
