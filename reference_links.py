"""
Real reference links for the Learning Hub's document summary - replaces a
hardcoded placeholder that was returned for every single document
regardless of content (confirmed: passmain_groq.py's process_file() never
parsed any links out of the LLM's response at all - it just hardcoded the
same "https://en.wikipedia.org/wiki/Special:Search?search=education" URL
for every chunk of every document, every time).

Same Serper.dev account as topic_images.py, but the general web-search
endpoint instead of the image-search one, since these need to be real,
clickable reference pages - not illustrations.
"""
import os
import requests
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

SERPER_API_KEY = os.getenv("SERPER_API_KEY", "").strip()
SERPER_SEARCH_URL = "https://google.serper.dev/search"


def fetch_reference_links(query: str, limit: int = 3) -> list:
    """Top `limit` real web results as {title, url, description}, or [] on any failure/missing key."""
    if not SERPER_API_KEY:
        print("[REFERENCE-LINKS] [SKIPPED] SERPER_API_KEY is not set - no reference links will be fetched.")
        return []
    if not query:
        print("[REFERENCE-LINKS] [SKIPPED] No search topic given - nothing to search for.")
        return []

    print(f"[REFERENCE-LINKS] Searching Serper for reference links on: '{query}'...")
    try:
        resp = requests.post(
            SERPER_SEARCH_URL,
            headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
            json={"q": query, "num": limit},
            timeout=10,
        )
        if resp.status_code == 429:
            print(f"[QUOTA LIMIT HIT] Serper key '...{SERPER_API_KEY[-6:]}' hit its rate/usage limit (429) - "
                  f"you need a fresh SERPER_API_KEY. Skipping reference links for '{query}'.")
            return []
        if resp.status_code in (401, 403):
            print(f"[INVALID KEY] Serper key '...{SERPER_API_KEY[-6:]}' was rejected ({resp.status_code}) - "
                  f"it's likely invalid or expired. Replace SERPER_API_KEY in .env.")
            return []
        resp.raise_for_status()

        results = resp.json().get("organic", [])
        links = [
            {
                "title": (r.get("title") or "").strip(),
                "url": r.get("link", ""),
                "description": (r.get("snippet") or "").strip(),
            }
            for r in results[:limit] if r.get("link")
        ]
        if links:
            print(f"[REFERENCE-LINKS] [OK] Found {len(links)} real reference link(s) for '{query}'.")
        else:
            print(f"[REFERENCE-LINKS] [NOTICE] Serper returned no usable links for '{query}'.")
        return links
    except Exception as e:
        print(f"[REFERENCE-LINKS] [ERROR] Failed to fetch reference links for '{query}': {e}")
        return []
