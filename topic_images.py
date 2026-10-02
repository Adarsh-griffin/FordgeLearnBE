"""
On-demand illustrative images for one AI Tutor lesson topic (the lesson
screen's content area) - a single Serper.dev Google Images search per
topic, hotlinked directly with no download step and no S3 re-hosting.

Deliberately NOT reusing image_finder.py's get_educational_images/
upload_image_to_s3 pipeline: that module imports `ratelimit` and `boto3`,
and `ratelimit` isn't even installed in this environment (confirmed:
`import ratelimit` fails) - it was already dead code, never wired into
any route. It also re-uploads every image to the same S3 bucket as
TTS audio/video, which conflicts with this app's storage rule ("only TTS
audio/video ever depend on S3" - everything else must keep working even
if the AWS account disappears). A raw Serper image URL is just as
external either way, but at least this adds zero NEW dependency on S3.

Caller (test_groq.py's get_or_fetch_topic_images) caches the result on
the document's topic_graph, so this only ever runs once per topic, no
matter how many students study the same document.
"""
import os
import requests
from dotenv import load_dotenv, find_dotenv
from topic_domains import preferred_domains, site_restricted_query

load_dotenv(find_dotenv())

SERPER_API_KEY = os.getenv("SERPER_API_KEY", "").strip()
SERPER_URL = "https://google.serper.dev/images"


def _is_image_reachable(url: str) -> bool:
    """Quick HEAD check (GET fallback, since some hosts reject HEAD) that a
    Serper result URL actually loads as an image - a meaningful fraction of
    raw search-result URLs are hotlink-protected or dead, which otherwise
    only surfaces as a permanently broken <img> on the frontend with no way
    to tell which slot will fail ahead of time."""
    headers = {"User-Agent": "Mozilla/5.0 (compatible; LearnForgeBot/1.0)"}
    try:
        resp = requests.head(url, headers=headers, timeout=4, allow_redirects=True)
        if resp.status_code < 400 and resp.headers.get("content-type", "").startswith("image/"):
            return True
        if resp.status_code in (405, 403):
            # Some hosts reject HEAD specifically but serve GET fine.
            resp = requests.get(url, headers=headers, timeout=4, stream=True)
            ok = resp.status_code < 400 and resp.headers.get("content-type", "").startswith("image/")
            resp.close()
            return ok
        return False
    except Exception:
        return False


def _search_image_candidates(query: str, num: int) -> list:
    """One Serper Images call -> list of raw image URLs, or [] on failure."""
    resp = requests.post(
        SERPER_URL,
        headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
        json={"q": query, "num": num, "autocorrect": True, "safe": "active"},
        timeout=10,
    )
    if resp.status_code == 429:
        print(f"[QUOTA LIMIT HIT] Serper key '...{SERPER_API_KEY[-6:]}' hit its rate/usage limit (429) - "
              f"you need a fresh SERPER_API_KEY. Skipping images for '{query}'.")
        return []
    if resp.status_code in (401, 403):
        print(f"[INVALID KEY] Serper key '...{SERPER_API_KEY[-6:]}' was rejected ({resp.status_code}) - "
              f"it's likely invalid or expired. Replace SERPER_API_KEY in .env.")
        return []
    if resp.status_code == 400:
        # Confirmed cause on a free-plan key: the "(site:a OR site:b)" query
        # pattern (fetch_topic_images' site-restricted search) is rejected
        # on Serper's Images endpoint specifically - "Query pattern not
        # allowed for free accounts". Not fatal: the caller falls back to a
        # plain, unrestricted query next.
        print(f"[TOPIC-IMAGES] [NOTICE] Serper rejected this image query (400: {resp.text[:150]}) for '{query}'.")
        return []
    resp.raise_for_status()
    return [img["imageUrl"] for img in resp.json().get("images", []) if img.get("imageUrl")]


def fetch_topic_images(topic_title: str, limit: int = 4) -> list:
    """Top `limit` VERIFIED-reachable image URLs for this topic, or [] on
    any failure/missing key. Prefers trusted subject-specific domains
    (GeeksforGeeks/W3Schools for programming/CS topics, BYJU'S/Vedantu/
    Toppr/Aakash for everything else), tops up with a plain unrestricted
    search if that doesn't return enough, and filters out dead/hotlink-
    protected results at every step so a broken thumbnail never gets
    cached."""
    if not SERPER_API_KEY:
        print("[TOPIC-IMAGES] [SKIPPED] SERPER_API_KEY is not set - no images will be fetched for any topic.")
        return []
    if not topic_title:
        print("[TOPIC-IMAGES] [SKIPPED] No topic title given - nothing to search for.")
        return []

    domains = preferred_domains(topic_title)
    print(f"[TOPIC-IMAGES] Searching Serper for images on: '{topic_title}' "
          f"(preferring {', '.join(domains)})...")
    try:
        over_fetch = max(limit * 3, limit + 6)
        query = f"{topic_title} educational diagram"

        # The site-restricted "(site:a OR site:b)" query is rejected outright
        # by Serper's Images endpoint on a free-plan key ("Query pattern not
        # allowed for free accounts", HTTP 400) even though the identical
        # pattern works fine on their /search (web) endpoint - confirmed via
        # a direct API call, which is why reference links worked but images
        # never did. Isolated in its own try/except so that rejection just
        # means "0 candidates from the restricted search" and execution
        # still falls through to the plain query below, instead of the
        # exception aborting fetch_topic_images entirely before the plain
        # (working) query ever runs.
        try:
            candidates = _search_image_candidates(site_restricted_query(query, domains), over_fetch)
        except Exception as e:
            print(f"[TOPIC-IMAGES] [NOTICE] Site-restricted image search failed ({e}) - "
                  f"falling back to a plain search for '{query}'.")
            candidates = []
        if len(candidates) < over_fetch:
            seen = set(candidates)
            for url in _search_image_candidates(query, over_fetch):
                if url not in seen:
                    candidates.append(url)
                    seen.add(url)

        urls = []
        for candidate in candidates:
            if _is_image_reachable(candidate):
                urls.append(candidate)
                if len(urls) >= limit:
                    break

        if urls:
            print(f"[TOPIC-IMAGES] [OK] Found {len(urls)}/{limit} verified image(s) for '{topic_title}' "
                  f"(checked {len(candidates)} candidate(s)).")
        else:
            print(f"[TOPIC-IMAGES] [NOTICE] No verified-reachable images for '{topic_title}' "
                  f"(checked {len(candidates)} candidate(s)).")
        return urls
    except Exception as e:
        print(f"[TOPIC-IMAGES] [ERROR] Failed to fetch images for '{topic_title}': {e}")
        return []
