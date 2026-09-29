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

load_dotenv(find_dotenv())

SERPER_API_KEY = os.getenv("SERPER_API_KEY", "").strip()
SERPER_URL = "https://google.serper.dev/images"


def fetch_topic_images(topic_title: str, limit: int = 4) -> list:
    """Top `limit` image URLs for this topic, or [] on any failure/missing key."""
    if not SERPER_API_KEY:
        print("[TOPIC-IMAGES] [SKIPPED] SERPER_API_KEY is not set - no images will be fetched for any topic.")
        return []
    if not topic_title:
        print("[TOPIC-IMAGES] [SKIPPED] No topic title given - nothing to search for.")
        return []

    print(f"[TOPIC-IMAGES] Searching Serper for images on: '{topic_title}'...")
    try:
        resp = requests.post(
            SERPER_URL,
            headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
            json={"q": f"{topic_title} educational diagram", "num": limit, "autocorrect": True, "safe": "active"},
            timeout=10,
        )
        if resp.status_code == 429:
            print(f"[QUOTA LIMIT HIT] Serper key '...{SERPER_API_KEY[-6:]}' hit its rate/usage limit (429) - "
                  f"you need a fresh SERPER_API_KEY. Skipping images for '{topic_title}'.")
            return []
        if resp.status_code in (401, 403):
            print(f"[INVALID KEY] Serper key '...{SERPER_API_KEY[-6:]}' was rejected ({resp.status_code}) - "
                  f"it's likely invalid or expired. Replace SERPER_API_KEY in .env.")
            return []
        resp.raise_for_status()

        images = resp.json().get("images", [])
        urls = [img["imageUrl"] for img in images[:limit] if img.get("imageUrl")]
        if urls:
            print(f"[TOPIC-IMAGES] [OK] Found {len(urls)} image(s) for '{topic_title}'.")
        else:
            print(f"[TOPIC-IMAGES] [NOTICE] Serper returned no usable images for '{topic_title}'.")
        return urls
    except Exception as e:
        print(f"[TOPIC-IMAGES] [ERROR] Failed to fetch images for '{topic_title}': {e}")
        return []
