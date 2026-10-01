"""
Utility helpers for connecting the Tavus lipsync API with NeuroLearn.

Features:
1. Upload learning TTS audio files to AWS S3 and derive their public URLs.
2. Look up the latest audio file in an S3 prefix (per student/document folder).
3. Use Tavus to generate a lipsync video for the most recent audio and store it
   under `collections/<folder>/videos` so the frontend can stream it.

This module can be used as a standalone script (`python lipsync.py`) or
imported inside the Flask backend (`test_groq.py`) to expose API endpoints.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import requests
from dotenv import load_dotenv

try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError
except ImportError:  # pragma: no cover - handled at runtime
    boto3 = None  # type: ignore
    BotoCoreError = ClientError = Exception  # type: ignore

load_dotenv()

# ---------------------------------------------------------------------------
# Tavus configuration
# ---------------------------------------------------------------------------
TAVUS_API_KEYS_STR = os.getenv("TAVUS_API_KEY", "")
TAVUS_API_KEYS = [k.strip() for k in TAVUS_API_KEYS_STR.split(',') if k.strip()]
REPLICA_ID = os.getenv("REPLICA_ID")

_current_tavus_key_index = 0

def get_current_tavus_key():
    global _current_tavus_key_index
    if not TAVUS_API_KEYS:
        return None
    return TAVUS_API_KEYS[_current_tavus_key_index]

def rotate_tavus_key():
    global _current_tavus_key_index
    if not TAVUS_API_KEYS:
        return
    _current_tavus_key_index = (_current_tavus_key_index + 1) % len(TAVUS_API_KEYS)
    print(f"Rotating to Tavus API key index: {_current_tavus_key_index}")

def execute_tavus_with_retry(func, *args, **kwargs):
    """
    Execute a function that uses the Tavus API key.
    If it fails with a 4xx/5xx error (likely auth or rate limit), rotate the key and retry.
    """
    if not TAVUS_API_KEYS:
         # Fallback if no keys configured, though _require_tavus_credentials checks this
         return func(None, *args, **kwargs)

    max_retries = len(TAVUS_API_KEYS)
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            api_key = get_current_tavus_key()
            return func(api_key, *args, **kwargs)
        except Exception as e:
            # Do not retry on timeouts (polling exceeded) or missing resources (404)
            if isinstance(e, (TimeoutError, FileNotFoundError)):
                raise e
            
            print(f"Tavus attempt {attempt + 1} failed with key index {_current_tavus_key_index}: {e}")
            last_exception = e
            rotate_tavus_key()
    
    raise last_exception

TAVUS_BASE = "https://tavusapi.com"
VIDEOS_CREATE = f"{TAVUS_BASE}/v2/videos"
VIDEOS_GET = lambda vid: f"{TAVUS_BASE}/v2/videos/{vid}"
DEFAULT_AUDIO_URL = os.getenv("AUDIO_PUBLIC_URL")
DEFAULT_VIDEO_NAME = os.getenv("DEFAULT_LIPSYNC_VIDEO_NAME", "tavus_video.mp4")

# ---------------------------------------------------------------------------
# AWS / storage configuration
# ---------------------------------------------------------------------------
COLLECTIONS_ROOT = Path(os.getenv("COLLECTIONS_ROOT", "collections")).resolve()
COLLECTIONS_ROOT.mkdir(parents=True, exist_ok=True)

AWS_AUDIO_BUCKET = os.getenv("AWS_AUDIO_BUCKET") or os.getenv("AWS_S3_BUCKET")
AWS_AUDIO_PREFIX = os.getenv("AWS_AUDIO_PREFIX", "learning-tts")
AWS_VIDEO_PREFIX = os.getenv("AWS_VIDEO_PREFIX", "learning-videos")
AWS_REGION = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"
AWS_AUDIO_PUBLIC_BASE_URL = os.getenv("AWS_AUDIO_PUBLIC_BASE_URL")
AWS_AUDIO_UPLOAD_ACL = os.getenv("AWS_AUDIO_UPLOAD_ACL", "public-read")

_s3_client = None


# ---------------------------------------------------------------------------
# AWS helpers
# ---------------------------------------------------------------------------
def get_s3_client():
    """Return a cached boto3 client or raise if configuration is missing."""
    global _s3_client
    if _s3_client is None:
        if boto3 is None:
            raise RuntimeError("boto3 is required for AWS interactions. Install boto3 to continue.")
        if not AWS_AUDIO_BUCKET:
            raise RuntimeError("AWS_AUDIO_BUCKET (or AWS_S3_BUCKET) is not configured.")
        _s3_client = boto3.client("s3", region_name=AWS_REGION)
    return _s3_client


def build_audio_prefix(folder: Optional[str] = None) -> str:
    segments = [AWS_AUDIO_PREFIX.strip("/")] if AWS_AUDIO_PREFIX else []
    if folder:
        segments.append(folder.strip("/"))
    return "/".join(filter(None, segments))


def build_audio_key(filename: str, folder: Optional[str] = None) -> str:
    prefix = build_audio_prefix(folder)
    return "/".join(filter(None, [prefix, filename]))


def build_audio_public_url(key: str) -> str:
    if not AWS_AUDIO_BUCKET:
        raise RuntimeError("AWS_AUDIO_BUCKET is not set; cannot build public URL.")
    if AWS_AUDIO_PUBLIC_BASE_URL:
        base = AWS_AUDIO_PUBLIC_BASE_URL.rstrip("/")
        return f"{base}/{key.lstrip('/')}"
    region = AWS_REGION or "us-east-1"
    if region == "us-east-1":
        return f"https://{AWS_AUDIO_BUCKET}.s3.amazonaws.com/{key}"
    return f"https://{AWS_AUDIO_BUCKET}.s3.{region}.amazonaws.com/{key}"


def upload_audio_to_s3(
    audio_bytes: bytes,
    filename: str,
    folder: Optional[str] = None,
    content_type: str = "audio/wav",
) -> Dict[str, str]:
    """Upload audio bytes to S3 and return the object metadata."""
    client = get_s3_client()
    key = build_audio_key(filename, folder)
    put_kwargs = {
        "Bucket": AWS_AUDIO_BUCKET,
        "Key": key,
        "Body": audio_bytes,
        "ContentType": content_type,
    }
    if AWS_AUDIO_UPLOAD_ACL:
        put_kwargs["ACL"] = AWS_AUDIO_UPLOAD_ACL
    try:
        client.put_object(**put_kwargs)
    except (BotoCoreError, ClientError) as exc:  # pragma: no cover - network call
        raise RuntimeError(f"Failed to upload audio to S3: {exc}") from exc

    return {"key": key, "url": build_audio_public_url(key)}


def build_video_key(filename: str, folder: Optional[str] = None) -> str:
    segments = [AWS_VIDEO_PREFIX.strip("/")] if AWS_VIDEO_PREFIX else []
    if folder:
        segments.append(folder.strip("/"))
    return "/".join(filter(None, segments + [filename]))


def upload_video_to_s3(
    video_path: Path,
    folder: Optional[str] = None,
    content_type: str = "video/mp4",
) -> Dict[str, str]:
    """Upload video file to S3 and return the object metadata."""
    client = get_s3_client()
    filename = video_path.name
    key = build_video_key(filename, folder)
    
    # Read file content
    with open(video_path, "rb") as f:
        video_bytes = f.read()

    put_kwargs = {
        "Bucket": AWS_AUDIO_BUCKET,  # Reusing the same bucket
        "Key": key,
        "Body": video_bytes,
        "ContentType": content_type,
    }
    if AWS_AUDIO_UPLOAD_ACL:
        put_kwargs["ACL"] = AWS_AUDIO_UPLOAD_ACL
    try:
        client.put_object(**put_kwargs)
    except (BotoCoreError, ClientError) as exc:
        raise RuntimeError(f"Failed to upload video to S3: {exc}") from exc
    
    # Reuse build_audio_public_url logic as it just constructs URL based on key and bucket
    return {"key": key, "url": build_audio_public_url(key)}

def get_latest_audio_from_s3(folder: Optional[str] = None) -> Optional[Dict[str, str]]:
    """Return the latest (newest LastModified) audio object for a folder."""
    client = get_s3_client()
    paginator = client.get_paginator("list_objects_v2")
    latest_obj = None

    prefixes_to_try = []
    if folder:
        clean_folder = folder.strip("/")
        if AWS_AUDIO_PREFIX:
            prefixes_to_try.append(f"{AWS_AUDIO_PREFIX.strip('/')}/{clean_folder}/")
            prefixes_to_try.append(f"{AWS_AUDIO_PREFIX.strip('/')}/{clean_folder}")
        prefixes_to_try.append(f"{clean_folder}/")
    if AWS_AUDIO_PREFIX:
        prefixes_to_try.append(f"{AWS_AUDIO_PREFIX.strip('/')}/")
    prefixes_to_try.append("")

    for prefix in prefixes_to_try:
        params = {"Bucket": AWS_AUDIO_BUCKET}
        if prefix:
            params["Prefix"] = prefix
        try:
            for page in paginator.paginate(**params):
                for obj in page.get("Contents", []):
                    if obj.get("Size", 0) == 0:
                        continue
                    if latest_obj is None or obj["LastModified"] > latest_obj["LastModified"]:
                        latest_obj = obj
        except Exception as e:
            print(f"[LIPSYNC] S3 list error for prefix '{prefix}': {e}")

        if latest_obj:
            print(f"🔍 [LIPSYNC] Found latest audio in S3 using prefix '{prefix}': {latest_obj['Key']}")
            break

    if not latest_obj:
        print(f"⚠️ [LIPSYNC] Warning: No audio objects found in S3 bucket '{AWS_AUDIO_BUCKET}'.")
        return None

    key = latest_obj["Key"]
    return {
        "key": key,
        "url": build_audio_public_url(key),
        "last_modified": latest_obj["LastModified"].isoformat(),
    }


# ---------------------------------------------------------------------------
# Tavus helpers
# ---------------------------------------------------------------------------
def _require_tavus_credentials():
    if not TAVUS_API_KEYS or not REPLICA_ID:
        raise RuntimeError("TAVUS_API_KEY (comma-separated) and REPLICA_ID must be set in the environment.")


def create_tavus_video(audio_url: str, video_name: str = "cloud-pipe-video", callback_url: Optional[str] = None):
    _require_tavus_credentials()

    def _do_create(api_key, a_url, v_name, cb_url):
        headers = {"Content-Type": "application/json", "x-api-key": api_key}
        body = {"replica_id": REPLICA_ID, "audio_url": a_url, "video_name": v_name}
        if cb_url:
            body["callback_url"] = cb_url

        resp = requests.post(VIDEOS_CREATE, headers=headers, json=body, timeout=30)
        try:
            resp.raise_for_status()
        except Exception:
            if resp.status_code == 429:
                print(f"[QUOTA LIMIT HIT] Tavus key ending in '...{api_key[-6:] if api_key else '?'}' hit its rate/usage limit (429). Rotating - if this happens on every key, you need a fresh TAVUS_API_KEY.")
            elif resp.status_code in (401, 403):
                print(f"[INVALID KEY] Tavus key ending in '...{api_key[-6:] if api_key else '?'}' was rejected ({resp.status_code}) - it's likely invalid/expired. Replace TAVUS_API_KEY in .env.")
            else:
                print("Tavus create failed:", resp.status_code, resp.text)
            raise
        return resp.json()

    return execute_tavus_with_retry(_do_create, audio_url, video_name, callback_url)


def poll_video(video_id: str, interval: int = 5, timeout: int = 600):
    _require_tavus_credentials()
    
    def _do_poll(api_key, vid, intr, to):
        headers = {"x-api-key": api_key}
        t0 = time.time()
        poll_count = 0
        last_status = None
        
        while True:
            poll_count += 1
            elapsed = time.time() - t0
            
            r = requests.get(VIDEOS_GET(vid), headers=headers, timeout=20)
            if r.status_code != 200:
                if r.status_code == 429:
                    print(f"[QUOTA LIMIT HIT] Tavus key ending in '...{api_key[-6:] if api_key else '?'}' hit its rate/usage limit (429) while polling. Rotating - if this happens on every key, you need a fresh TAVUS_API_KEY.")
                elif r.status_code in (401, 403):
                    print(f"[INVALID KEY] Tavus key ending in '...{api_key[-6:] if api_key else '?'}' was rejected ({r.status_code}) while polling. Replace TAVUS_API_KEY in .env.")
                else:
                    print(f"Poll error (attempt {poll_count}, elapsed={elapsed:.0f}s): {r.status_code} {r.text}")
                if r.status_code == 404:
                    raise FileNotFoundError(f"Video {vid} not found (404).")
                if r.status_code in (401, 403, 429):
                    r.raise_for_status()
            
            data = r.json()
            status = data.get("status")
            
            if status != last_status or poll_count == 1 or int(elapsed) % 30 == 0:
                print(f"⏳ [LIPSYNC-POLL #{poll_count}] status={status}, elapsed={elapsed:.0f}s/{to}s, video_id={vid[:10]}...")
                last_status = status
            else:
                if poll_count % 6 == 0:
                    print(f"⏳ [LIPSYNC-POLL] Still {status}... ({elapsed:.0f}s elapsed)")
            
            if status == "ready":
                print(f"✅ [LIPSYNC] Tavus video generation ready after {elapsed:.0f}s and {poll_count} polls")
                return data
            if status == "error":
                error_msg = data.get("error_message", "No error message provided")
                raise RuntimeError(f"Tavus returned error: {error_msg} | Full response: {json.dumps(data)}")
            if elapsed > to:
                raise TimeoutError(f"Timed out waiting for Tavus video after {elapsed:.0f}s ({poll_count} polls). Last status: {status}")
            time.sleep(intr)

    return execute_tavus_with_retry(_do_poll, video_id, interval, timeout)


def download_url(url: str, out_path: Path):
    print("Downloading:", url)
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    with open(out_path, "wb") as f:
        for chunk in r.iter_content(8192):
            if chunk:
                f.write(chunk)
    print("Saved to:", out_path.resolve())


def generate_lipsync_video(
    target_folder: str,
    audio_url: Optional[str] = None,
    video_filename: Optional[str] = None,
) -> Dict[str, object]:
    """Generate a Tavus video for the latest audio in S3 and save it locally & upload to S3."""
    folder_name = target_folder.strip() or "default"
    videos_dir = COLLECTIONS_ROOT.joinpath(folder_name, "videos")
    videos_dir.mkdir(parents=True, exist_ok=True)

    print(f"🎬 [LIPSYNC] Starting video generation pipeline for folder='{folder_name}'...")
    audio_info = None
    if audio_url:
        print(f"🔊 [LIPSYNC] Using explicit S3 audio URL: {audio_url}")
        audio_info = {"url": audio_url, "key": None}
    else:
        print(f"🔍 [LIPSYNC] Searching S3 for latest audio file in folder '{folder_name}'...")
        audio_info = get_latest_audio_from_s3(folder_name)
        if not audio_info:
            raise RuntimeError(
                f"No learning TTS audio files were found in S3 for folder '{folder_name}'. "
                "Ensure speech audio has been uploaded to S3 before invoking summary video generation."
            )
        audio_url = audio_info["url"]
        print(f"🔊 [LIPSYNC] Retrieved latest S3 audio URL: {audio_url}")

    filename = video_filename or f"learning-video-{int(time.time())}.mp4"
    print(f"⚡ [LIPSYNC] Creating Tavus lipsync video job (audio_url='{audio_url}', replica_id='{REPLICA_ID}')...")
    create_resp = create_tavus_video(audio_url, video_name=Path(filename).stem)
    video_id = create_resp.get("video_id") or create_resp.get("id")
    if not video_id:
        raise RuntimeError(f"No video_id returned by Tavus: {create_resp}")

    print(f"⏳ [LIPSYNC] Tavus video job created with ID '{video_id}'. Polling status until complete...")
    video_data = poll_video(video_id, interval=5, timeout=600)
    download_url_field = (
        video_data.get("download_url")
        or video_data.get("hosted_url")
        or video_data.get("result", {}).get("download_url")
    )
    if not download_url_field:
        raise RuntimeError("Tavus response did not include a downloadable video URL.")

    out_path = videos_dir / filename
    print(f"📥 [LIPSYNC] Downloading synthesized video from Tavus to {out_path}...")
    download_url(download_url_field, out_path)

    # Upload to S3
    s3_video_info = None
    try:
        print(f"☁️ [LIPSYNC] Uploading generated summary video to S3 bucket '{AWS_AUDIO_BUCKET}'...")
        s3_video_info = upload_video_to_s3(out_path, folder=folder_name)
        print(f"✅ [LIPSYNC] Summary video successfully uploaded to S3: {s3_video_info['url']}")
    except Exception as e:
        print(f"⚠️ [LIPSYNC] Warning: Failed to upload video to S3: {e}")

    relative_path = out_path.relative_to(COLLECTIONS_ROOT).as_posix()
    result = {
        "video_path": str(out_path),
        "relative_path": relative_path,
        "video_filename": filename,
        "audio": audio_info,
        "tavus": {"video_id": video_id, "raw": video_data},
        "s3_video": s3_video_info,
    }
    print(f"🎉 [LIPSYNC] [SUCCESS] Summary video generated and ready: filename='{filename}', S3 URL='{s3_video_info['url'] if s3_video_info else relative_path}'")
    return result


# ---------------------------------------------------------------------------
# CLI runner (optional)
# ---------------------------------------------------------------------------
def main():
    target_folder = os.getenv("DEFAULT_COLLECTION_FOLDER", "demo")
    audio_url = DEFAULT_AUDIO_URL
    if not audio_url:
        print("No AUDIO_PUBLIC_URL provided. Falling back to the latest S3 audio object.")
    try:
        result = generate_lipsync_video(target_folder=target_folder, audio_url=audio_url)
        print("✅ Lipsync video generated:", result["video_path"])
    except Exception as exc:
        print(f"❌ Unable to generate lipsync video: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()