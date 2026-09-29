"""
Durable PDF storage, for deploying to Render (or any host with an ephemeral
filesystem).

The bug this fixes: `/api/upload` used to only ever `file.save(filepath)`
to local disk, and store that local path in MongoDB. MongoDB itself is
fine here - it only ever held the small metadata (filename, page text,
mastery, etc.), never the PDF bytes, which was never the problem. The
actual problem is Render's local disk isn't persistent across a dyno
restart/redeploy/recycle - the moment that happens, every "filePath" in
Mongo points at a file that no longer exists. This is exactly what caused
a real failure seen in testing: passmain_groq.py trying to re-open an
uploaded PDF that was simply gone.

The fix: every uploaded PDF is also uploaded to the same S3 bucket this
app already uses for audio/video (see lipsync.py), under a documents/
prefix, and that S3 key is what's durably stored in Mongo. Local disk is
now just a same-request convenience cache - anything that needs the file
later calls ensure_local_copy() first, which transparently re-downloads
it from S3 if it's not sitting on disk anymore, so every existing
PdfReader/PyPDF2 call site keeps working unchanged.

GridFS (already used for TTS audio in test_groq.py) was the other option,
but PDFs would compete with the rest of the app's data for MongoDB
Atlas's free-tier storage quota (512MB, per this project's own
architecture spec) - S3 storage is separate from that and effectively
free at this scale, so it's the better fit here.
"""
import os

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# Reuses the same bucket/region/ACL config lipsync.py already uses for
# audio/video - no new AWS setup needed, just a new prefix inside it.
AWS_AUDIO_BUCKET = os.getenv("AWS_AUDIO_BUCKET") or os.getenv("AWS_S3_BUCKET")
AWS_REGION = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"
AWS_DOCS_PREFIX = os.getenv("AWS_DOCS_PREFIX", "documents")
AWS_AUDIO_PUBLIC_BASE_URL = os.getenv("AWS_AUDIO_PUBLIC_BASE_URL")
AWS_AUDIO_UPLOAD_ACL = os.getenv("AWS_AUDIO_UPLOAD_ACL", "public-read")

_s3_client = None


def get_s3_client():
    global _s3_client
    if _s3_client is None:
        if not AWS_AUDIO_BUCKET:
            raise RuntimeError("AWS_AUDIO_BUCKET (or AWS_S3_BUCKET) is not configured.")
        _s3_client = boto3.client("s3", region_name=AWS_REGION)
    return _s3_client


def _build_public_url(key: str) -> str:
    if AWS_AUDIO_PUBLIC_BASE_URL:
        return f"{AWS_AUDIO_PUBLIC_BASE_URL.rstrip('/')}/{key}"
    if AWS_REGION == "us-east-1":
        return f"https://{AWS_AUDIO_BUCKET}.s3.amazonaws.com/{key}"
    return f"https://{AWS_AUDIO_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{key}"


def upload_pdf_bytes(file_bytes: bytes, filename: str) -> dict:
    """Uploads the raw PDF bytes to S3 - the durable copy. Returns {key, url}."""
    client = get_s3_client()
    key = f"{AWS_DOCS_PREFIX.strip('/')}/{filename}"
    put_kwargs = {
        "Bucket": AWS_AUDIO_BUCKET,
        "Key": key,
        "Body": file_bytes,
        "ContentType": "application/pdf",
    }
    if AWS_AUDIO_UPLOAD_ACL:
        put_kwargs["ACL"] = AWS_AUDIO_UPLOAD_ACL
    try:
        client.put_object(**put_kwargs)
    except (BotoCoreError, ClientError) as exc:
        raise RuntimeError(f"Failed to upload PDF to S3: {exc}") from exc
    return {"key": key, "url": _build_public_url(key)}


def download_pdf_bytes(s3_key: str) -> bytes:
    """Fetches the raw PDF bytes back from S3."""
    client = get_s3_client()
    try:
        obj = client.get_object(Bucket=AWS_AUDIO_BUCKET, Key=s3_key)
        return obj["Body"].read()
    except (BotoCoreError, ClientError) as exc:
        raise RuntimeError(f"Failed to download PDF from S3 (key='{s3_key}'): {exc}") from exc


def ensure_local_copy(file_doc: dict, local_dir: str = "uploads") -> str:
    """
    Returns a local filesystem path to this document's PDF, downloading it
    from S3 first if the path Mongo remembers no longer exists on disk -
    the normal case after a Render restart, since local disk is ephemeral
    there. Callers that already do `open(path)` / `PdfReader(path)` need no
    other change: call this first instead of trusting file_doc['filePath']
    directly, and use the path it returns.
    """
    local_path = file_doc.get("filePath")
    if local_path and os.path.exists(local_path):
        return local_path

    s3_key = file_doc.get("s3_key")
    if not s3_key:
        raise FileNotFoundError(
            f"No local copy and no s3_key on file document '{file_doc.get('originalName')}' - "
            "this document was uploaded before durable S3 storage was added, or the original "
            "upload's S3 write failed. Re-upload it to fix this."
        )

    os.makedirs(local_dir, exist_ok=True)
    filename = file_doc.get("originalName") or os.path.basename(s3_key)
    target_path = os.path.join(local_dir, filename)
    pdf_bytes = download_pdf_bytes(s3_key)
    with open(target_path, "wb") as f:
        f.write(pdf_bytes)
    print(f"[STORAGE] Restored local copy of '{filename}' from S3 (was missing from disk).")
    return target_path
