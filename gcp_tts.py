"""
Google Cloud Text-to-Speech - the single TTS provider for every audio
generation call in this app (the Learning Hub's "Play Audio" summary
narration, and the AI Tutor chat's per-message "Listen" playback). Both
previously called Groq's PlayAI TTS model directly and independently;
this is the one place that talks to GCP now, so every caller gets the
exact same voice everywhere, by design.
"""
import html
import re
import os
import base64
import requests
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

GOOGLE_TTS_API_KEY = os.getenv("GOOGLE_TTS_API_KEY", "").strip()
GOOGLE_TTS_VOICE_NAME = os.getenv("GOOGLE_TTS_VOICE_NAME", "en-US-Studio-O").strip()
GOOGLE_TTS_URL = "https://texttospeech.googleapis.com/v1/text:synthesize"


def clean_text_for_tts(text: str) -> str:
    """
    Sanitizes markdown, HTML tags (<...>), LaTeX, URLs, and structural code tokens
    so GCP TTS speaks ONLY clean, natural human words matching what is rendered on screen.
    """
    if not text:
        return ""

    # Decode HTML entities like &lt; &gt; &amp; &quot;
    s = html.unescape(text)

    # 1. Strip raw HTML tags completely (e.g. <div class="...">, <span>, <br/>, <p>, <fileId>)
    s = re.sub(r'<[^>]+>', ' ', s)

    # 2. Strip Markdown Image links ![alt](url) and Hyperlinks [text](url) -> keep only 'text'
    s = re.sub(r'!\[.*?\]\(.*?\)', ' ', s)
    s = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', s)

    # 3. Strip URLs (http://... or https://...)
    s = re.sub(r'https?://\S+', ' ', s)

    # 4. Strip Code blocks ```...``` and inline code `...`
    s = re.sub(r'```[\s\S]*?```', ' ', s)
    s = re.sub(r'`([^`]+)`', r'\1', s)

    # 5. Clean LaTeX / Math expressions ($$x=y$$ or \(x=y\)) -> keep inner text or clean symbols
    s = re.sub(r'\$\$([\s\S]*?)\$\$', r'\1', s)
    s = re.sub(r'\$([^\$]+)\$', r'\1', s)
    s = re.sub(r'\\\(|\\\)', ' ', s)
    s = re.sub(r'\\\[|\\\]', ' ', s)
    s = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'\1 over \2', s)

    # 6. Replace mathematical symbols with natural spoken English words
    s = re.sub(r'\s*<\s*=', ' is less than or equal to ', s)
    s = re.sub(r'\s*>\s*=', ' is greater than or equal to ', s)
    s = re.sub(r'\s*<\s*', ' is less than ', s)
    s = re.sub(r'\s*>\s*', ' is greater than ', s)
    s = re.sub(r'\s*=\s*', ' equals ', s)
    s = re.sub(r'\s*\+\s*', ' plus ', s)
    s = re.sub(r'\s*─►|\s*─>|\s*->|\s*➔|\s*→\s*', ' leads to ', s)

    # 7. Strip structural Markdown tokens (#, ##, ###, *, **, _, __, ||, --, ~~, >)
    s = re.sub(r'#{1,6}\s*', ' ', s)
    s = re.sub(r'[\*\_\~\#\|\-\=]{2,}', ' ', s)
    s = re.sub(r'[\*\_\~]', ' ', s)

    # 8. Strip visual emoji boxes / block icons (🟦, 🟪, 🟧, 🟩, 💡, 🧑‍🏫, 📌, ⚠️, 🔬, 🧠)
    s = re.sub(r'[🟦🟪🟧🟩💡🧑‍🏫📌⚠️🔬🧠✓]', ' ', s)

    # 9. Clean up multiple whitespaces & newlines
    s = re.sub(r'\s+', ' ', s)
    
    return s.strip()


def synthesize_speech(text: str, voice_name: str | None = None) -> bytes:
    """
    Synthesizes `text` with GCP TTS and returns raw WAV bytes (LINEAR16 -
    matches every existing call site's "audio/wav" content type and .wav
    filenames, so nothing downstream - GridFS storage, S3 upload, lipsync -
    needs to change any format handling).
    """
    if not GOOGLE_TTS_API_KEY:
        raise RuntimeError("GOOGLE_TTS_API_KEY is not set in .env")
    if not text:
        raise RuntimeError("No text provided to synthesize")

    # Automatically clean HTML tags, markdown symbols, LaTeX, < > etc.
    cleaned = clean_text_for_tts(text)
    text_payload = cleaned if cleaned else text

    payload = {
        "input": {"text": text_payload[:4500]},
        "voice": {
            "languageCode": "en-US",
            "name": voice_name or GOOGLE_TTS_VOICE_NAME,
        },
        "audioConfig": {"audioEncoding": "LINEAR16"},
    }

    resp = requests.post(f"{GOOGLE_TTS_URL}?key={GOOGLE_TTS_API_KEY}", json=payload, timeout=30)

    if resp.status_code == 429:
        print(f"[QUOTA LIMIT HIT] GCP TTS key '...{GOOGLE_TTS_API_KEY[-6:]}' hit its rate/usage limit (429) - "
              f"you need a fresh GOOGLE_TTS_API_KEY.")
        raise RuntimeError("GCP TTS quota exceeded")

    # Confirmed live against the real endpoint: an invalid/unauthorized key
    # comes back as HTTP 400 with reason "API_KEY_INVALID" or "PERMISSION_DENIED"
    # in the body - NOT 401/403 like most APIs. Checking status code alone
    # would misclassify this as a generic error instead of "swap the key."
    reason = ""
    if resp.status_code in (400, 401, 403):
        try:
            reason = resp.json().get("error", {}).get("status", "")
        except Exception:
            reason = ""
    if resp.status_code in (401, 403) or reason in ("PERMISSION_DENIED", "UNAUTHENTICATED") or "API_KEY_INVALID" in resp.text:
        print(f"[INVALID KEY] GCP TTS key '...{GOOGLE_TTS_API_KEY[-6:]}' was rejected ({resp.status_code} {reason}) - "
              f"it's likely invalid, or the Text-to-Speech API isn't enabled for this key's GCP project.")
        raise RuntimeError(f"GCP TTS key rejected ({resp.status_code})")
    if resp.status_code == 400 and reason == "RESOURCE_EXHAUSTED":
        print(f"[QUOTA LIMIT HIT] GCP TTS key '...{GOOGLE_TTS_API_KEY[-6:]}' hit its rate/usage limit - "
              f"you need a fresh GOOGLE_TTS_API_KEY.")
        raise RuntimeError("GCP TTS quota exceeded")
    if not resp.ok:
        print(f"[GCP-TTS] [ERROR] Request failed ({resp.status_code}): {resp.text[:300]}")
        raise RuntimeError(f"GCP TTS request failed ({resp.status_code})")

    audio_b64 = resp.json().get("audioContent")
    if not audio_b64:
        raise RuntimeError("GCP TTS response had no audioContent")

    used_voice = voice_name or GOOGLE_TTS_VOICE_NAME
    print(f"[GCP-TTS] [OK] Synthesized {len(text)} char(s) of text with voice '{used_voice}'.")
    return base64.b64decode(audio_b64)
