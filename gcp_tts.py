"""
Google Cloud Text-to-Speech - the single TTS provider for every audio
generation call in this app (the Learning Hub's "Play Audio" summary
narration, and the AI Tutor chat's per-message "Listen" playback).
Includes automatic gTTS fallback if GCP key is missing, quota exceeded, or invalid.
"""
import html
import re
import os
import base64
import io
import requests
from dotenv import load_dotenv, find_dotenv

try:
    from gtts import gTTS
    HAS_GTTS = True
except ImportError:
    HAS_GTTS = False

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

    # 1. Strip raw HTML tags completely
    s = re.sub(r'<[^>]+>', ' ', s)

    # 2. Strip Markdown Image links ![alt](url) and Hyperlinks [text](url) -> keep only 'text'
    s = re.sub(r'!\[.*?\]\(.*?\)', ' ', s)
    s = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', s)

    # 3. Strip URLs
    s = re.sub(r'https?://\S+', ' ', s)

    # 4. Strip Code blocks and inline code
    s = re.sub(r'```[\s\S]*?```', ' ', s)
    s = re.sub(r'`([^`]+)`', r'\1', s)

    # 5. Clean LaTeX / Math expressions
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

    # 7. Strip structural Markdown tokens
    s = re.sub(r'#{1,6}\s*', ' ', s)
    s = re.sub(r'[\*\_\~\#\|\-\=]{2,}', ' ', s)
    s = re.sub(r'[\*\_\~]', ' ', s)

    # 8. Strip visual emoji boxes / block icons
    s = re.sub(r'[🟦🟪🟧🟩💡🧑‍🏫📌⚠️🔬🧠✓]', ' ', s)

    # 9. Clean up multiple whitespaces & newlines
    s = re.sub(r'\s+', ' ', s)
    
    return s.strip()


def _fallback_gtts(text: str) -> bytes:
    if not HAS_GTTS:
        raise RuntimeError("gTTS package not installed")
    fp = io.BytesIO()
    tts = gTTS(text=text[:3000], lang='en')
    tts.write_to_fp(fp)
    return fp.getvalue()


def synthesize_speech(text: str, voice_name: str | None = None) -> bytes:
    """
    Synthesizes `text` with GCP TTS, with automatic gTTS fallback if GCP key is missing,
    quota exceeded, or invalid.
    """
    if not text:
        raise RuntimeError("No text provided to synthesize")

    # Automatically clean HTML tags, markdown symbols, LaTeX, < > etc.
    cleaned = clean_text_for_tts(text)
    text_payload = cleaned if cleaned else text

    if not GOOGLE_TTS_API_KEY:
        print("[GCP-TTS] GOOGLE_TTS_API_KEY not set, using gTTS fallback.")
        return _fallback_gtts(text_payload)

    payload = {
        "input": {"text": text_payload[:4500]},
        "voice": {
            "languageCode": "en-US",
            "name": voice_name or GOOGLE_TTS_VOICE_NAME,
        },
        "audioConfig": {"audioEncoding": "LINEAR16"},
    }

    try:
        resp = requests.post(f"{GOOGLE_TTS_URL}?key={GOOGLE_TTS_API_KEY}", json=payload, timeout=20)
        if resp.ok:
            audio_b64 = resp.json().get("audioContent")
            if audio_b64:
                used_voice = voice_name or GOOGLE_TTS_VOICE_NAME
                print(f"[GCP-TTS] [OK] Synthesized {len(text)} char(s) of text with GCP voice '{used_voice}'.")
                return base64.b64decode(audio_b64)
        
        print(f"[GCP-TTS] GCP request returned {resp.status_code}: {resp.text[:200]}. Falling back to gTTS...")
        return _fallback_gtts(text_payload)
    except Exception as err:
        print(f"[GCP-TTS] Exception during GCP synthesis: {err}. Falling back to gTTS...")
        return _fallback_gtts(text_payload)
