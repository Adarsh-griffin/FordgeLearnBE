"""
Shared Groq client: multi-key rotation + retry.

Before this module existed, the same "parse GROQ_API_KEY, rotate on failure,
retry" logic was copy-pasted into ingest.py, retrieval.py, passmain_groq.py
and test_groq.py (four separate copies, some rotating on failure, some not).
This is the single place that owns it now; each module keeps its own thin
prompt-calling wrapper (they differ slightly, e.g. test_groq.py's
`reasoning_effort` param) but sources key rotation from here.
"""
import json
import os
import re
from groq import Groq
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

GROQ_API_KEYS_STR = os.getenv("GROQ_API_KEY", "")
GROQ_KEYS = [k.strip() for k in GROQ_API_KEYS_STR.split(",") if k.strip()]

if not GROQ_KEYS:
    raise ValueError(
        "GROQ_API_KEY is not configured. Set it in LearnBack/.env "
        "(comma-separated values are supported for key rotation)."
    )

_current_key_index = 0


def get_client() -> Groq:
    """Return a Groq client using the currently active key."""
    return Groq(api_key=GROQ_KEYS[_current_key_index])


def rotate_key() -> None:
    global _current_key_index
    _current_key_index = (_current_key_index + 1) % len(GROQ_KEYS)
    print(f"[GROQ] Rotating to API key index: {_current_key_index}")


def execute_with_retry(func, *args, **kwargs):
    """
    Call func(client, *args, **kwargs), rotating to the next Groq key and
    retrying on any exception, up to once per configured key.
    """
    max_retries = len(GROQ_KEYS)
    last_exception = None
    for attempt in range(max_retries):
        try:
            client = get_client()
            return func(client, *args, **kwargs)
        except Exception as e:
            print(f"[GROQ] Attempt {attempt + 1} failed with key index {_current_key_index}: {e}")
            last_exception = e
            rotate_key()
    raise last_exception


def groq_generate(prompt, model="openai/gpt-oss-20b", max_tokens=512, temperature=0.7, json_mode=False):
    """
    Generic helper for NEW code. Existing modules (ingest.py, test_groq.py,
    etc.) keep their own groq_generate wrappers to preserve exact prior
    behavior; this one is for code that doesn't need to match legacy quirks.
    Returns the raw text, or a parsed dict if json_mode=True (None on failure).

    Prefer groq_generate_json() below over json_mode=True for anything with
    a nested schema (an options array, a dict of lists, etc.) - see its
    docstring for why.
    """
    def _do_generate(client, p, mt, temp):
        kwargs = dict(
            model=model,
            messages=[{"role": "user", "content": p}],
            temperature=temp,
            max_completion_tokens=mt,
            top_p=1,
            # gpt-oss models can spend the ENTIRE token budget on internal
            # reasoning and emit no visible output at all for prompts with
            # thin/vague context (confirmed while building the diagnostic:
            # reasoning_tokens=1998/2000, finish_reason="length", content="").
            # test_groq.py's own hand-written groq_generate already sets
            # this for the same reason - matching it here.
            reasoning_effort="low",
            stream=False,
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        completion = client.chat.completions.create(**kwargs)
        return completion.choices[0].message.content.strip()

    try:
        text = execute_with_retry(_do_generate, prompt, max_tokens, temperature)
        if json_mode:
            return json.loads(text)
        return text
    except Exception as e:
        print(f"[GROQ] Error generating response after retries: {e}")
        return None


def groq_generate_json(prompt: str, max_tokens: int = 900, temperature: float = 0.4):
    """
    Plain-text completion + manual JSON parsing, instead of
    response_format={"type": "json_object"} (what groq_generate's json_mode
    uses). That mode proved unreliable on openai/gpt-oss-20b for nested
    schemas (an options array, a dict of lists) - reproducibly a 400
    json_validate_failed, every retry/key rotation, regardless of prompt
    wording (found while building the diagnostic quiz's MCQ generation).
    This is the preferred path for any new structured-output call.
    Returns None on any failure - callers must treat that as "try again
    later", never as a value to trust blindly.
    """
    def _do_generate(client, p, mt, temp):
        completion = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": p}],
            temperature=temp,
            max_completion_tokens=mt,
            top_p=1,
            reasoning_effort="low",
            stream=False,
        )
        return completion.choices[0].message.content

    try:
        text = execute_with_retry(_do_generate, prompt, max_tokens, temperature)
    except Exception as e:
        print(f"[GROQ] groq_generate_json failed after retries: {e}")
        return None

    if not text:
        return None

    cleaned = re.sub(r'^```(?:json)?\s*', '', text.strip())
    cleaned = re.sub(r'\s*```$', '', cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        print(f"[GROQ] groq_generate_json: failed to parse JSON: {e}\nRaw: {cleaned[:300]}")
        return None
