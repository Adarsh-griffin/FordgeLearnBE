"""
Identity verification for the AI Tutor routes only.

Nothing else in the app requires this — every pre-existing route (upload,
QA, assessment, video) stays public/unauthenticated. Only new `/api/tutor/*`
routes are wrapped with @require_auth.

The "Get Started" flow lets a student either sign in (Clerk) or skip and
continue anonymously (a UUID generated client-side and kept in
localStorage). @require_auth accepts either:
  1. `Authorization: Bearer <clerk session token>` - verified against
     Clerk's public JWKS (no secret key needed for this part); g.user_id
     becomes the Clerk `sub` claim.
  2. `X-Anonymous-Id: <uuid>` - no server-side account at all; g.user_id
     becomes `anon:<uuid>`. This student model never follows them to
     another browser/device, and is lost if they clear site data - that's
     the accepted tradeoff for skipping sign-in.
If a student signs in later, their anonymous history is simply left behind
under the old `anon:<uuid>` id (no migration in this build).

Required env var (LearnBack/.env):
  CLERK_PUBLISHABLE_KEY  - same value the frontend uses (VITE_CLERK_PUBLISHABLE_KEY).
                           This is a *publishable* key, not a secret; safe to
                           duplicate here. It's used only to derive Clerk's
                           JWKS URL automatically.
Optional overrides:
  CLERK_JWT_ISSUER       - set this instead if key-derivation ever breaks
                           (Clerk Dashboard > API Keys shows the exact
                           "Frontend API URL" to use here).
  CLERK_SECRET_KEY       - not needed for token verification; kept for any
                           future call to Clerk's Backend API (e.g. fetching
                           a user's email/profile).
"""
import base64
import os
import re
from functools import wraps

import jwt
from dotenv import load_dotenv, find_dotenv
from flask import request, jsonify, g
from jwt import PyJWKClient

load_dotenv(find_dotenv())

CLERK_JWT_ISSUER = os.getenv("CLERK_JWT_ISSUER", "").rstrip("/")
CLERK_PUBLISHABLE_KEY = os.getenv("CLERK_PUBLISHABLE_KEY", "pk_test_YXdhaXRlZC1oYWdmaXNoLTgwMDMuY2xlcmsuYWNjb3VudHMuZGV2JA").strip()

_jwks_client = None


def _derive_issuer_from_publishable_key(publishable_key: str):
    """
    A Clerk publishable key is `pk_(test|live)_<base64url(domain + '$')>`.
    Decoding it gives us the Frontend API domain without a separate env var.
    """
    try:
        _, _, encoded = publishable_key.split("_", 2)
        padded = encoded + "=" * (-len(encoded) % 4)
        decoded = base64.b64decode(padded).decode("utf-8")
        domain = decoded.rstrip("$")
        return f"https://{domain}" if domain else None
    except Exception:
        return None


def _resolve_issuer() -> str:
    if CLERK_JWT_ISSUER:
        return CLERK_JWT_ISSUER
    if CLERK_PUBLISHABLE_KEY:
        derived = _derive_issuer_from_publishable_key(CLERK_PUBLISHABLE_KEY)
        if derived:
            return derived
    raise RuntimeError(
        "Cannot determine the Clerk issuer. Set CLERK_PUBLISHABLE_KEY "
        "(same value as VITE_CLERK_PUBLISHABLE_KEY) or CLERK_JWT_ISSUER "
        "in LearnBack/.env."
    )


def _get_jwks_client() -> PyJWKClient:
    global _jwks_client
    if _jwks_client is None:
        issuer = _resolve_issuer()
        _jwks_client = PyJWKClient(f"{issuer}/.well-known/jwks.json")
    return _jwks_client


def verify_clerk_token(token: str) -> dict:
    """Verify signature + issuer + expiry, return the decoded JWT payload."""
    issuer = _resolve_issuer()
    signing_key = _get_jwks_client().get_signing_key_from_jwt(token)
    payload = jwt.decode(
        token,
        signing_key.key,
        algorithms=["RS256"],
        issuer=issuer,
        options={"verify_aud": False},
    )
    return payload


_ANON_ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")


def require_auth(f):
    """
    Flask route decorator: resolves g.user_id from a Clerk bearer token if
    present, otherwise falls back to the X-Anonymous-Id header. Rejects the
    request only if neither is present/valid.
    """

    @wraps(f)
    def wrapper(*args, **kwargs):
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header.split(" ", 1)[1].strip()
            try:
                payload = verify_clerk_token(token)
            except Exception as e:
                return jsonify({"error": "Invalid or expired session token", "details": str(e)}), 401

            user_id = payload.get("sub")
            if not user_id:
                return jsonify({"error": "Token is missing a subject (user id) claim"}), 401

            g.user_id = user_id
            return f(*args, **kwargs)

        anon_id = request.headers.get("X-Anonymous-Id", "").strip()
        if anon_id and _ANON_ID_RE.match(anon_id):
            g.user_id = f"anon:{anon_id}"
            return f(*args, **kwargs)

        return jsonify({
            "error": "Provide either 'Authorization: Bearer <token>' (signed in) "
                     "or 'X-Anonymous-Id: <id>' (skipped sign-in)."
        }), 401

    return wrapper


def resolve_user_id_soft() -> str | None:
    """Same identity resolution as @require_auth (Clerk bearer token, else
    X-Anonymous-Id), but never rejects the request - returns None if neither
    is present/valid. For routes that aren't gated behind sign-in (upload,
    file listing) but still need to know *whose* data this is, so one
    student's uploads are never shown to another (see /api/upload and
    /api/files in test_groq.py)."""
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header.split(" ", 1)[1].strip()
        try:
            payload = verify_clerk_token(token)
            user_id = payload.get("sub")
            if user_id:
                return user_id
        except Exception as e:
            # Previously silent - a real verification failure here (e.g. a
            # CLERK_PUBLISHABLE_KEY mismatch between this backend and the
            # frontend that issued the token) would otherwise look
            # identical to "no identity provided at all", with nothing in
            # the logs to tell them apart.
            print(f"[AUTH] [WARNING] Soft Clerk token verification failed on {request.path}: {e}")

    anon_id = request.headers.get("X-Anonymous-Id", "").strip()
    if anon_id and _ANON_ID_RE.match(anon_id):
        return f"anon:{anon_id}"

    return None
