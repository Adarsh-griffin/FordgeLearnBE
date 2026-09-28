"""
Phase 1 of the AI Tutor build: turn the existing linear PageIndex tree
(files.page_index.structure, built deterministically by ingest.py) into a
prerequisite graph.

Design choice, per tutorplan.txt's "don't let the LLM hallucinate the
curriculum" principle: the topic list itself (id, title, page_range) is
built in plain Python, 1:1 from the document structure ingest.py already
extracted - the LLM never invents, renames, or reorders topics. Its only
job is proposing prerequisite EDGES between those fixed nodes, and every
edge it returns is validated afterward (must reference a real topic id,
never itself). This bounds how wrong the LLM can go to "picked a bad edge",
never "invented a topic that isn't in the book".

Cached once per document as files.topic_graph (see /api/tutor/topics in
test_groq.py) - this module doesn't touch MongoDB itself.
"""
from datetime import datetime, timezone

from groq_client import groq_generate


def _build_topics_from_structure(structure: list) -> list:
    """One topic node per page_index.structure entry, in document order."""
    topics = []
    for i, entry in enumerate(structure):
        page_range = entry.get("page_range") or [None, None]
        topics.append({
            "id": f"t{i + 1}",
            "title": entry.get("title") or f"Section {i + 1}",
            "page_range": page_range,
            "summary": entry.get("summary", ""),
            "prerequisites": [],  # filled in by _propose_prerequisites
        })
    return topics


def section_preview(topic: dict, pages_text: list, max_chars: int = 300) -> str:
    """
    Short text preview for a topic's page range, to ground LLM reasoning
    about it. Public (no leading underscore) because diagnostic.py reuses
    this exact logic for question generation instead of duplicating it.
    """
    start, end = topic["page_range"]
    if start is None:
        return topic.get("summary", "")

    snippets = []
    for p in pages_text:
        if start <= p.get("page_num", -1) <= (end or start):
            snippets.append(p.get("text", ""))
        if len(snippets) >= 2:
            break

    preview = " ".join(snippets)[:max_chars].replace("\n", " ").strip()
    return preview or topic.get("summary", "")


def _propose_prerequisites(topics: list, pages_text: list) -> dict:
    """
    One Groq JSON-mode call: given the FIXED, ordered topic list (id + title
    + a short preview), return which topic ids are prerequisites for which.
    Returns {} on any failure - callers treat that as "no edges found" and
    ship a graph with just the (still useful) topic list.
    """
    if len(topics) < 2:
        return {}

    topic_lines = [
        f"{t['id']} | {t['title']} | preview: {section_preview(t, pages_text)}"
        for t in topics
    ]

    prompt = f"""You are analyzing a textbook's section list to find prerequisite relationships.

Below is the FIXED, ordered list of sections (id | title | short preview).
Do not invent new sections or rename these - only reason about which
sections a student needs to understand BEFORE another section will make
sense.

{chr(10).join(topic_lines)}

Return ONLY a JSON object of this exact shape (omit a section entirely if
it has no prerequisites):
{{
  "prerequisites": {{
    "t3": ["t1", "t2"],
    "t5": ["t3"]
  }}
}}
"""
    result = groq_generate(prompt, max_tokens=1024, temperature=0.1, json_mode=True)
    if not isinstance(result, dict):
        return {}
    prereqs = result.get("prerequisites")
    return prereqs if isinstance(prereqs, dict) else {}


def extract_topic_graph(page_index: dict, pages_text: list) -> dict:
    """
    Build the prerequisite graph for a document.
    Returns {"topics": [{id, title, page_range, summary, prerequisites}], "generated_at": iso}.
    """
    structure = (page_index or {}).get("structure") or []
    topics = _build_topics_from_structure(structure)
    valid_ids = {t["id"] for t in topics}

    raw_prereqs = _propose_prerequisites(topics, pages_text or [])

    for t in topics:
        proposed = raw_prereqs.get(t["id"], [])
        if not isinstance(proposed, list):
            proposed = []
        clean = []
        for pid in proposed:
            if isinstance(pid, str) and pid in valid_ids and pid != t["id"] and pid not in clean:
                clean.append(pid)
        t["prerequisites"] = clean

    return {
        "topics": topics,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
