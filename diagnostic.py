"""
Phase 2 of the AI Tutor build: an adaptive diagnostic quiz that produces
real, evidence-based mastery numbers per topic instead of a self-reported
"beginner/intermediate/expert" level (tutorplan.txt section 3-4).

Design choices, matching the same "don't let the LLM decide things Python
can decide deterministically" principle used in knowledge_graph.py:
  - Questions are multiple-choice with a server-known correct_key, so
    CORRECTNESS is a plain Python string comparison - the LLM never grades
    its own question.
  - Mastery is a simple banded update (+0.2 / -0.15), not an LLM guess.
  - The LLM is only asked to (a) write one grounded MCQ per topic, and
    (b) on a wrong answer, name the likely misconception in a few words -
    both bounded, low-stakes generation tasks, never the scoring itself.
  - "Harder" / "easier" branching (tutorplan.txt section 4) is operationalized
    using the real prerequisite graph from knowledge_graph.py: a correct
    answer advances to the next topic in prerequisite order (deeper); a
    wrong answer jumps back to test the least-tested prerequisite of the
    current topic (more foundational) before continuing.
"""
from groq_client import groq_generate_json
from knowledge_graph import section_preview

MAX_QUESTIONS = 7


def topological_order(topics: list) -> list:
    """
    Kahn's algorithm over topic prerequisites, so topics are always probed
    only after (or in place of) their prerequisites. Ties broken by
    original document order for determinism. Falls back to appending
    anything caught in an unexpected cycle in its original order - the
    edges from knowledge_graph.py are validated, but two independently
    proposed edges could still form a cycle, and this must not crash on it.
    """
    ids_in_order = [t["id"] for t in topics]
    index_of = {tid: i for i, tid in enumerate(ids_in_order)}
    by_id = {t["id"]: t for t in topics}

    in_degree = {tid: 0 for tid in ids_in_order}
    dependents = {tid: [] for tid in ids_in_order}
    for t in topics:
        for pre in t.get("prerequisites", []):
            if pre in by_id and pre != t["id"]:
                in_degree[t["id"]] += 1
                dependents[pre].append(t["id"])

    available = sorted(
        (tid for tid in ids_in_order if in_degree[tid] == 0),
        key=lambda tid: index_of[tid],
    )
    order = []
    seen = set()
    while available:
        tid = available.pop(0)
        if tid in seen:
            continue
        seen.add(tid)
        order.append(tid)

        newly_available = []
        for dep in dependents[tid]:
            in_degree[dep] -= 1
            if in_degree[dep] == 0 and dep not in seen:
                newly_available.append(dep)
        if newly_available:
            available.extend(newly_available)
            available.sort(key=lambda tid: index_of[tid])

    for tid in ids_in_order:  # cycle leftovers, in original order
        if tid not in seen:
            order.append(tid)

    return order


def update_mastery(previous: float, correct: bool) -> float:
    """Banded update, never LLM-decided. previous defaults to 0.0 (unassessed)."""
    if correct:
        return round(min(1.0, previous + 0.2), 3)
    return round(max(0.0, previous - 0.15), 3)


def compute_weak_prerequisites(topics: list, mastery: dict, threshold: float = 0.5, limit: int = 5) -> list:
    """
    Topics below the mastery threshold that at least one OTHER topic
    depends on - i.e. actual blocking gaps, not just any weak topic.
    Ordered by how many topics depend on them (most-blocking first).
    """
    depended_on = set()
    dependency_count = {}
    for t in topics:
        for pre in t.get("prerequisites", []):
            depended_on.add(pre)
            dependency_count[pre] = dependency_count.get(pre, 0) + 1

    weak = [tid for tid in depended_on if mastery.get(tid, 0.0) < threshold]
    weak.sort(key=lambda tid: dependency_count.get(tid, 0), reverse=True)
    return weak[:limit]


def generate_diagnostic_question(topic: dict, pages_text: list) -> dict | None:
    """One grounded MCQ testing real understanding of `topic`. None on failure."""
    preview = section_preview(topic, pages_text)
    prompt = f"""You are creating ONE diagnostic multiple-choice question to test whether
a student already understands a specific concept, before teaching begins.

Topic: {topic['title']}
Context: {preview or topic.get('summary', '')}

Write one multiple-choice question with exactly 4 options testing real
understanding of this topic (not a trivia/memorization question - it
should require actually understanding the concept to answer correctly).
Return ONLY a JSON object of this exact shape, with no markdown code
fences and no extra commentary:
{{
  "question": "...",
  "options": [
    {{"key": "A", "text": "..."}},
    {{"key": "B", "text": "..."}},
    {{"key": "C", "text": "..."}},
    {{"key": "D", "text": "..."}}
  ],
  "correct_key": "A"
}}
"""
    result = groq_generate_json(prompt, max_tokens=900, temperature=0.4)
    if not isinstance(result, dict):
        return None

    options = result.get("options")
    if (
        not isinstance(options, list)
        or len(options) < 2
        or not result.get("question")
        or not result.get("correct_key")
    ):
        return None

    valid_keys = {o.get("key") for o in options if isinstance(o, dict)}
    if result["correct_key"] not in valid_keys:
        return None

    return result


def classify_misconception(topic: dict, question_payload: dict, selected_key: str) -> str | None:
    """Best-effort, non-fatal: a short label for why a wrong answer was wrong."""
    options_str = "\n".join(
        f"{o.get('key')}) {o.get('text')}" for o in question_payload.get("options", []) if isinstance(o, dict)
    )
    prompt = f"""A student answered a diagnostic question incorrectly.

Topic: {topic['title']}
Question: {question_payload.get('question')}
Options:
{options_str}
Correct answer: {question_payload.get('correct_key')}
Student's answer: {selected_key}

In 5-10 words, name the likely misconception behind this wrong answer.
Return ONLY a JSON object, with no markdown code fences and no extra
commentary: {{"misconception": "..."}}
"""
    result = groq_generate_json(prompt, max_tokens=300, temperature=0.3)
    if isinstance(result, dict) and isinstance(result.get("misconception"), str):
        return result["misconception"].strip() or None
    return None
