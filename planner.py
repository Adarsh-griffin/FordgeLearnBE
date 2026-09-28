"""
Phase 3 of the AI Tutor build: turn `topic_graph + mastery + goal + time`
into a sequenced, explained study plan - not a flat "Day 1: Chapter 1" list
(tutorplan.txt section 6-7, 13).

Same division of responsibility as diagnostic.py: Python decides every
QUANTITY (which topics make the cut, how much time each gets, in what
order) from the real mastery numbers and the real prerequisite graph -
never the LLM. The one Groq call only annotates that fixed, ordered,
already-budgeted list with a teaching strategy and a one-line reason per
topic, and every value it returns is validated against a known-good set
before being trusted (same pattern as knowledge_graph.py's prerequisite-id
validation and diagnostic.py's correct_key validation).
"""
from datetime import datetime, timezone

from diagnostic import topological_order
from groq_client import groq_generate_json

VALID_DELIVERY_MODES = {"worked_example", "socratic", "direct_explanation", "review"}

# Per-band time budget in minutes and Python-owned pedagogical intent -
# tutorplan.txt section 7's rule, made concrete:
#   <60%  mastery -> "teach"    : needs real instruction
#   60-75%        -> "practice": explain + examples + practice
#   75-90%        -> "apply"   : practice + application, light touch
#   >90%          -> "review"  : skip/refresher only
_BAND_MINUTES = {"teach": 15, "practice": 10, "apply": 7, "review": 3}


def mastery_band(score: float) -> str:
    if score < 0.6:
        return "teach"
    if score < 0.75:
        return "practice"
    if score < 0.9:
        return "apply"
    return "review"


def _misconceptions_by_topic(diagnostic_log: list) -> dict:
    """topic_id -> [misconception strings], from the diagnostic's wrong answers."""
    out = {}
    for entry in diagnostic_log or []:
        if entry.get("misconception"):
            out.setdefault(entry["topic_id"], []).append(entry["misconception"])
    return out


def _select_and_budget(topics: list, mastery: dict, available_minutes: int):
    """
    Walk topics in prerequisite order, band each by mastery, and greedily
    fill the time budget. Always includes at least one topic even if it
    alone exceeds the budget, so the plan is never empty. Returns
    (selected: list[dict], deferred: list[dict]) - both Python topic dicts
    annotated with band/estimated_minutes, not yet LLM-annotated.
    """
    by_id = {t["id"]: t for t in topics}
    order = topological_order(topics)

    selected, deferred = [], []
    total = 0
    for tid in order:
        topic = by_id[tid]
        band = mastery_band(mastery.get(tid, 0.0))
        minutes = _BAND_MINUTES[band]
        annotated = {**topic, "band": band, "estimated_minutes": minutes, "mastery": mastery.get(tid, 0.0)}

        if not selected or total + minutes <= available_minutes:
            selected.append(annotated)
            total += minutes
        else:
            deferred.append(annotated)

    return selected, deferred, total


def _annotate_with_llm(selected: list, goal: str, misconceptions: dict) -> dict:
    """
    One Groq call: for each selected topic (fixed order, fixed time budget),
    propose a delivery_mode and a one-line reason grounded in the mastery
    evidence. Returns {} on failure - callers fall back to a sensible
    Python-only default per topic rather than blocking plan generation on
    an LLM call succeeding.
    """
    if not selected:
        return {}

    lines = []
    for t in selected:
        note = f" | known misconception: {misconceptions[t['id']][-1]}" if misconceptions.get(t["id"]) else ""
        lines.append(
            f"{t['id']} | {t['title']} | mastery: {round(t['mastery'] * 100)}% ({t['band']})"
            f" | preview: {t.get('summary', '')}{note}"
        )

    prompt = f"""A student's goal is: "{goal}"

Below is a FIXED, ordered list of topics already selected for their study
session, with their measured mastery level and how many minutes each will
get. Do not reorder, add, or remove topics - only annotate each one.

{chr(10).join(lines)}

For each topic id, choose ONE delivery_mode from exactly these options:
- "worked_example": for procedural/quantitative topics the student hasn't
  mastered yet - show a fully worked example before they try one alone.
- "socratic": for topics where a specific misconception is listed - ask a
  leading question that targets that exact misconception rather than
  re-explaining from scratch.
- "direct_explanation": for conceptual/recall topics with no misconception,
  not yet mastered.
- "review": for topics already at high mastery - a brief refresher only,
  do not re-teach it as if new.

Also write a one-sentence "reason" for each topic explaining why it's
being taught now, in this depth, referencing the actual mastery percentage
or misconception above - not a generic sentence that could apply to any
topic.

Return ONLY a JSON object of this exact shape:
{{
  "steps": {{
    "t1": {{"delivery_mode": "worked_example", "reason": "..."}},
    "t2": {{"delivery_mode": "review", "reason": "..."}}
  }}
}}
"""
    result = groq_generate_json(prompt, max_tokens=1400, temperature=0.4)
    steps = result.get("steps") if isinstance(result, dict) else None
    return steps if isinstance(steps, dict) else {}


def generate_study_plan(topics: list, mastery: dict, goal: str, available_minutes: int, diagnostic_log: list = None) -> dict:
    """
    Build the full plan. Returns:
    {
      "goal", "available_minutes", "total_minutes", "generated_at",
      "steps": [{topic_id, title, mastery, band, estimated_minutes,
                 delivery_mode, reason, prerequisites}, ...],
      "deferred_topics": [{topic_id, title, band}, ...]  # didn't fit the budget
    }
    """
    selected, deferred, total_minutes = _select_and_budget(topics, mastery, available_minutes)
    misconceptions = _misconceptions_by_topic(diagnostic_log)
    llm_steps = _annotate_with_llm(selected, goal, misconceptions)

    steps = []
    for t in selected:
        annotation = llm_steps.get(t["id"]) if isinstance(llm_steps.get(t["id"]), dict) else {}
        delivery_mode = annotation.get("delivery_mode")
        if delivery_mode not in VALID_DELIVERY_MODES:
            # Fallback if the LLM call failed or returned something invalid:
            # a safe Python-only default from the band alone.
            delivery_mode = "review" if t["band"] == "review" else "direct_explanation"
        reason = annotation.get("reason") if isinstance(annotation.get("reason"), str) else None
        if not reason:
            reason = f"Mastery is {round(t['mastery'] * 100)}% ({t['band']}), based on your diagnostic."

        steps.append({
            "topic_id": t["id"],
            "title": t["title"],
            "mastery": t["mastery"],
            "band": t["band"],
            "estimated_minutes": t["estimated_minutes"],
            "delivery_mode": delivery_mode,
            "reason": reason,
            "prerequisites": t.get("prerequisites", []),
        })

    return {
        "goal": goal,
        "available_minutes": available_minutes,
        "total_minutes": total_minutes,
        "steps": steps,
        "deferred_topics": [
            {"topic_id": t["id"], "title": t["title"], "band": t["band"]} for t in deferred
        ],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
