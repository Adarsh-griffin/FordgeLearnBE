"""
Phase 4 of the AI Tutor build: the actual lesson delivery loop - Teach ->
Check -> Adapt (tutorplan.txt section 9), one topic at a time from the
study plan built in Phase 3.

Same division of responsibility as every prior phase: Python owns state
and every decision it CAN make deterministically (which topic is next,
whether to advance/reteach/remediate, the mastery update itself); the LLM
only teaches and judges free-response understanding (there's no
deterministic key for "did this paragraph demonstrate understanding" the
way there was for the diagnostic's MCQs), and even that judgment is
reduced to a single bool the mastery formula consumes - never a made-up
numeric score.

Adapt rules, operationalized concretely:
  - checkpoint passed          -> advance to the next planned topic
  - checkpoint failed, and the
    topic has an untested weak
    prerequisite                -> teach that prerequisite first (a real,
                                    inserted micro-lesson - a remediation
                                    queue, not just a message), then
                                    return to the original topic
  - checkpoint failed, no weak
    prerequisite left to try     -> reteach the SAME topic with a
                                    strategy it hasn't tried yet
  - checkpoint failed 3 times   -> advance anyway (never get a student
                                    stuck in an infinite loop on one topic)
"""
from datetime import datetime, timezone

from groq_client import groq_generate_json
from knowledge_graph import section_preview
from diagnostic import update_mastery
from planner import mastery_band, VALID_DELIVERY_MODES

MAX_RETEACH_ATTEMPTS = 3
# Cycle order when re-teaching after a wrong checkpoint - "review" is
# excluded, it's only ever chosen by the planner for already-mastered topics.
_RETEACH_CYCLE = ["worked_example", "socratic", "direct_explanation"]


def recent_misconceptions_for_topic(diagnostic_log: list, lesson_log: list, topic_id: str, limit: int = 3) -> list:
    """Most recent misconception labels for this topic, from both the diagnostic and prior lesson checkpoints."""
    entries = [e for e in (diagnostic_log or []) + (lesson_log or []) if e.get("topic_id") == topic_id and e.get("misconception")]
    entries.sort(key=lambda e: e.get("timestamp") or datetime.min)
    return [e["misconception"] for e in entries[-limit:]]


def weakest_untested_prereq(topic: dict, mastery: dict, already_remediated: list) -> str | None:
    """The lowest-mastery prerequisite of `topic` not already remediated this session, if any is actually weak."""
    candidates = [
        pid for pid in topic.get("prerequisites", [])
        if pid not in already_remediated and mastery.get(pid, 0.0) < 0.5
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda pid: mastery.get(pid, 0.0))


def choose_delivery_mode(planned_mode: str, attempts: list, is_remediation: bool) -> str:
    """
    First attempt at a topic -> use what the planner (Phase 3) already
    decided. A remediation micro-lesson or a re-teach after a wrong
    checkpoint -> cycle to a strategy not yet tried for this topic.
    """
    if not attempts and not is_remediation:
        return planned_mode if planned_mode in VALID_DELIVERY_MODES else "direct_explanation"

    for mode in _RETEACH_CYCLE:
        if mode not in attempts:
            return mode
    return "direct_explanation"  # every strategy already tried - fall back rather than error


def generate_lesson(
    topic: dict,
    pages_text: list,
    mastery: dict,
    misconceptions: list,
    attempts: list,
    delivery_mode: str,
    is_remediation: bool,
) -> dict | None:
    """One Groq call: the actual teaching content + a checkpoint question. None on failure."""
    preview = section_preview(topic, pages_text)
    prereq_lines = "\n".join(
        f"  - {pid}: {round(mastery.get(pid, 0.0) * 100)}% mastery" for pid in topic.get("prerequisites", [])
    ) or "  (no prerequisites)"
    misconception_lines = "\n".join(f"  - {m}" for m in misconceptions) or "  (none known)"

    strategy_notes = {
        "worked_example": "Show one fully worked example step by step, then a second, lighter-guided example the student finishes conceptually themselves.",
        "socratic": "Do NOT give the answer directly. Ask a leading question first that targets the known misconception above, before any explanation.",
        "direct_explanation": "Give a clear, concise explanation with one illustrative example.",
        "review": "This student already knows this - a brief 2-3 sentence refresher only, do not re-teach it as if new.",
    }

    context_note = (
        "This is a brief prerequisite refresher inserted because the student is struggling with a "
        "later topic that depends on this one - keep it short and focused on just the gap."
        if is_remediation else ""
    )
    attempts_note = (
        f"Previous attempts to teach this exact topic already used: {', '.join(attempts)}. "
        "Do NOT repeat that approach - this must be a genuinely different angle."
        if attempts else ""
    )

    prompt = f"""You are an adaptive AI tutor teaching one topic at a time.

Topic: {topic['title']}
Content preview: {preview or topic.get('summary', '')}
Student's mastery of this topic: {round(mastery.get(topic['id'], 0.0) * 100)}%
Prerequisite mastery:
{prereq_lines}
Known misconceptions for this topic:
{misconception_lines}

Teaching strategy to use: {delivery_mode}
{strategy_notes.get(delivery_mode, '')}
{context_note}
{attempts_note}

Write a short lesson. Return ONLY a JSON object of this exact shape, with
no markdown code fences and no extra commentary:
{{
  "objective": "one sentence: what the student will understand after this",
  "explanation": "the main teaching content, following the strategy above",
  "example": "one concrete example or worked problem illustrating the idea",
  "checkpoint_question": "one question to check real understanding - should require genuine understanding to answer, not just recall of a definition"
}}
"""
    result = groq_generate_json(prompt, max_tokens=1400, temperature=0.5)
    if not isinstance(result, dict):
        return None
    required = ("objective", "explanation", "checkpoint_question")
    if not all(isinstance(result.get(k), str) and result.get(k) for k in required):
        return None
    result.setdefault("example", "")
    return result


def grade_checkpoint(topic: dict, lesson: dict, student_answer: str) -> dict | None:
    """
    Judges free-response understanding (there's no deterministic key for
    this, unlike the diagnostic's MCQs) and reduces that judgment to a
    single bool - the mastery formula, not the LLM, decides the actual
    score. None on failure.
    """
    prompt = f"""A student was taught the following and then asked a checkpoint question.

Topic: {topic['title']}
What they were taught: {lesson.get('explanation', '')}
Checkpoint question: {lesson.get('checkpoint_question', '')}
Student's answer: {student_answer}

Judge whether their answer demonstrates real understanding of the concept
(minor wording differences are fine; the substance must be correct - an
empty, off-topic, or "I don't know" answer is NOT understanding).

Return ONLY a JSON object of this exact shape, with no markdown code
fences and no extra commentary:
{{
  "understood": true,
  "feedback": "2-3 sentences of feedback. If wrong, don't just say incorrect - explain the actual gap using an example, the way a good tutor would.",
  "misconception": "a short label for the specific misconception if not understood, else null"
}}
"""
    result = groq_generate_json(prompt, max_tokens=700, temperature=0.3)
    if not isinstance(result, dict) or not isinstance(result.get("understood"), bool):
        return None
    if not isinstance(result.get("feedback"), str) or not result["feedback"]:
        return None
    if not isinstance(result.get("misconception"), str):
        result["misconception"] = None
    return result
