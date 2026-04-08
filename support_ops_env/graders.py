from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from support_ops_env.tasks import SupportTask

# Keep task scores safely inside the open interval even after common
# 3-decimal rounding in logs or downstream validation.
SCORE_EPSILON = 1e-3


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def clamp_task_score(score: float) -> float:
    bounded_score = min(1.0 - SCORE_EPSILON, max(SCORE_EPSILON, score))
    return round(bounded_score, 3)


def grade_workspace(task: SupportTask, workspace: Dict[str, Any], ticket_opened: bool, searched_articles: List[str]) -> Tuple[float, Dict[str, float], List[str]]:
    expectation = task.expectation
    issues: List[str] = []
    components: Dict[str, float] = {}

    components["ticket_opened"] = 1.0 if ticket_opened else 0.0

    if expectation.required_articles:
        hits = len(set(searched_articles) & set(expectation.required_articles))
        components["research"] = hits / len(expectation.required_articles)
        if hits < len(expectation.required_articles):
            issues.append("Missing one or more key knowledge base articles.")
    else:
        components["research"] = 1.0

    components["priority"] = 1.0 if workspace.get("priority") == expectation.priority else 0.0
    if components["priority"] == 0.0:
        issues.append("Priority is incorrect.")

    components["queue"] = 1.0 if workspace.get("queue") == expectation.queue else 0.0
    if components["queue"] == 0.0:
        issues.append("Queue assignment is incorrect.")

    current_tags = set(workspace.get("tags", []))
    required_tags = set(expectation.required_tags)
    components["tags"] = len(current_tags & required_tags) / len(required_tags) if required_tags else 1.0
    if components["tags"] < 1.0:
        issues.append("Missing one or more required tags.")

    current_escalations = set(workspace.get("escalations", []))
    required_escalations = set(expectation.required_escalations)
    if required_escalations:
        components["escalations"] = len(current_escalations & required_escalations) / len(required_escalations)
        if components["escalations"] < 1.0:
            issues.append("Missing one or more required escalations.")
    else:
        components["escalations"] = 1.0

    expected_follow_up = expectation.follow_up_hours
    if expected_follow_up is None:
        components["follow_up"] = 1.0
    else:
        components["follow_up"] = 1.0 if workspace.get("follow_up_hours") == expected_follow_up else 0.0
        if components["follow_up"] == 0.0:
            issues.append("Follow-up window is incorrect.")

    components["status"] = 1.0 if workspace.get("status") == expectation.status else 0.0
    if components["status"] == 0.0:
        issues.append("Final case status is incorrect.")

    reply = _normalize(workspace.get("draft_reply", ""))
    if expectation.required_reply_phrases:
        matched = sum(1 for phrase in expectation.required_reply_phrases if _normalize(phrase) in reply)
        components["reply"] = matched / len(expectation.required_reply_phrases)
    else:
        components["reply"] = 1.0
    if components["reply"] < 1.0:
        issues.append("Reply is missing required customer-facing details.")

    if expectation.forbidden_reply_phrases:
        forbidden_hits = sum(1 for phrase in expectation.forbidden_reply_phrases if _normalize(phrase) in reply)
        if forbidden_hits:
            components["reply"] = max(0.0, components["reply"] - 0.5)
            issues.append("Reply contains a forbidden phrase.")

    components["finalized"] = 1.0 if workspace.get("finalized") else 0.0
    if components["finalized"] == 0.0:
        issues.append("Case has not been finalized.")

    weights = {
        "ticket_opened": 0.05,
        "research": 0.15,
        "priority": 0.10,
        "queue": 0.10,
        "tags": 0.10,
        "escalations": 0.15,
        "follow_up": 0.10,
        "status": 0.10,
        "reply": 0.10,
        "finalized": 0.05,
    }
    score = sum(components[name] * weight for name, weight in weights.items())
    return clamp_task_score(score), components, issues


def compute_reward(
    task: SupportTask,
    previous_score: float,
    current_score: float,
    components: Dict[str, float],
    action_signature_seen_before: bool,
    invalid_action: bool,
    finalized_successfully: bool,
) -> Tuple[float, float, float, List[str]]:
    penalties: List[str] = []
    shaping_delta = current_score - previous_score
    reward = shaping_delta

    step_cost = 0.01
    reward -= step_cost

    if action_signature_seen_before:
        reward -= 0.03
        penalties.append("repeated_action")

    if invalid_action:
        reward -= 0.08
        penalties.append("invalid_action")

    if finalized_successfully and current_score >= 0.95:
        reward += 0.10

    reward = max(-1.0, min(1.0, round(reward, 4)))
    return reward, round(shaping_delta, 4), step_cost, penalties
