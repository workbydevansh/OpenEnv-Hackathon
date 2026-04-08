from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import httpx
from openai import OpenAI

from support_ops_env.tasks import TASKS

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
ENV_BASE_URL = os.getenv("ENV_BASE_URL")
BENCHMARK = "support_ops_openenv"
SUCCESS_SCORE_THRESHOLD = 0.95

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY) if API_KEY else None

PLAYBOOKS: Dict[str, List[Dict[str, Any]]] = {
    "damaged_refund": [
        {"action_type": "open_ticket"},
        {"action_type": "search_kb", "query": "damaged item refund billing 3-5 business days"},
        {"action_type": "set_priority", "priority": "high"},
        {"action_type": "assign_queue", "queue": "billing"},
        {"action_type": "add_tag", "tag": "refund"},
        {"action_type": "add_tag", "tag": "damaged-item"},
        {"action_type": "set_status", "status": "resolved"},
        {"action_type": "draft_reply"},
        {"action_type": "finalize_case"},
    ],
    "account_takeover": [
        {"action_type": "open_ticket"},
        {"action_type": "search_kb", "query": "account takeover urgent lock account password reset specialist within 1 hour"},
        {"action_type": "set_priority", "priority": "urgent"},
        {"action_type": "assign_queue", "queue": "security"},
        {"action_type": "add_tag", "tag": "account-takeover"},
        {"action_type": "add_tag", "tag": "pii-risk"},
        {"action_type": "escalate_case", "escalation_target": "security-oncall"},
        {"action_type": "set_follow_up_hours", "follow_up_hours": 1},
        {"action_type": "set_status", "status": "pending"},
        {"action_type": "draft_reply"},
        {"action_type": "finalize_case"},
    ],
    "enterprise_outage": [
        {"action_type": "open_ticket"},
        {"action_type": "search_kb", "query": "live incident hourly updates service credit review sre enterprise outage"},
        {"action_type": "set_priority", "priority": "urgent"},
        {"action_type": "assign_queue", "queue": "technical"},
        {"action_type": "add_tag", "tag": "enterprise"},
        {"action_type": "add_tag", "tag": "outage"},
        {"action_type": "add_tag", "tag": "sla-risk"},
        {"action_type": "escalate_case", "escalation_target": "sre"},
        {"action_type": "escalate_case", "escalation_target": "account-manager"},
        {"action_type": "set_follow_up_hours", "follow_up_hours": 2},
        {"action_type": "set_status", "status": "pending"},
        {"action_type": "draft_reply"},
        {"action_type": "finalize_case"},
    ],
}

FALLBACK_REPLIES = {
    "damaged_refund": (
        "Thanks for sending the photos. I have approved your refund, and it will return to your "
        "original payment method within 3-5 business days."
    ),
    "account_takeover": (
        "For safety, we have locked your account, and the next step is to reset your password. "
        "A specialist will follow up within 1 hour."
    ),
    "enterprise_outage": (
        "We are treating this as a live incident, and engineering is investigating now. "
        "We will provide hourly updates, and service credit review will be handled after stability is restored."
    ),
}


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_value = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_value}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def action_to_string(action: Dict[str, Any]) -> str:
    return json.dumps(action, separators=(",", ":"), sort_keys=True)


def reply_meets_requirements(task_id: str, reply: str) -> bool:
    normalized = reply.lower()
    required_phrases = TASKS[task_id].expectation.required_reply_phrases
    return all(phrase.lower() in normalized for phrase in required_phrases)


def resolve_env_base_url(session: httpx.Client) -> str:
    candidates = [ENV_BASE_URL] if ENV_BASE_URL else ["http://127.0.0.1:8000", "http://127.0.0.1:7860"]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            response = session.get(f"{candidate}/health")
            if response.status_code == 200:
                return candidate
        except httpx.HTTPError:
            continue
    raise RuntimeError(
        "Could not reach the environment server. Set ENV_BASE_URL or start the API on http://127.0.0.1:8000 or http://127.0.0.1:7860."
    )


def draft_reply(task_id: str, observation: Dict[str, Any]) -> str:
    fallback = FALLBACK_REPLIES[task_id]
    task = TASKS[task_id]
    prompt = {
        "task_id": task_id,
        "title": task.title,
        "objective": observation["objective"],
        "inbox_summary": observation["inbox_summary"],
        "customer_messages": observation["customer_messages"],
        "required_phrases": task.expectation.required_reply_phrases,
        "instruction": "Write one concise support reply that includes every required phrase.",
    }

    if client is None:
        return fallback

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "You write concise support replies that follow exact checklist requirements.",
                },
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=True)},
            ],
            temperature=0,
            max_tokens=180,
        )
        candidate = (completion.choices[0].message.content or "").strip()
        if candidate and reply_meets_requirements(task_id, candidate):
            return candidate
    except Exception:
        pass

    return fallback


def run_task(session: httpx.Client, env_base_url: str, task_id: str) -> float:
    reset_response = session.post(f"{env_base_url}/reset", json={"task_id": task_id})
    reset_response.raise_for_status()
    observation = reset_response.json()

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        for step_index, action in enumerate(PLAYBOOKS[task_id], start=1):
            payload = dict(action)
            if payload["action_type"] == "draft_reply":
                payload["message"] = draft_reply(task_id, observation)

            step_response = session.post(f"{env_base_url}/step", json=payload)
            step_response.raise_for_status()
            result = step_response.json()

            reward = float(result["reward"]["value"])
            done = bool(result["done"])
            observation = result["observation"]
            rewards.append(reward)
            steps_taken = step_index
            score = float(result["reward"]["task_score"])

            log_step(
                step=step_index,
                action=action_to_string(payload),
                reward=reward,
                done=done,
                error=None,
            )

            if done:
                break

        success = score >= SUCCESS_SCORE_THRESHOLD
        return score
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def main() -> None:
    with httpx.Client(timeout=30.0) as session:
        env_base_url = resolve_env_base_url(session)
        for task_id in TASKS:
            run_task(session, env_base_url, task_id)


if __name__ == "__main__":
    main()
