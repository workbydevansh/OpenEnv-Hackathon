from __future__ import annotations

import os
from typing import Any, Dict

import httpx

from inference import FALLBACK_REPLIES, PLAYBOOKS, resolve_env_base_url
from support_ops_env.tasks import TASKS


def assert_condition(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run_task_validation(session: httpx.Client, env_base_url: str, task_id: str) -> Dict[str, Any]:
    reset_response = session.post(f"{env_base_url}/reset", json={"task_id": task_id})
    reset_response.raise_for_status()
    observation = reset_response.json()

    assert_condition(observation["task_id"] == task_id, f"Reset returned wrong task_id for {task_id}.")
    assert_condition("workspace" in observation, "Observation is missing workspace.")
    assert_condition("next_allowed_actions" in observation, "Observation is missing next_allowed_actions.")

    final_payload: Dict[str, Any] | None = None
    final_result: Dict[str, Any] | None = None
    step_count = 0

    for step_count, action in enumerate(PLAYBOOKS[task_id], start=1):
        payload = dict(action)
        if payload["action_type"] == "draft_reply":
            payload["message"] = FALLBACK_REPLIES[task_id]

        step_response = session.post(f"{env_base_url}/step", json=payload)
        step_response.raise_for_status()
        result = step_response.json()

        reward_value = float(result["reward"]["value"])
        task_score = float(result["reward"]["task_score"])
        assert_condition(-1.0 <= reward_value <= 1.0, f"Reward out of range for {task_id}.")
        assert_condition(0.0 < task_score < 1.0, f"Task score must be strictly between 0 and 1 for {task_id}.")

        final_payload = payload
        final_result = result
        if result["done"]:
            break

    assert_condition(final_payload is not None, f"No actions executed for {task_id}.")
    assert_condition(final_result is not None, f"No final result captured for {task_id}.")
    assert_condition(final_payload["action_type"] == "finalize_case", f"{task_id} did not finalize the case.")
    assert_condition(final_result["done"] is True, f"{task_id} did not terminate after finalization.")
    assert_condition(final_result["info"]["passed"] is True, f"{task_id} did not pass the grader.")
    assert_condition(
        float(final_result["reward"]["task_score"]) >= 0.95,
        f"{task_id} score was below the success threshold.",
    )

    state_response = session.get(f"{env_base_url}/state")
    state_response.raise_for_status()
    state = state_response.json()

    assert_condition(state["done"] is True, f"State endpoint was not terminal for {task_id}.")
    assert_condition(state["task_id"] == task_id, f"State endpoint returned wrong task for {task_id}.")
    assert_condition(state["workspace"]["finalized"] is True, f"Workspace was not finalized for {task_id}.")
    assert_condition(len(state["history"]) == step_count, f"History length mismatch for {task_id}.")

    return {
        "task_id": task_id,
        "steps": step_count,
        "score": float(final_result["reward"]["task_score"]),
    }


def main() -> None:
    requested_base_url = os.getenv("ENV_BASE_URL")
    with httpx.Client(timeout=30.0) as session:
        env_base_url = requested_base_url or resolve_env_base_url(session)

        root_response = session.get(f"{env_base_url}/")
        root_response.raise_for_status()
        root_payload = root_response.json()
        assert_condition(root_payload["environment"] == "support_ops_env", "Root endpoint returned the wrong environment name.")
        assert_condition(len(root_payload["tasks"]) >= 3, "Root endpoint returned fewer than three tasks.")

        health_response = session.get(f"{env_base_url}/health")
        health_response.raise_for_status()
        assert_condition(health_response.json()["status"] == "healthy", "Health endpoint did not report healthy.")

        tasks_response = session.get(f"{env_base_url}/tasks")
        tasks_response.raise_for_status()
        tasks_payload = tasks_response.json()
        assert_condition(len(tasks_payload) >= 3, "Tasks endpoint returned fewer than three tasks.")

        results = [run_task_validation(session, env_base_url, task_id) for task_id in TASKS]

    print("Pre-validation passed.")
    for result in results:
        print(f"{result['task_id']}: score={result['score']:.3f} steps={result['steps']}")


if __name__ == "__main__":
    main()

