from __future__ import annotations

import uuid
from copy import deepcopy
from typing import Any, Dict, List, Tuple

from support_ops_env.graders import SCORE_EPSILON, compute_reward, grade_workspace
from support_ops_env.models import (
    StepResponse,
    SupportAction,
    SupportObservation,
    SupportReward,
    SupportState,
)
from support_ops_env.tasks import TASKS, SupportTask, task_catalog


class SupportOpsEnvironment:
    def __init__(self, default_task_id: str = "damaged_refund") -> None:
        self.default_task_id = default_task_id if default_task_id in TASKS else "damaged_refund"
        self._episode: Dict[str, Any] | None = None

    def list_tasks(self) -> List[Dict[str, str]]:
        return task_catalog()

    def reset(self, task_id: str | None = None) -> SupportObservation:
        task = TASKS.get(task_id or self.default_task_id, TASKS[self.default_task_id])
        self._episode = {
            "episode_id": str(uuid.uuid4()),
            "task": task,
            "step_count": 0,
            "done": False,
            "ticket_opened": False,
            "searched_articles": [],
            "last_kb_results": [],
            "last_outcome": "New episode created.",
            "history": [],
            "terminal_reason": None,
            "current_score": SCORE_EPSILON,
            "best_score": SCORE_EPSILON,
            "workspace": {
                "priority": None,
                "queue": None,
                "tags": [],
                "status": "open",
                "escalations": [],
                "follow_up_hours": None,
                "draft_reply": "",
                "finalized": False,
            },
        }
        return self._build_observation()

    def step(self, action: SupportAction) -> StepResponse:
        if self._episode is None:
            self.reset()

        assert self._episode is not None
        episode = self._episode
        task: SupportTask = episode["task"]

        if episode["done"]:
            observation = self._build_observation()
            reward = SupportReward(
                value=0.0,
                task_score=episode["current_score"],
                shaping_delta=0.0,
                step_cost=0.0,
                penalties=["episode_already_done"],
                components=grade_workspace(
                    task,
                    episode["workspace"],
                    episode["ticket_opened"],
                    episode["searched_articles"],
                )[1],
            )
            return StepResponse(
                observation=observation,
                reward=reward,
                done=True,
                info={"terminal_reason": episode["terminal_reason"], "task_score": episode["current_score"]},
            )

        previous_score = episode["current_score"]
        action_signature = action.model_dump(mode="json")
        action_signature_seen_before = action_signature in [item["action"] for item in episode["history"]]
        invalid_action, outcome = self._apply_action(action)

        episode["step_count"] += 1
        score, components, issues = grade_workspace(
            task,
            episode["workspace"],
            episode["ticket_opened"],
            episode["searched_articles"],
        )
        episode["current_score"] = score
        episode["best_score"] = max(episode["best_score"], score)

        finalized_successfully = action.action_type == "finalize_case" and not invalid_action
        reward_value, shaping_delta, step_cost, penalties = compute_reward(
            task=task,
            previous_score=previous_score,
            current_score=score,
            components=components,
            action_signature_seen_before=action_signature_seen_before,
            invalid_action=invalid_action,
            finalized_successfully=finalized_successfully,
        )

        if action.action_type == "finalize_case":
            episode["done"] = True
            episode["terminal_reason"] = "finalized"
        elif episode["step_count"] >= task.max_steps:
            episode["done"] = True
            episode["terminal_reason"] = "max_steps_reached"

        episode["last_outcome"] = outcome
        episode["history"].append(
            {
                "step": episode["step_count"],
                "action": action_signature,
                "outcome": outcome,
                "score": score,
            }
        )

        observation = self._build_observation()
        reward = SupportReward(
            value=reward_value,
            task_score=score,
            shaping_delta=shaping_delta,
            step_cost=step_cost,
            penalties=penalties,
            components=components,
        )
        info = {
            "task_score": score,
            "grader_components": components,
            "grader_issues": issues,
            "terminal_reason": episode["terminal_reason"],
            "passed": score >= 0.95 and episode["done"],
        }
        return StepResponse(observation=observation, reward=reward, done=episode["done"], info=info)

    def state(self) -> SupportState:
        if self._episode is None:
            self.reset()

        assert self._episode is not None
        episode = self._episode
        task: SupportTask = episode["task"]
        return SupportState(
            episode_id=episode["episode_id"],
            task_id=task.task_id,
            title=task.title,
            difficulty=task.difficulty,
            objective=task.objective,
            step_count=episode["step_count"],
            max_steps=task.max_steps,
            done=episode["done"],
            current_score=episode["current_score"],
            best_score=episode["best_score"],
            ticket_opened=episode["ticket_opened"],
            searched_articles=list(episode["searched_articles"]),
            workspace=deepcopy(episode["workspace"]),
            history=deepcopy(episode["history"]),
            terminal_reason=episode["terminal_reason"],
            available_tasks=self.list_tasks(),
        )

    def _apply_action(self, action: SupportAction) -> Tuple[bool, str]:
        assert self._episode is not None
        episode = self._episode
        workspace = episode["workspace"]
        task: SupportTask = episode["task"]

        if action.action_type == "open_ticket":
            episode["ticket_opened"] = True
            return False, "Opened the full ticket thread."

        if action.action_type == "search_kb":
            if not action.query:
                return True, "search_kb requires a query."
            results = self._search_kb(task, action.query)
            episode["last_kb_results"] = results
            episode["searched_articles"] = list(
                dict.fromkeys(episode["searched_articles"] + [item["article_id"] for item in results])
            )
            return False, f"Found {len(results)} knowledge base articles."

        if action.action_type == "set_priority":
            if not action.priority:
                return True, "set_priority requires a priority value."
            workspace["priority"] = action.priority
            return False, f"Priority set to {action.priority}."

        if action.action_type == "assign_queue":
            if not action.queue:
                return True, "assign_queue requires a queue."
            workspace["queue"] = action.queue
            return False, f"Queue assigned to {action.queue}."

        if action.action_type == "add_tag":
            if not action.tag:
                return True, "add_tag requires a tag."
            normalized_tag = action.tag.strip().lower()
            if normalized_tag not in workspace["tags"]:
                workspace["tags"].append(normalized_tag)
                return False, f"Added tag {normalized_tag}."
            return False, f"Tag {normalized_tag} was already present."

        if action.action_type == "set_status":
            if not action.status:
                return True, "set_status requires a status."
            workspace["status"] = action.status
            return False, f"Case status set to {action.status}."

        if action.action_type == "escalate_case":
            if not action.escalation_target:
                return True, "escalate_case requires an escalation target."
            if action.escalation_target not in workspace["escalations"]:
                workspace["escalations"].append(action.escalation_target)
                return False, f"Escalated case to {action.escalation_target}."
            return False, f"Escalation {action.escalation_target} was already requested."

        if action.action_type == "set_follow_up_hours":
            if action.follow_up_hours is None:
                return True, "set_follow_up_hours requires a numeric follow-up window."
            workspace["follow_up_hours"] = action.follow_up_hours
            return False, f"Follow-up set to {action.follow_up_hours} hours."

        if action.action_type == "draft_reply":
            if not action.message:
                return True, "draft_reply requires a message."
            workspace["draft_reply"] = action.message.strip()
            return False, "Customer reply drafted."

        if action.action_type == "finalize_case":
            workspace["finalized"] = True
            return False, "Case finalized and submitted to the grader."

        return True, f"Unsupported action type: {action.action_type}"

    def _search_kb(self, task: SupportTask, query: str) -> List[Dict[str, str]]:
        query_terms = {term for term in query.lower().split() if term}
        ranked: List[Tuple[int, Dict[str, str]]] = []
        for article in task.knowledge_base:
            haystack = " ".join([article.title, article.body, " ".join(article.tags)]).lower()
            score = sum(1 for term in query_terms if term in haystack)
            if score:
                ranked.append(
                    (
                        score,
                        {
                            "article_id": article.article_id,
                            "title": article.title,
                            "summary": article.body,
                        },
                    )
                )
        ranked.sort(key=lambda item: (-item[0], item[1]["article_id"]))
        return [item[1] for item in ranked[:3]]

    def _build_observation(self) -> SupportObservation:
        assert self._episode is not None
        episode = self._episode
        task: SupportTask = episode["task"]
        workspace = deepcopy(episode["workspace"])
        workspace["known_task_count"] = len(TASKS)

        return SupportObservation(
            task_id=task.task_id,
            title=task.title,
            difficulty=task.difficulty,
            objective=task.objective,
            inbox_summary=task.inbox_summary,
            customer_messages=task.customer_messages if episode["ticket_opened"] else [],
            attachments=task.attachments if episode["ticket_opened"] else [],
            kb_results=deepcopy(episode["last_kb_results"]),
            workspace=workspace,
            last_outcome=episode["last_outcome"],
            next_allowed_actions=[
                "open_ticket",
                "search_kb",
                "set_priority",
                "assign_queue",
                "add_tag",
                "set_status",
                "escalate_case",
                "set_follow_up_hours",
                "draft_reply",
                "finalize_case",
            ],
            steps_remaining=max(task.max_steps - episode["step_count"], 0),
        )


