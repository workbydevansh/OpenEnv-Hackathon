from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


ActionType = Literal[
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
]

Priority = Literal["low", "medium", "high", "urgent"]
CaseStatus = Literal["open", "pending", "resolved"]
QueueName = Literal["billing", "fulfillment", "security", "technical", "account-management"]
EscalationTarget = Literal["security-oncall", "sre", "account-manager"]


class SupportAction(BaseModel):
    action_type: ActionType
    query: Optional[str] = None
    priority: Optional[Priority] = None
    queue: Optional[QueueName] = None
    tag: Optional[str] = None
    status: Optional[CaseStatus] = None
    escalation_target: Optional[EscalationTarget] = None
    follow_up_hours: Optional[int] = Field(default=None, ge=1, le=168)
    message: Optional[str] = None


class SupportReward(BaseModel):
    value: float = Field(..., ge=-1.0, le=1.0)
    task_score: float = Field(..., gt=0.0, lt=1.0)
    shaping_delta: float
    step_cost: float
    penalties: List[str] = Field(default_factory=list)
    components: Dict[str, float] = Field(default_factory=dict)


class SupportObservation(BaseModel):
    task_id: str
    title: str
    difficulty: Literal["easy", "medium", "hard"]
    objective: str
    inbox_summary: str
    customer_messages: List[str] = Field(default_factory=list)
    attachments: List[str] = Field(default_factory=list)
    kb_results: List[Dict[str, str]] = Field(default_factory=list)
    workspace: Dict[str, Any] = Field(default_factory=dict)
    last_outcome: str = ""
    next_allowed_actions: List[ActionType] = Field(default_factory=list)
    steps_remaining: int


class SupportState(BaseModel):
    episode_id: str
    task_id: str
    title: str
    difficulty: Literal["easy", "medium", "hard"]
    objective: str
    step_count: int
    max_steps: int
    done: bool
    current_score: float = Field(..., gt=0.0, lt=1.0)
    best_score: float = Field(..., gt=0.0, lt=1.0)
    ticket_opened: bool
    searched_articles: List[str] = Field(default_factory=list)
    workspace: Dict[str, Any] = Field(default_factory=dict)
    history: List[Dict[str, Any]] = Field(default_factory=list)
    terminal_reason: Optional[str] = None
    available_tasks: List[Dict[str, str]] = Field(default_factory=list)


class StepResponse(BaseModel):
    observation: SupportObservation
    reward: SupportReward
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


class ResetRequest(BaseModel):
    task_id: Optional[str] = None

