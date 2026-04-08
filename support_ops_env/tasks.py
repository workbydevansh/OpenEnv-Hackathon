from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal


Difficulty = Literal["easy", "medium", "hard"]


@dataclass(frozen=True)
class KnowledgeArticle:
    article_id: str
    title: str
    body: str
    tags: List[str]


@dataclass(frozen=True)
class TaskExpectation:
    priority: str
    queue: str
    required_tags: List[str]
    status: str
    required_escalations: List[str] = field(default_factory=list)
    follow_up_hours: int | None = None
    required_articles: List[str] = field(default_factory=list)
    required_reply_phrases: List[str] = field(default_factory=list)
    forbidden_reply_phrases: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class SupportTask:
    task_id: str
    title: str
    difficulty: Difficulty
    objective: str
    inbox_summary: str
    customer_messages: List[str]
    attachments: List[str]
    knowledge_base: List[KnowledgeArticle]
    expectation: TaskExpectation
    max_steps: int = 10


TASKS: Dict[str, SupportTask] = {
    "damaged_refund": SupportTask(
        task_id="damaged_refund",
        title="Damaged Item Refund",
        difficulty="easy",
        objective="Approve the refund, route the case correctly, and close the ticket with a compliant reply.",
        inbox_summary=(
            "Order #8812: customer received a shattered mug, already uploaded damage photos, "
            "and needs a refund before an event this Friday."
        ),
        customer_messages=[
            "Hi team, the ceramic mug in order #8812 arrived shattered. I uploaded two photos and do not want a replacement because the event is this Friday.",
            "Please let me know how quickly the refund will show up on my card.",
        ],
        attachments=["photo_front.jpg", "photo_box.jpg"],
        knowledge_base=[
            KnowledgeArticle(
                article_id="RET-201",
                title="Refund policy for damaged shipments",
                body=(
                    "If the customer already attached proof of damage, support can approve the refund immediately. "
                    "Refunds settle to the original payment method within 3-5 business days."
                ),
                tags=["refund", "damaged-item", "billing"],
            ),
            KnowledgeArticle(
                article_id="FUL-104",
                title="When to route damaged-item cases",
                body="Damaged item refunds should be owned by the billing queue after verification.",
                tags=["damaged-item", "billing"],
            ),
        ],
        expectation=TaskExpectation(
            priority="high",
            queue="billing",
            required_tags=["refund", "damaged-item"],
            status="resolved",
            required_articles=["RET-201", "FUL-104"],
            required_reply_phrases=[
                "approved your refund",
                "3-5 business days",
                "original payment method",
            ],
        ),
    ),
    "account_takeover": SupportTask(
        task_id="account_takeover",
        title="Account Takeover Response",
        difficulty="medium",
        objective="Treat the report as urgent, secure the account, and hand off to the security team with a safe customer message.",
        inbox_summary=(
            "User reports unfamiliar password reset emails and an unrecognized shipping address added to the account this morning."
        ),
        customer_messages=[
            "I got three password reset emails that I did not request, and my account now shows a shipping address in another state.",
            "I am worried someone can see my saved card and order history. Please lock the account if needed.",
        ],
        attachments=["auth_log_excerpt.txt"],
        knowledge_base=[
            KnowledgeArticle(
                article_id="SEC-301",
                title="Immediate response for suspected account takeover",
                body=(
                    "For suspected account takeover, lock the account, force a password reset, set priority to urgent, "
                    "and escalate to security-oncall."
                ),
                tags=["security", "account-takeover", "pii-risk"],
            ),
            KnowledgeArticle(
                article_id="AUTH-110",
                title="Customer reply template for security investigations",
                body=(
                    "Tell the customer the account has been locked temporarily, a password reset is required, "
                    "and a specialist will follow up within 1 hour."
                ),
                tags=["security", "customer-reply"],
            ),
        ],
        expectation=TaskExpectation(
            priority="urgent",
            queue="security",
            required_tags=["account-takeover", "pii-risk"],
            status="pending",
            required_escalations=["security-oncall"],
            follow_up_hours=1,
            required_articles=["SEC-301", "AUTH-110"],
            required_reply_phrases=[
                "locked your account",
                "reset your password",
                "specialist",
                "within 1 hour",
            ],
        ),
        max_steps=11,
    ),
    "enterprise_outage": SupportTask(
        task_id="enterprise_outage",
        title="Enterprise Outage Triage",
        difficulty="hard",
        objective="Triage a production outage report for an enterprise customer, escalate to SRE, and keep the customer on an active incident workflow.",
        inbox_summary=(
            "Enterprise customer reports API 5xx errors across multiple regions, contract has a premium SLA, and their launch window starts in two hours."
        ),
        customer_messages=[
            "Our EU and US jobs started failing with 502s at 09:12 UTC and the failures are still ongoing.",
            "We need confirmation that engineering is engaged and whether SLA review will happen if this is not fixed before launch.",
        ],
        attachments=["grafana_screenshot.png", "request_ids.csv"],
        knowledge_base=[
            KnowledgeArticle(
                article_id="OPS-220",
                title="Major outage customer communications",
                body=(
                    "When multiple regions are impacted, treat it as a live incident, escalate to sre immediately, and promise hourly updates while investigation continues."
                ),
                tags=["incident", "outage", "technical"],
            ),
            KnowledgeArticle(
                article_id="SLA-008",
                title="Enterprise SLA review workflow",
                body=(
                    "For premium enterprise customers during service incidents, note that service credit review happens after stability is restored. "
                    "Keep the case pending and set a 2 hour follow-up."
                ),
                tags=["sla-risk", "enterprise", "account-management"],
            ),
        ],
        expectation=TaskExpectation(
            priority="urgent",
            queue="technical",
            required_tags=["enterprise", "outage", "sla-risk"],
            status="pending",
            required_escalations=["sre", "account-manager"],
            follow_up_hours=2,
            required_articles=["OPS-220", "SLA-008"],
            required_reply_phrases=[
                "live incident",
                "engineering is investigating",
                "hourly updates",
                "service credit review",
            ],
        ),
        max_steps=13,
    ),
}


def task_catalog() -> List[Dict[str, str]]:
    return [
        {
            "task_id": task.task_id,
            "title": task.title,
            "difficulty": task.difficulty,
            "objective": task.objective,
        }
        for task in TASKS.values()
    ]




