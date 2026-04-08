from __future__ import annotations

import os

import uvicorn
from fastapi import Body, FastAPI

from support_ops_env.environment import SupportOpsEnvironment
from support_ops_env.models import ResetRequest, StepResponse, SupportAction, SupportObservation, SupportState

app = FastAPI(
    title="SupportOps OpenEnv",
    description="Customer support operations simulator with graded tasks and trajectory rewards.",
    version="0.1.0",
)
env = SupportOpsEnvironment()


@app.get("/")
def root() -> dict[str, object]:
    return {
        "status": "ok",
        "environment": "support_ops_env",
        "spec": ["reset", "step", "state"],
        "tasks": env.list_tasks(),
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.get("/tasks")
def tasks() -> list[dict[str, str]]:
    return env.list_tasks()


@app.post("/reset", response_model=SupportObservation)
def reset(payload: ResetRequest | None = Body(default=None)) -> SupportObservation:
    task_id = payload.task_id if payload else None
    return env.reset(task_id=task_id)


@app.post("/step", response_model=StepResponse)
def step(action: SupportAction) -> StepResponse:
    return env.step(action)


@app.get("/state", response_model=SupportState)
def state() -> SupportState:
    return env.state()


@app.post("/state", response_model=SupportState)
def state_post() -> SupportState:
    return env.state()


def main() -> None:
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run("server.app:app", host=host, port=port)

if __name__ == "__main__":
    main()
