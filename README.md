---
title: SupportOps OpenEnv
sdk: docker
app_port: 7860
tags:
  - openenv
  - fastapi
  - customer-support
---

# SupportOps OpenEnv

SupportOps OpenEnv is a real-world customer support operations simulator built for the standard OpenEnv `reset()`, `step()`, and `state()` interaction loop. The environment models common human work: reading tickets, searching internal policy articles, assigning ownership, tagging the case, drafting a customer reply, and finalizing the workflow for grading.

## Motivation

This environment focuses on customer support because it creates realistic agent pressure:

- the agent must combine information gathering, routing, policy use, and communication
- partial progress matters, so shaped reward is more informative than binary reward alone
- outcomes can still be graded deterministically

## API surface

The FastAPI server exposes:

- `POST /reset`
- `POST /step`
- `GET /state`

Helper endpoints:

- `GET /`
- `GET /health`
- `GET /tasks`

## Typed spaces

### Action

`SupportAction` supports these action types:

- `open_ticket`
- `search_kb`
- `set_priority`
- `assign_queue`
- `add_tag`
- `set_status`
- `escalate_case`
- `set_follow_up_hours`
- `draft_reply`
- `finalize_case`

Fields include `query`, `priority`, `queue`, `tag`, `status`, `escalation_target`, `follow_up_hours`, and `message`.

### Observation

`SupportObservation` includes:

- task metadata: `task_id`, `title`, `difficulty`, `objective`
- initial inbox summary
- full customer thread and attachments after `open_ticket`
- latest knowledge-base search results
- current workspace snapshot
- `last_outcome`, `next_allowed_actions`, and `steps_remaining`

### State

`SupportState` adds:

- episode metadata
- current and best score
- whether the ticket has been opened
- searched article ids
- action history
- terminal reason
- task catalog

### Reward

`SupportReward` includes:

- scalar reward value in `[-1, 1]`
- normalized task score in `[0, 1]`
- shaping delta
- step cost
- penalties
- component-level grader breakdown

## Tasks

The environment ships with three deterministic tasks and graders:

1. `damaged_refund` (easy)
   Approve a damaged-item refund, route the case to billing, write a compliant customer reply, and resolve the case.
2. `account_takeover` (medium)
   Treat the case as urgent, route to security, escalate to `security-oncall`, set a one-hour follow-up, and leave the case pending.
3. `enterprise_outage` (hard)
   Treat the case as a live incident, escalate to SRE and the account manager, set a two-hour follow-up, and keep the case pending.

Each grader checks a deterministic rubric across:

- ticket opening
- article retrieval
- queue and priority correctness
- required tags
- required escalations
- follow-up window
- final status
- required reply phrases
- finalization

All final task scores are normalized to `[0, 1]`.

## Reward function

The environment rewards progress across the full trajectory:

- positive delta for improving the grader score
- `0.01` step cost each turn
- additional penalty for repeated actions
- additional penalty for invalid actions
- bonus for high-quality finalization

This makes the environment useful for learning instead of only pass/fail evaluation.

## Repo structure

```text
support_ops_env/
  __init__.py
  environment.py
  graders.py
  models.py
  tasks.py
server/
  __init__.py
  app.py
Dockerfile
DEPLOYMENT.md
inference.py
openenv.yaml
pre_validation_script.py
server.py
```

## Local setup

```powershell
uv venv
.venv\Scripts\activate
uv pip install -e .
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

## Baseline inference

The required baseline script is [`inference.py`](/D:/Scaler_Hackathon/inference.py). It:

- uses the OpenAI Python client for all LLM calls
- reads `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN` or `OPENAI_API_KEY`
- emits only the required `[START]`, `[STEP]`, and `[END]` stdout lines
- runs all three tasks
- falls back to deterministic compliant replies when no API key is present, which keeps local smoke tests reproducible

Environment variables:

```powershell
$env:API_BASE_URL="https://router.huggingface.co/v1"
$env:MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
$env:HF_TOKEN="your-token"
$env:ENV_BASE_URL="http://127.0.0.1:8000"
python inference.py
```

Expected baseline scores with the included task playbooks and safe reply fallbacks:

- `damaged_refund`: `1.00`
- `account_takeover`: `1.00`
- `enterprise_outage`: `1.00`

## Docker and Hugging Face Space

Build and run locally:

```powershell
docker build -t support-ops-openenv .
docker run -p 7860:7860 support-ops-openenv
```

The README front matter configures the repo as a Docker-based Hugging Face Space and tags it with `openenv`.

Deployment steps are documented in [`DEPLOYMENT.md`](/D:/Scaler_Hackathon/DEPLOYMENT.md).

Minimum Space configuration:

- SDK: `Docker`
- App port: `7860`
- Variables/secrets:
  - `API_BASE_URL`
  - `MODEL_NAME`
  - `HF_TOKEN`

## Validation

Quick checks:

```powershell
curl http://127.0.0.1:7860/
curl http://127.0.0.1:7860/health
curl -Method Post http://127.0.0.1:7860/reset
```

If `openenv` is installed:

```powershell
openenv validate
```

The provided [`pre_validation_script.py`](/D:/Scaler_Hackathon/pre_validation_script.py) performs a local or hosted smoke test of the API surface, task catalog, reward ranges, and deterministic pass conditions:

```powershell
$env:ENV_BASE_URL="http://127.0.0.1:8000"
python pre_validation_script.py
```

If you deploy the environment, point `ENV_BASE_URL` at the Hugging Face Space URL before running the same script.

Example against a deployed Space:

```powershell
$env:ENV_BASE_URL="https://<your-space-subdomain>.hf.space"
python pre_validation_script.py
```

## Notes

- `POST /reset` accepts an optional JSON body such as `{"task_id": "enterprise_outage"}`.
- If no task id is supplied, the environment defaults to `damaged_refund`.
