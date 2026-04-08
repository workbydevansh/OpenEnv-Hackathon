# Deployment Guide

## Hugging Face Space Setup

1. Create a new Hugging Face Space.
2. Choose `Docker` as the Space SDK.
3. Connect this repository or upload the project files.
4. Make sure the repo root contains:
   - `README.md`
   - `Dockerfile`
   - `openenv.yaml`
   - `inference.py`
   - `pre_validation_script.py`
   - `server.py`
   - `server/`
   - `support_ops_env/`

## Space Variables

Set these in the Space settings under Variables and secrets:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

Recommended values:

- `API_BASE_URL=https://router.huggingface.co/v1`
- `MODEL_NAME=Qwen/Qwen2.5-72B-Instruct`
- `HF_TOKEN=<your token>`

`ENV_BASE_URL` is not required inside the Space itself, but is useful when running `inference.py` or `pre_validation_script.py` from another machine against the deployed Space URL.

## Local Verification Before Push

Start the API:

```powershell
.\\.venv\\Scripts\\python.exe -m uvicorn server.app:app --host 127.0.0.1 --port 7860
```

Run the validator:

```powershell
$env:ENV_BASE_URL="http://127.0.0.1:7860"
.\\.venv\\Scripts\\python.exe pre_validation_script.py
```

Run the baseline:

```powershell
$env:ENV_BASE_URL="http://127.0.0.1:7860"
.\\.venv\\Scripts\\python.exe inference.py
```

## Post-Deploy Verification

After the Space is live, replace `<space-url>` with your deployed URL and run:

```powershell
$env:ENV_BASE_URL="<space-url>"
.\\.venv\\Scripts\\python.exe pre_validation_script.py
```

Optional quick checks:

```powershell
Invoke-WebRequest -UseBasicParsing <space-url>/health | Select-Object -ExpandProperty Content
Invoke-WebRequest -UseBasicParsing -Method Post <space-url>/reset | Select-Object -ExpandProperty Content
```

## Submission Checklist

- The Space returns `200` on `/health`.
- `POST /reset` returns a valid typed observation.
- `POST /step` returns a valid typed step response.
- `GET /state` returns the current typed state.
- `pre_validation_script.py` passes against the deployed URL.
- `inference.py` completes and prints `[START]`, `[STEP]`, and `[END]` logs.
- The Space environment has `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN` configured.
- The repo root still contains `inference.py`.
