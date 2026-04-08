FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HOST=0.0.0.0
ENV PORT=7860

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r /app/requirements.txt

COPY support_ops_env /app/support_ops_env
COPY server /app/server
COPY openenv.yaml /app/openenv.yaml
COPY server.py /app/server.py
COPY inference.py /app/inference.py
COPY pre_validation_script.py /app/pre_validation_script.py
COPY README.md /app/README.md

EXPOSE 7860

CMD ["sh", "-c", "uvicorn server.app:app --host ${HOST:-0.0.0.0} --port ${PORT:-7860}"]
