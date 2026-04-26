ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest

# Build the React frontend
FROM node:20-slim AS frontend-builder
WORKDIR /app
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

FROM ${BASE_IMAGE} AS builder

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

COPY . /app/env

WORKDIR /app/env/meverse

RUN if ! command -v uv >/dev/null 2>&1; then \
        curl -LsSf https://astral.sh/uv/install.sh | sh && \
        mv /root/.local/bin/uv /usr/local/bin/uv && \
        mv /root/.local/bin/uvx /usr/local/bin/uvx; \
    fi

RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
        uv sync --frozen --no-install-project --no-editable; \
    else \
        uv sync --no-install-project --no-editable; \
    fi

RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
        uv sync --frozen --no-editable; \
    else \
        uv sync --no-editable; \
    fi

# Install root-level dependencies and backend dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    VIRTUAL_ENV=/app/env/meverse/.venv \
    uv pip install --python /app/env/meverse/.venv/bin/python -r /app/env/requirements.txt -r /app/env/backend/requirements.txt

FROM ${BASE_IMAGE}

WORKDIR /app

COPY --from=builder /app/env/meverse/.venv /app/.venv
COPY --from=builder /app/env /app/env
COPY --from=frontend-builder /app/dist /app/env/frontend/dist
# OpenEnv web interface looks for /app/README.md for the Playground readme dropdown
COPY --from=builder /app/env/README.md /app/README.md

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/env:/app/env/meverse:$PYTHONPATH"

EXPOSE 7860

CMD ["sh", "-c", "cd /app/env && uvicorn backend.app:app --host 0.0.0.0 --port 7860"]
