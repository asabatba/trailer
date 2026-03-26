FROM ghcr.io/astral-sh/uv:0.11 AS uv

FROM python:3.14-slim-trixie AS builder

COPY --from=uv /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV UV_PROJECT_ENVIRONMENT=/app/.venv

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY .python-version pyproject.toml README.md uv.lock ./
COPY src ./src

RUN uv sync --frozen --no-dev --no-editable

FROM python:3.14-slim-trixie AS runtime

ENV PATH=/app/.venv/bin:$PATH
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv

EXPOSE 8000

CMD ["trailer-server", "--host", "0.0.0.0", "--port", "8000"]
