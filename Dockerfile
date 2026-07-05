FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --upgrade pip && \
    python -m pip install '.[ai]'

RUN useradd --create-home --uid 10001 hurgor && \
    chown -R hurgor:hurgor /app
USER hurgor

ENTRYPOINT ["hurgor-client"]

