FROM ghcr.io/astral-sh/uv:bookworm-slim

# Needed to build speakeasy2. Not sure why it isn't using the wheel.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
  --mount=type=cache,target=/var/lib/apt,sharing=locked \
  apt update && apt-get install -y cmake

ENV PYTHON_PATH=/app
WORKDIR /app
COPY ./pyproject.toml ./uv.lock example/populate_db.py /app/
COPY ./abstract2gene /app/abstract2gene/

RUN --mount=type=cache,target=/root/.cache \
  uv python install 3.12 && \
  uv python pin 3.12 && \
  uv sync --group app

RUN cat <<__EOF__ > startup.sh
#! /usr/bin/env bash

uv run python populate_db.py
uv run fastapi dev --host=0.0.0.0 webapp/main.py
__EOF__

RUN chmod u+x startup.sh

COPY ./webapp /app/webapp/

EXPOSE 8000
CMD ["./startup.sh"]
