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

RUN echo "#! /usr/bin/env bash" > startup.sh && \
    echo "uv run python populate_db.py" >> startup.sh && \
    echo "uv run fastapi \$A2G_MODE --host=0.0.0.0 webapp/main.py" >> startup.sh

RUN chmod u+x startup.sh

COPY ./webapp /app/webapp/

EXPOSE 8000
CMD ["./startup.sh"]
