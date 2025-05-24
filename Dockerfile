FROM ghcr.io/astral-sh/uv:bookworm-slim

ENV PYTHON_PATH=/app
WORKDIR /app
COPY ./pyproject.toml ./uv.lock example/webapp/populate_db.py /app/
COPY ./abstract2gene /app/abstract2gene/

ARG EXTRA_TYPE=cpu
RUN --mount=type=cache,target=/root/.cache \
  uv python install 3.12 && \
  uv python pin 3.12 && \
  uv sync --extra app --extra ${EXTRA_TYPE}

RUN echo "#! /usr/bin/env bash" > startup.sh && \
  echo "uv run --extra ${EXTRA_TYPE} python populate_db.py" >> startup.sh && \
  echo "uv run --extra ${EXTRA_TYPE} fastapi \$A2G_MODE --host=abstract2gene webapp/main.py" >> startup.sh

RUN chmod u+x startup.sh

COPY ./webapp /app/webapp/

EXPOSE 8000
CMD ["./startup.sh"]
