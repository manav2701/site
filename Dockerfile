# Agent worker for Manav's deck.
#
# This image is also the reason the agent can run at all on the machine it was
# written on: LiveKit's Python stack ships an unsigned native library that the
# host's Windows Application Control policy blocks. Inside a Linux container
# that policy does not apply, and this is the same artefact any host will run.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# libgomp is needed by onnxruntime, which the silero/turn-detector models use.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates libgomp1 \
 && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /usr/local/bin/uv

WORKDIR /app

# Dependencies first so edits to agent.py or the knowledge file do not
# reinstall the whole stack.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY agent.py ./
COPY knowledge ./knowledge

# Bake the model weights into the image rather than downloading them on every
# cold start, which would otherwise show up as silence on the first question.
RUN uv run python agent.py download-files

# Credentials come from the environment at run time, never from the image.
CMD ["uv", "run", "python", "agent.py", "start"]
