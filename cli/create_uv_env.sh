#!/bin/bash
#
# Download and install uv (if not already installed), then create the
# project's virtual environment and install dependencies with uv sync.

set -e

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "error: uv not found after installation; check your PATH" >&2
    exit 1
fi

uv sync
