#!/usr/bin/env bash
set -e  # exit immediately if any command fails
set -u  # error on unset variables

echo "Setting up python environment"
# Go to repo root (directory containing this script)
cd "$(dirname "$0")"

# Ensure uv is installed
if ! command -v uv >/dev/null 2>&1; then
  echo "[info] uv not found; installing with pip --user"
  python -m pip install --user -U uv
  export PATH="$HOME/.local/bin:$PATH"
fi

# Create venv + install exact deps from lock
uv venv .venv
uv sync

echo "Checking if data is already processed"
uv run python dataset.py get-riskychoice-data

echo "Running decoding analyses..."

uv run python train.py --process-name window-decoding
uv run python train.py --process-name single-region-decoding

echo "Running plots..."

uv run python plots.py --process-name plot-windowed-decoding
uv run python plots.py --process-name plot-single-region-decoding

echo "All done!"

