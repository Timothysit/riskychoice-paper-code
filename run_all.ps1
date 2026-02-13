$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

# 1) Ensure uv is installed
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
  Write-Host "[info] uv not found; installing with pip --user"
  python -m pip install --user -U uv

  # Add user Scripts to PATH for this session
  $userScripts = Join-Path $env:APPDATA "Python\Python*\Scripts"
  $env:PATH = "$env:PATH;$env:USERPROFILE\.local\bin"
}

# 2) Create venv + install exact deps from lock
uv venv .venv
uv sync

# 3) Run pipeline
uv run python dataset.py get-riskychoice-data
uv run python train.py --process-name window-decoding
uv run python train.py --process-name single-region-decoding
uv run python plots.py --process-name plot-windowed-decoding
uv run python plots.py --process-name plot-single-region-decoding

Write-Host "[done]"