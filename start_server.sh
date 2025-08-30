#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
# activate venv if present
if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
else
  echo ".venv not found, trying system python environment"
fi
# start uvicorn in background, store pid, and tail the log
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload > uvicorn.log 2>&1 &
echo $! > uvicorn.pid
tail -n +1 -f uvicorn.log
