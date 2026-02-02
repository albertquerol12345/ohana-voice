#!/usr/bin/env bash
set -euo pipefail

BASE="/home/albert/ohana_voice_mvp"
ASR_VENV="${ASR_VENV:-$HOME/.venvs/asr}"
LOGS="$BASE/logs"
SESSION_DIR="$LOGS/sessions"
DURATION="${1:-60}"
PID_FILE="$LOGS/whisper_app.pid"

mkdir -p "$SESSION_DIR"

if [[ -d "$ASR_VENV/lib/python3.12/site-packages/nvidia/cublas/lib" ]]; then
  export LD_LIBRARY_PATH="$ASR_VENV/lib/python3.12/site-packages/nvidia/cublas/lib:$ASR_VENV/lib/python3.12/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"
fi

if [[ -f "$PID_FILE" ]]; then
  OLD_PID="$(cat "$PID_FILE" || true)"
  if [[ -n "${OLD_PID:-}" ]] && ps -p "$OLD_PID" >/dev/null 2>&1; then
    kill "$OLD_PID" || true
    sleep 1
  fi
fi

cd "$BASE"
echo "Iniciando sesion de $DURATION segundos..."
RECORD_SESSION_SEC="$DURATION" nohup .venv/bin/python -m backend.app > "$LOGS/whisper_app.log" 2>&1 &
echo $! > "$PID_FILE"

echo "Habla ahora. Tiempo restante: $DURATION s"
sleep "$DURATION"
sleep 2

LATEST_AUDIO="$(ls -t "$SESSION_DIR"/session_*.wav 2>/dev/null | head -n 1 || true)"
if [[ -z "${LATEST_AUDIO:-}" ]]; then
  echo "No se encontro audio en $SESSION_DIR"
  exit 1
fi

echo "Transcribiendo: $LATEST_AUDIO"
if ! .venv/bin/python -m backend.tools.transcribe_session "$LATEST_AUDIO"; then
  echo "Fallo GPU. Reintentando en CPU..."
  .venv/bin/python -m backend.tools.transcribe_session "$LATEST_AUDIO" --device cpu --compute-type int8
fi

echo "Listo. Abriendo carpeta de resultados..."
xdg-open "$SESSION_DIR" >/dev/null 2>&1 || true
