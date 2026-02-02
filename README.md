# Ohana Voice (Portfolio Copy)

![Ohana Voice preview](assets/preview.gif)

Offline-first voice ordering MVP for a kitchen POS flow. Speech is processed locally and validated before sending to the kitchen (human-in-the-loop).

## Highlights
- Offline ASR (Vosk / Kaldi grammar mode)
- Optional Whisper streaming route
- Low-latency UI with confidence and partial updates
- Designed for noisy QSR environments

## Quick start (Vosk grammar mode)
```bash
cd OHANA_VOICE_PUBLIC
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt

# Download Vosk Spanish model (small)
curl -L -o vosk-model-small-es-0.42.zip https://alphacephei.com/vosk/models/vosk-model-small-es-0.42.zip
unzip -q vosk-model-small-es-0.42.zip

# Run server
.venv/bin/python backend/kaldi_server.py
```
Open: http://localhost:8010/?ws=2700

## UI demo (no backend)
```bash
cd OHANA_VOICE_PUBLIC/frontend
python3 -m http.server 8010
```
Open: http://localhost:8010/?demo=1

**Expected output (UI sample):**
![Ohana Voice UI](assets/preview.gif)

## Whisper streaming route (optional)
```bash
.venv/bin/python -m backend.app
```
Open: http://localhost:8010/?ws=2700

## Repo structure (simplified)
- `backend/` ASR server, streaming pipeline, matcher, metrics
- `frontend/` lightweight UI (static)
- `backend/keywords.json` command vocabulary

## Notes
- Large models, logs, and training data are removed in this portfolio copy.
- Full technical README is in `README_FULL.md`.
