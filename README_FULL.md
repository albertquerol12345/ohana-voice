Ohana Kitchen Voice MVP

Quick start
1) Create venv + install deps
   python3 -m venv .venv
   . .venv/bin/activate
   pip install -r backend/requirements.txt

2) Download Vosk Spanish model (small)
   # From the project root:
   curl -L -o vosk-model-small-es-0.42.zip https://alphacephei.com/vosk/models/vosk-model-small-es-0.42.zip
   unzip -q vosk-model-small-es-0.42.zip

Optional (higher accuracy, larger download ~1.4GB)
   curl -L -o vosk-model-es-0.42.zip https://alphacephei.com/vosk/models/vosk-model-es-0.42.zip
   unzip -q vosk-model-es-0.42.zip
   (If present, the app will prefer this larger model by default.)

3) Run the app
   .venv/bin/python backend/server.py

Then open:
   http://localhost:8000

Whisper streaming mode (Route A - recommended)
1) Ensure deps are installed (includes faster-whisper)
   python3 -m venv .venv
   . .venv/bin/activate
   pip install -r backend/requirements.txt

2) Run the new backend
   .venv/bin/python -m backend.app

3) Open the UI
   http://localhost:8010/?ws=2700

Config via .env (copy from .env.example)
   cp .env.example .env
   # edit .env as needed

Tools
- Benchmark RTF:
  .venv/bin/python backend/tools/bench_rtf.py path/to/audio.wav --models tiny,base,small
- Build observed aliases from audio folders:
  .venv/bin/python backend/tools/build_aliases_from_audio.py path/to/dataset --output backend/keywords_observed.json

DTW voice-trained mode (single speaker)
Long-mode (low cognitive load: one audio per burger)
1) Record long audios (say each burger 25x with 1-2s gaps)
   .venv/bin/python backend/record_long.py --repeats 25

2) Segment into samples
   .venv/bin/python backend/segment_long.py

3) Train templates from long-mode samples
   .venv/bin/python backend/train_dtw.py --samples ./dtw_samples_long --skip-list ""

4) Run the DTW server
   .venv/bin/python backend/dtw_server.py

1) Record your voice samples (about 10-15 min)
   .venv/bin/python backend/record_samples.py --repeats 25

2) Audit quality (recommended)
   .venv/bin/python backend/audit_samples.py
   Review: ./dtw_audit_summary.txt

3) Re-record only bad samples (optional)
   .venv/bin/python backend/record_samples.py --plan ./dtw_retake_plan.json

4) Train templates (auto-skips flagged samples)
   .venv/bin/python backend/train_dtw.py

5) Run the DTW server
   .venv/bin/python backend/dtw_server.py

KWS mode (recommended if DTW is unstable)
1) Train KWS model (uses dtw_samples_long)
   .venv/bin/python backend/kws_train.py --samples ./dtw_samples_long

2) Run the KWS server
   .venv/bin/python backend/kws_server.py

Kaldi grammar mode (open-source, no training)
1) Ensure Vosk small model exists:
   ./vosk-model-small-es-0.42

2) Run the Kaldi grammar server:
   .venv/bin/python backend/kaldi_server.py

Open:
   http://localhost:8010/?ws=2700

Notes
- Keywords/aliases live in backend/keywords.json.
- Burger ingredients live in frontend/data/burgers.json.
- If you place the Vosk model elsewhere, set VOSK_MODEL_PATH.
- The large model does not support runtime grammar; use:
  VOSK_MODEL_PATH=./vosk-model-es-0.42 VOSK_USE_GRAMMAR=0 .venv/bin/python backend/server.py
- By default only phonetic aliases are used. To allow extra ASR fallbacks:
  VOSK_INCLUDE_ASR_ALIASES=1
- If port 8000 is busy, set a different static port:
  STATIC_PORT=8010 .venv/bin/python backend/server.py
- If you change WS port, open the UI with a query param:
  http://localhost:8010/?ws=2701
- Detection logs are written to:
  ./logs/detections.jsonl
- Hands-free mode is on by default (no mic button). To enable manual mode:
  http://localhost:8010/?handsfree=0
- VAD (voice activity detection) is disabled by default for low-volume speech.
  Set VOSK_USE_VAD=1 to enable.
  Tunables: VAD_AGGRESSIVENESS=2 (0-3), VAD_START_FRAMES=3, VAD_END_FRAMES=8, VAD_FRAME_MS=20.
  Force speech when RMS is high enough: VAD_FORCE_RMS=40.
  Max utterance length before forced flush: VAD_MAX_FRAMES=120.
- Auto gain is enabled by default for low-volume speech:
  VOSK_AUTO_GAIN=1 VOSK_TARGET_RMS=2600 VOSK_MIN_RMS=40 VOSK_MAX_GAIN=16.0
  Optional fixed gain: VOSK_INPUT_GAIN=1.2
- Low-latency partial commit (faster UI updates):
  VOSK_PARTIAL_SIMILARITY=0.84 VOSK_PARTIAL_MIN_CHARS=4 VOSK_PARTIAL_STABILITY=2
  Final match threshold: VOSK_FINAL_SIMILARITY=0.78

Whisper streaming tuning (Route A)
- Profiles: PROFILE=balanced | low_latency | robust
- Window params: WINDOW_SEC, HOP_SEC
- Thresholds: THRESH_BEST, THRESH_MARGIN
- Histéresis: HYSTERESIS_K, HYSTERESIS_WINDOW_SEC
- Cooldown: COOLDOWN_SEC
- Gate: USE_GATE, GATE_FACTOR, GATE_MIN_RMS, GATE_HANGOVER_SEC, GATE_NOISE_ALPHA
- Fallback: ENABLE_FALLBACK, MODEL_FALLBACK, FALLBACK_SCORE_BAND, FALLBACK_MARGIN_LOW, FALLBACK_MARGIN_HIGH
- Logs: LOG_PATH=logs/detections.jsonl

DTW tuning
- Samples live in ./dtw_samples
- Templates live in ./backend/dtw_templates.npz
- Adjust thresholds:
  DTW_MAX_DISTANCE=0.0 (override per-key thresholds if > 0)
  DTW_MIN_MARGIN_RATIO=0.12 (bigger = stricter)
- VAD and gain params are shared with DTW server:
  VAD_END_FRAMES=8 VAD_FRAME_MS=20 DTW_INPUT_GAIN=1.2

TTS samples (for quick validation)
   .venv/bin/python backend/generate_tts.py
   VOSK_MODEL_PATH=./vosk-model-es-0.42 VOSK_USE_GRAMMAR=0 .venv/bin/python backend/test_audio.py
