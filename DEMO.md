# Ohana Voice — Demo Guide

See voice ordering in action.

---

## 🎯 What You'll See

**Before:** Staff typing orders manually during rush  
**After:** Voice command → Validation UI → Kitchen display

Example flow:
```
Staff: "Doble cheese con bacon y sin cebolla"
System: [Detects: Doble Cheese Burger + Bacon - Onion]
Staff: [Clicks "Send to Kitchen" after visual verification]
Kitchen: [Receives formatted ticket]
```

---

## 🚀 Option 1: Browser Demo (30 seconds)

No installation, no microphone setup.

```bash
git clone https://github.com/albertquerol12345/ohana-voice.git
cd ohana_voice_mvp
python -m http.server 8080 --directory frontend
```

Open: `http://localhost:8080/?demo=1`

### What works in demo mode:
- ✅ Browse 19-item catalog
- ✅ See ingredient modifiers (+bacon, -onion, etc.)
- ✅ Simulate order flow (buttons instead of voice)
- ✅ Kitchen display preview

---

## 🚀 Option 2: Full Voice Demo (5 minutes)

Requires microphone and Python setup.

```bash
# Setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt

# Download model
curl -L -o vosk-model-small-es-0.42.zip https://alphacephei.com/vosk/models/vosk-model-small-es-0.42.zip
unzip -q vosk-model-small-es-0.42.zip

# Run
.venv/bin/python backend/server.py
```

Open: `http://localhost:8000`

### Try these commands:
| Say... | Result |
|--------|--------|
| "Doble cheese" | Adds Double Cheese Burger |
| "Con bacon" | Adds bacon modifier |
| "Sin cebolla" | Removes onion |
| "Coca cola grande" | Adds Large Coke |
| "Enviar pedido" | Sends to validation queue |

---

## 📊 Demo Catalog (19 items)

Location: `frontend/data/burgers.json`

| Category | Items |
|----------|-------|
| Burgers | Doble Cheese, Classic Burger, BBQ Bacon, Chicken Crispy |
| Sides | Fries (S/M/L), Onion Rings, Nuggets (6/9/12) |
| Drinks | Coca-Cola, Fanta, Sprite (S/M/L) |
| Desserts | McFlurry Oreo, Apple Pie |

All items support:
- Add/remove ingredients
- Size selection
- Combo bundling

---

## 🖼️ Visual Walkthrough

| Screen | Purpose |
|--------|---------|
| **Menu Grid** | Visual catalog with images and prices |
| **Voice Status** | Live transcription and confidence indicator |
| **Order Builder** | Current items with modifiers |
| **Validation Modal** | Final check before sending to kitchen |
| **Kitchen Display** | Order queue for kitchen staff |

---

## 🎓 Voice Training (Optional)

For personalized recognition (DTW mode):

```bash
# 1. Record 25 samples of each item (~15 min)
.venv/bin/python backend/record_samples.py --repeats 25

# 2. Train templates
.venv/bin/python backend/train_dtw.py

# 3. Run personalized server
.venv/bin/python backend/dtw_server.py
```

Now the system recognizes YOUR voice saying "Doble cheese" specifically.

---

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| Microphone not detected | Use `?demo=1` mode or check browser permissions |
| Recognition poor | Reduce background noise or try Whisper mode |
| False triggers | Adjust `VOSK_FINAL_SIMILARITY` threshold |
| Latency high | Enable `VOSK_PARTIAL_SIMILARITY` for faster UI updates |

---

## 💡 Key Design Decisions

1. **Human-in-the-loop** — Voice recognition is never 100%, so every order requires visual confirmation
2. **Offline-first** — Works without internet (crucial for food trucks, basement kitchens)
3. **Grammar-constrained** — Limited vocabulary = higher accuracy than general ASR
4. **Hands-free mode** — No button to hold; just speak and system listens

---

## 📁 Files to Explore

- `frontend/data/burgers.json` — Catalog structure
- `backend/keywords.json` — Voice command aliases
- `logs/detections.jsonl` — Recognition logs
