# Ohana Voice — Demo Guide

Demo visual de comandas por voz (QSR) con **catálogo real** de 19 burgers.

---

## ⚡ Opción 1: Demo en navegador (UI-only)

```bash
git clone https://github.com/albertquerol12345/ohana-voice.git
cd ohana-voice
python -m http.server 8080 --directory frontend
```

Abre: `http://localhost:8080/?demo=1`

✅ Navegar catálogo (19 burgers)  
✅ Simular flujo de pedido con botones  
❗ *No hay reconocimiento de voz en el navegador; es solo UI*

### Comandos de ejemplo (demo)
- "una Big Ohana"
- "una Boo con cheddar"
- "una Wazoski"
- "una Vegana"
- "enviar pedido"

---

## 🎙️ Opción 2: Demo con voz (backend local)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt

# Descargar modelo Vosk (español)
curl -L -o vosk-model-small-es-0.42.zip https://alphacephei.com/vosk/models/vosk-model-small-es-0.42.zip
unzip -q vosk-model-small-es-0.42.zip

# Ejecutar servidor
.venv/bin/python backend/server.py
```

Abre: `http://localhost:8000`

---

## 📋 Catálogo real (19 burgers)

Ubicación: `frontend/data/burgers.json`

Nombres (display):
- Big Ohana, Boo, Cobra, Dormilona, Feliz, Fiona, Gruñona, Lilo, Monumental, Mudita,
  Mulan, Ohana, Sabia, Stich, Sully, Tartufo, Tímida, Vegana, Wazoski

---

## 🧪 Notas

- El demo usa **datos reales del catálogo**.
- La parte de voz requiere backend local (Vosk/Kaldi). 
- Si no quieres instalar nada, usa el modo demo UI en el navegador.
