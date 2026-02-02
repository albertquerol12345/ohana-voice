import json
import re
import subprocess
import unicodedata
from pathlib import Path

from gtts import gTTS

ROOT_DIR = Path(__file__).resolve().parents[1]
PHRASES_PATH = Path(__file__).with_name("tts_phrases.json")
OUTPUT_DIR = ROOT_DIR / "audio_samples"


def slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text)
    return text.strip("_").lower()


def main():
    if not PHRASES_PATH.exists():
        raise SystemExit(f"Missing {PHRASES_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with PHRASES_PATH.open("r", encoding="utf-8") as f:
        phrases = json.load(f)["phrases"]

    manifest = []

    for idx, entry in enumerate(phrases, start=1):
        key = entry["key"]
        phrase = entry["phrase"]
        slug = slugify(key)
        mp3_path = OUTPUT_DIR / f"{idx:02d}_{slug}.mp3"
        wav_path = OUTPUT_DIR / f"{idx:02d}_{slug}.wav"

        tts = gTTS(text=phrase, lang="es", tld="es")
        tts.save(str(mp3_path))

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(mp3_path),
                "-ar",
                "16000",
                "-ac",
                "1",
                str(wav_path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        mp3_path.unlink(missing_ok=True)

        manifest.append({"key": key, "phrase": phrase, "file": wav_path.name})

    manifest_path = OUTPUT_DIR / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump({"samples": manifest}, f, ensure_ascii=False, indent=2)

    print(f"Generated {len(manifest)} audio samples in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
