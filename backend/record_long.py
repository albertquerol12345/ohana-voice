import argparse
import json
import time
import wave
from pathlib import Path

import sounddevice as sd

SAMPLE_RATE = 16000
ROOT_DIR = Path(__file__).resolve().parents[1]
KEYWORDS_PATH = Path(__file__).with_name("keywords.json")


def load_keywords():
    with KEYWORDS_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    keys = []
    prompts = {}
    for key, aliases in data.items():
        keys.append(key)
        prompt = key
        if isinstance(aliases, dict):
            phonetic = aliases.get("phonetic") or []
            if phonetic:
                prompt = phonetic[0]
        elif aliases:
            prompt = aliases[0]
        prompts[key] = prompt
    return keys, prompts


def parse_keys(value: str | None, all_keys: list[str]) -> list[str]:
    if not value:
        return all_keys
    wanted = [item.strip() for item in value.split(",") if item.strip()]
    selected = []
    for key in all_keys:
        if key in wanted:
            selected.append(key)
    missing = [item for item in wanted if item not in all_keys]
    if missing:
        raise SystemExit(f"Unknown keys: {', '.join(missing)}")
    return selected


def write_wav(path: Path, samples):
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(samples.tobytes())


def main():
    parser = argparse.ArgumentParser(description="Record one long audio per keyword.")
    parser.add_argument("--out-dir", default=str(ROOT_DIR / "dtw_long"))
    parser.add_argument("--keys", default=None, help="Comma-separated list of keys")
    parser.add_argument("--repeats", type=int, default=25)
    parser.add_argument("--seconds", type=float, default=0.0, help="Override total seconds")
    parser.add_argument("--per-repeat", type=float, default=2.2)
    parser.add_argument("--countdown", type=int, default=5)
    args = parser.parse_args()

    keys, prompts = load_keywords()
    selected_keys = parse_keys(args.keys, keys)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for key in selected_keys:
        prompt = prompts[key]
        total_seconds = args.seconds if args.seconds > 0 else args.repeats * args.per_repeat
        total_seconds = max(total_seconds, args.repeats * 1.2)
        print(f"\n== {key} ==")
        print(f"Di: {prompt}")
        print(f"Repite {args.repeats} veces, con pausas de 1-2s.")
        print(f"Duracion total: ~{int(total_seconds)}s")
        input("Pulsa Enter para empezar a grabar...")
        for i in range(args.countdown, 0, -1):
            print(f"Empieza en {i}...")
            time.sleep(1)
        print("Grabando...")
        audio = sd.rec(
            int(total_seconds * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
        )
        sd.wait()
        path = out_dir / f"{key}.wav"
        write_wav(path, audio)
        print(f"Guardado: {path}")


if __name__ == "__main__":
    main()
