import argparse
import csv
import json
import os
from collections import Counter, defaultdict

from faster_whisper import WhisperModel

from backend.matcher import normalize_text


def load_mapping(input_path: str, mapping_csv: str | None):
    items = []
    if mapping_csv:
        with open(mapping_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                audio = row.get("path") or row.get("audio")
                key = row.get("key") or row.get("burger")
                if audio and key:
                    items.append((audio, key))
        return items

    for root, _, files in os.walk(input_path):
        for name in files:
            if not name.lower().endswith((".wav", ".flac", ".mp3", ".m4a")):
                continue
            key = os.path.basename(root)
            items.append((os.path.join(root, name), key))
    return items


def transcribe(model: WhisperModel, audio_path: str, language: str | None):
    segments, _ = model.transcribe(
        audio_path,
        beam_size=1,
        temperature=0.0,
        language=language,
        vad_filter=False,
        condition_on_previous_text=False,
    )
    return " ".join(seg.text.strip() for seg in segments).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Folder with subfolders per burger or audio root")
    parser.add_argument("--mapping-csv", help="CSV with columns: path,key", default=None)
    parser.add_argument("--output", default="backend/keywords_observed.json")
    parser.add_argument("--model", default="base")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compute-type", default="int8_float16")
    parser.add_argument("--language", default="auto")
    parser.add_argument("--min-count", type=int, default=2)
    parser.add_argument("--top-n", type=int, default=8)
    args = parser.parse_args()

    language = None if args.language == "auto" else args.language
    items = load_mapping(args.input, args.mapping_csv)
    if not items:
        print("No audio files found.")
        return

    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
    counts = defaultdict(Counter)

    for audio_path, key in items:
        text = transcribe(model, audio_path, language)
        norm = normalize_text(text)
        if not norm:
            continue
        counts[key][norm] += 1

    observed = {}
    for key, counter in counts.items():
        filtered = [alias for alias, count in counter.most_common() if count >= args.min_count]
        observed[key] = filtered[: args.top_n]

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(observed, f, ensure_ascii=False, indent=2)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
