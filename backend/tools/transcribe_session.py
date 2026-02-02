import argparse
import json
import os
from datetime import datetime

from faster_whisper import WhisperModel


def resolve_default_model() -> str:
    local = os.path.expanduser("~/asr/models/whisper-large-v3-turbo-es-ct2")
    if os.path.exists(local):
        return local
    return "large-v3-turbo"


def main():
    parser = argparse.ArgumentParser(description="Transcribe a recorded session wav")
    parser.add_argument("audio", help="Path to wav file")
    parser.add_argument("--model", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compute-type", default="float16")
    parser.add_argument("--language", default="es")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--word-timestamps", action="store_true", default=True)
    parser.add_argument("--no-word-timestamps", action="store_false", dest="word_timestamps")
    parser.add_argument("--vad-filter", action="store_true", default=True)
    parser.add_argument("--no-vad-filter", action="store_false", dest="vad_filter")
    parser.add_argument("--chunk-sec", type=float, default=30.0)
    parser.add_argument("--overlap-sec", type=float, default=3.0)
    parser.add_argument("--out-txt", default=None)
    parser.add_argument("--out-json", default=None)
    args = parser.parse_args()

    model_name = args.model or resolve_default_model()
    language = None if args.language == "auto" else args.language
    chunk_length = int(args.chunk_sec) if args.chunk_sec and args.chunk_sec > 0 else None

    model = WhisperModel(model_name, device=args.device, compute_type=args.compute_type)
    segments, info = model.transcribe(
        args.audio,
        language=language,
        beam_size=args.beam_size,
        temperature=args.temperature,
        word_timestamps=args.word_timestamps,
        vad_filter=args.vad_filter,
        condition_on_previous_text=True,
        chunk_length=chunk_length,
    )
    segments = list(segments)
    text = "".join(seg.text for seg in segments).strip()

    base = os.path.splitext(args.audio)[0]
    out_txt = args.out_txt or f"{base}.txt"
    out_json = args.out_json or f"{base}.json"

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(text + "\n")

    payload = {
        "audio": args.audio,
        "model": model_name,
        "device": args.device,
        "compute_type": args.compute_type,
        "language": info.language,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "segments": [
            {
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
                "words": [
                    {"start": w.start, "end": w.end, "word": w.word}
                    for w in (seg.words or [])
                ],
            }
            for seg in segments
        ],
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"TXT: {out_txt}")
    print(f"JSON: {out_json}")


if __name__ == "__main__":
    main()
