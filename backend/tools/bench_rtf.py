import argparse
import time

import soundfile as sf
from faster_whisper import WhisperModel


def load_audio(path: str):
    data, sr = sf.read(path)
    if data.ndim > 1:
        data = data[:, 0]
    return data, sr


def run_benchmark(path: str, model_name: str, device: str, compute_type: str, language: str | None):
    audio, sr = load_audio(path)
    duration = len(audio) / sr if sr else 0.0
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    start = time.time()
    segments, _ = model.transcribe(
        audio,
        beam_size=1,
        temperature=0.0,
        language=language,
        vad_filter=False,
        condition_on_previous_text=False,
    )
    _ = list(segments)
    elapsed = time.time() - start
    rtf = elapsed / duration if duration else 0.0
    return duration, elapsed, rtf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", help="Path to a wav file for benchmark")
    parser.add_argument(
        "--models",
        default="tiny,base,small",
        help="Comma-separated model names (default: tiny,base,small)",
    )
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--compute-type", default="int8_float16", help="CTranslate2 compute type")
    parser.add_argument("--language", default="auto", help="es, en, or auto")
    args = parser.parse_args()

    language = None if args.language == "auto" else args.language
    models = [item.strip() for item in args.models.split(",") if item.strip()]
    print(f"Audio: {args.audio}")
    print(f"Device: {args.device} | Compute: {args.compute_type} | Language: {args.language}")
    print("")
    print("model\tseconds\telapsed\trtf")
    for model in models:
        duration, elapsed, rtf = run_benchmark(
            args.audio, model, args.device, args.compute_type, language
        )
        print(f"{model}\t{duration:.2f}\t{elapsed:.2f}\t{rtf:.2f}")


if __name__ == "__main__":
    main()
