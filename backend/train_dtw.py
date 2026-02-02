import argparse
import json
from pathlib import Path

import numpy as np

from dtw_utils import dtw_distance, extract_features, load_wav

ROOT_DIR = Path(__file__).resolve().parents[1]


def collect_samples(sample_dir: Path, keys: list[str] | None):
    samples = {}
    for key_dir in sorted(sample_dir.iterdir()):
        if not key_dir.is_dir():
            continue
        key = key_dir.name
        if keys and key not in keys:
            continue
        feats = []
        for wav_path in sorted(key_dir.glob("*.wav")):
            audio = load_wav(str(wav_path))
            feat = extract_features(audio)
            if feat.size == 0:
                continue
            feats.append(feat)
        if feats:
            samples[key] = feats
    return samples


def pairwise_distances(features: list[np.ndarray]) -> np.ndarray:
    count = len(features)
    dist = np.zeros((count, count), dtype=np.float32)
    for i in range(count):
        for j in range(i + 1, count):
            d = dtw_distance(features[i], features[j])
            dist[i, j] = d
            dist[j, i] = d
    return dist


def find_medoid_indices(features: list[np.ndarray], k: int) -> list[int]:
    if len(features) <= 1:
        return [0]
    dist = pairwise_distances(features)
    scores = dist.mean(axis=1)
    order = list(np.argsort(scores))
    return order[: max(1, min(k, len(order)))]


def compute_threshold(distances: list[float]) -> float:
    if not distances:
        return 1.0
    if len(distances) == 1:
        return max(distances[0] * 1.2, 0.35)
    p90 = float(np.percentile(distances, 90))
    return max(p90 * 1.1, 0.35)


def main():
    parser = argparse.ArgumentParser(description="Train DTW templates.")
    parser.add_argument("--samples", default=str(ROOT_DIR / "dtw_samples"))
    parser.add_argument("--out", default=str(Path(__file__).with_name("dtw_templates.npz")))
    parser.add_argument("--keys", default=None, help="Comma-separated list of keys")
    parser.add_argument("--skip-list", default=str(ROOT_DIR / "dtw_retake.txt"))
    parser.add_argument("--templates-per-key", type=int, default=3)
    args = parser.parse_args()

    sample_dir = Path(args.samples)
    if not sample_dir.exists():
        raise SystemExit(f"Missing samples dir: {sample_dir}")
    keys = None
    if args.keys:
        keys = [item.strip() for item in args.keys.split(",") if item.strip()]

    skip_paths = set()
    skip_path = Path(args.skip_list) if args.skip_list else None
    if skip_path and skip_path.exists():
        for line in skip_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            path = line.split("\t", 1)[0]
            skip_paths.add(path)

    def collect_samples_filtered():
        data = {}
        for key_dir in sorted(sample_dir.iterdir()):
            if not key_dir.is_dir():
                continue
            key = key_dir.name
            if keys and key not in keys:
                continue
            feats = []
            for wav_path in sorted(key_dir.glob("*.wav")):
                if str(wav_path) in skip_paths:
                    continue
                audio = load_wav(str(wav_path))
                feat = extract_features(audio)
                if feat.size == 0:
                    continue
                feats.append(feat)
            if feats:
                data[key] = feats
        return data

    samples = collect_samples_filtered()
    if not samples:
        raise SystemExit("No samples found.")

    labels = []
    templates = []
    thresholds = []
    meta = {
        "samples_dir": str(sample_dir),
        "counts": {},
        "templates_per_key": args.templates_per_key,
    }

    for key, feats in sorted(samples.items()):
        meta["counts"][key] = len(feats)
        medoid_indices = find_medoid_indices(feats, args.templates_per_key)
        for medoid_idx in medoid_indices:
            medoid = feats[medoid_idx]
            dists = [dtw_distance(feat, medoid) for feat in feats]
            threshold = compute_threshold(dists)
            labels.append(key)
            templates.append(medoid)
            thresholds.append(threshold)
        recent = thresholds[-len(medoid_indices) :]
        avg_thr = float(np.mean(recent)) if recent else 0.0
        print(f"{key}: samples={len(feats)} templates={len(medoid_indices)} threshold≈{avg_thr:.3f}")

    out_path = Path(args.out)
    np.savez_compressed(
        out_path,
        labels=np.array(labels, dtype=object),
        templates=np.array(templates, dtype=object),
        thresholds=np.array(thresholds, dtype=np.float32),
        meta=json.dumps(meta),
    )
    print(f"Saved templates to {out_path}")


if __name__ == "__main__":
    main()
