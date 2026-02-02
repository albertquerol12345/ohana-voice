import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from kws_utils import SAMPLE_RATE, energy, load_audio, log_mel, make_melspec, pad_or_trim

ROOT_DIR = Path(__file__).resolve().parents[1]


class KWSNet(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


def collect_samples(samples_dir: Path):
    items = []
    labels = sorted([p.name for p in samples_dir.iterdir() if p.is_dir()])
    for label in labels:
        for wav in sorted((samples_dir / label).glob("*.wav")):
            items.append((wav, label))
    return items, labels


def split_data(items: list, seed: int = 7, val_ratio: float = 0.2):
    random.Random(seed).shuffle(items)
    cut = int(len(items) * (1 - val_ratio))
    return items[:cut], items[cut:]


def build_batches(items, label_to_idx, mels, window_samples, device):
    x_list = []
    y_list = []
    for path, label in items:
        wav = load_audio(path)
        wav = pad_or_trim(wav, window_samples)
        mel = mels(wav)
        mel = log_mel(mel)
        mel = mel.unsqueeze(0)
        x_list.append(mel)
        y_list.append(label_to_idx[label])
    x = torch.stack(x_list).to(device)
    y = torch.tensor(y_list, dtype=torch.long).to(device)
    return x, y


def main():
    parser = argparse.ArgumentParser(description="Train KWS model.")
    parser.add_argument("--samples", default=str(ROOT_DIR / "dtw_samples_long"))
    parser.add_argument("--out", default=str(Path(__file__).with_name("kws_model.pt")))
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--window-seconds", type=float, default=1.0)
    parser.add_argument("--min-energy", type=float, default=0.003)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    samples_dir = Path(args.samples)
    if not samples_dir.exists():
        raise SystemExit(f"Missing samples dir: {samples_dir}")

    items, labels = collect_samples(samples_dir)
    items = [(p, l) for p, l in items if energy(load_audio(p)) >= args.min_energy]
    if not items:
        raise SystemExit("No usable samples found.")

    train_items, val_items = split_data(items, seed=args.seed)
    label_to_idx = {label: idx for idx, label in enumerate(labels)}

    device = torch.device(args.device)
    window_samples = int(args.window_seconds * SAMPLE_RATE)
    mels = make_melspec()

    x_train, y_train = build_batches(train_items, label_to_idx, mels, window_samples, device)
    x_val, y_val = build_batches(val_items, label_to_idx, mels, window_samples, device)

    model = KWSNet(len(labels)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        permutation = torch.randperm(x_train.size(0))
        total_loss = 0.0
        for i in range(0, x_train.size(0), args.batch_size):
            idx = permutation[i : i + args.batch_size]
            batch_x = x_train[idx]
            batch_y = y_train[idx]
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        with torch.no_grad():
            logits = model(x_val)
            preds = torch.argmax(logits, dim=1)
            acc = (preds == y_val).float().mean().item()
        print(f"Epoch {epoch:02d} loss={total_loss:.3f} val_acc={acc:.3f}")

    payload = {
        "state_dict": model.state_dict(),
        "labels": labels,
        "window_seconds": args.window_seconds,
        "sample_rate": SAMPLE_RATE,
    }
    torch.save(payload, args.out)
    print(f"Saved model to {args.out}")


if __name__ == "__main__":
    main()
