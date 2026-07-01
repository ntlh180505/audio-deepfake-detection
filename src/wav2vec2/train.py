import os
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .w2v2_model import Wav2Vec2DeepfakeDetector
from .data_utils import WavDataset, load_asv2019_protocol
from .metrics import compute_eer


def evaluate(model, loader, device):
    model.eval()

    scores = []
    labels = []

    with torch.no_grad():
        for batch in loader:
            x, y = batch[0], batch[1]

            x = x.to(device)
            y = y.to(device)

            logits = model(x)

            prob_spoof = torch.softmax(logits, dim=1)[:, 1]

            scores.extend(prob_spoof.detach().cpu().numpy().tolist())
            labels.extend(y.detach().cpu().numpy().tolist())

    eer = compute_eer(labels, scores)

    return eer


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_protocol", required=True)
    parser.add_argument("--dev_protocol", required=True)
    parser.add_argument("--train_audio_dir", required=True)
    parser.add_argument("--dev_audio_dir", required=True)

    parser.add_argument("--save_path", required=True)

    parser.add_argument("--aug_level", default="original", choices=["none", "original", "light"])

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=2)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Device:", device)
    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    train_df = load_asv2019_protocol(args.train_protocol, args.train_audio_dir)
    dev_df = load_asv2019_protocol(args.dev_protocol, args.dev_audio_dir)

    print("Train samples:", len(train_df))
    print("Dev samples:", len(dev_df))

    print("\nTrain label count:")
    print(train_df["label_str"].value_counts())

    print("\nDev label count:")
    print(dev_df["label_str"].value_counts())

    train_loader = DataLoader(
        WavDataset(train_df, train=True, aug_level=args.aug_level),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    dev_loader = DataLoader(
        WavDataset(dev_df, train=False, aug_level="none"),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = Wav2Vec2DeepfakeDetector().to(device)

    class_counts = train_df["label"].value_counts().sort_index()
    counts = torch.tensor(class_counts.values, dtype=torch.float)

    weights = 1.0 / counts
    weights = weights / weights.sum() * 2.0

    print("Class counts:", class_counts.to_dict())
    print("Class weights:", weights.tolist())

    criterion = nn.CrossEntropyLoss(weight=weights.to(device))

    optimizer = torch.optim.AdamW(
        [
            {"params": model.fc.parameters(), "lr": 3e-4},
            {"params": model.pool.parameters(), "lr": 3e-4},
            {"params": [model.layer_weights], "lr": 1e-3},
            {"params": model.wav2vec.encoder.layers[-1].parameters(), "lr": 1e-5},
            {"params": model.wav2vec.encoder.layers[-2].parameters(), "lr": 1e-5},
        ],
        weight_decay=1e-5,
    )

    best_eer = 1e9
    best_epoch = 0
    no_improve = 0

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()

        total_loss = 0.0

        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")

        for batch in loop:
            x, y = batch[0], batch[1]

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            logits = model(x)
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)

        dev_eer = evaluate(model, dev_loader, device)

        print(f"\nEpoch {epoch}")
        print(f"Train loss: {avg_loss:.6f}")
        print(f"Dev EER: {dev_eer:.6f}%")

        if dev_eer is not None and dev_eer < best_eer:
            best_eer = dev_eer
            best_epoch = epoch
            no_improve = 0

            torch.save(model.state_dict(), args.save_path)

            print(f"Saved best model: {args.save_path}")
            print(f"Best epoch: {best_epoch}, Best EER: {best_eer:.6f}%")
        else:
            no_improve += 1
            print(f"No improvement: {no_improve}/{args.patience}")

        if no_improve >= args.patience:
            print("Early stopping.")
            break

    print("\nTraining finished.")
    print(f"Best epoch: {best_epoch}")
    print(f"Best dev EER: {best_eer:.6f}%")


if __name__ == "__main__":
    main()
