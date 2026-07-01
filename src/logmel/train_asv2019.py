import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from tqdm import tqdm

from .config import LogMelConfig
from .data import (
    load_asv2019_protocol,
    attach_audio_paths,
    LogMelDataset,
)
from .model import build_logmel_resnet18
from .metrics import compute_eer, compute_auc, find_best_threshold


def evaluate(model, loader, device):
    model.eval()

    scores = []
    labels = []

    with torch.no_grad():
        for x, y, _ in tqdm(loader, desc="Eval"):
            x = x.to(device)

            logits = model(x)
            prob_spoof = torch.softmax(logits, dim=1)[:, 1]

            scores.extend(prob_spoof.cpu().numpy())
            labels.extend(y.numpy())

    return np.array(labels), np.array(scores)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_protocol", required=True)
    parser.add_argument("--dev_protocol", required=True)
    parser.add_argument("--train_audio_dir", required=True)
    parser.add_argument("--dev_audio_dir", required=True)
    parser.add_argument("--output_dir", default="checkpoints/logmel_resnet")

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    config = LogMelConfig()

    train_df = load_asv2019_protocol(args.train_protocol)
    dev_df = load_asv2019_protocol(args.dev_protocol)

    train_df = attach_audio_paths(
        train_df,
        args.train_audio_dir,
        file_col="file",
        extensions=(".flac",),
    )

    dev_df = attach_audio_paths(
        dev_df,
        args.dev_audio_dir,
        file_col="file",
        extensions=(".flac",),
    )

    train_df = train_df[train_df["path"].notna()].reset_index(drop=True)
    dev_df = dev_df[dev_df["path"].notna()].reset_index(drop=True)

    print("Train files:", len(train_df))
    print("Dev files:", len(dev_df))

    print("\nTrain label distribution:")
    print(train_df["label_str"].value_counts())

    print("\nDev label distribution:")
    print(dev_df["label_str"].value_counts())

    train_loader = DataLoader(
        LogMelDataset(train_df, train=True, config=config),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    dev_loader = DataLoader(
        LogMelDataset(dev_df, train=False, config=config),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = build_logmel_resnet18(
        num_classes=2,
        dropout=config.dropout,
    ).to(device)

    class_counts = train_df["label"].value_counts().sort_index()

    loss_weights = torch.tensor(
        [
            class_counts[1] / class_counts[0],
            1.0,
        ],
        dtype=torch.float32,
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=loss_weights)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_eer = 1.0
    counter = 0
    history = []

    best_model_path = os.path.join(args.output_dir, "best_logmel.pth")
    best_th_path = os.path.join(args.output_dir, "best_th.npy")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for x, y, _ in loop:
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

        labels, scores = evaluate(model, dev_loader, device)

        eer = compute_eer(labels, scores)
        auc = compute_auc(labels, scores)
        th = find_best_threshold(labels, scores)

        print(
            f"Epoch {epoch + 1}: "
            f"loss={avg_loss:.6f} | "
            f"dev_eer={eer * 100:.6f}% | "
            f"auc={auc:.6f} | "
            f"th={th:.4f}"
        )

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": avg_loss,
                "dev_eer": eer,
                "dev_auc": auc,
                "threshold": th,
            }
        )

        if eer < best_eer:
            best_eer = eer
            counter = 0

            torch.save(model.state_dict(), best_model_path)
            np.save(best_th_path, th)

            print(f"Saved best model to {best_model_path}")
        else:
            counter += 1
            print(f"No improvement: {counter}/{args.patience}")

            if counter >= args.patience:
                print("Early stopping triggered.")
                break

    history_path = os.path.join(args.output_dir, "training_log.csv")
    pd.DataFrame(history).to_csv(history_path, index=False)

    print("\n===== FINAL BEST DEV EVALUATION =====")

    model.load_state_dict(torch.load(best_model_path, map_location=device))

    labels, scores = evaluate(model, dev_loader, device)
    th = float(np.load(best_th_path))
    preds = (scores > th).astype(int)

    print("Best Dev EER:", compute_eer(labels, scores) * 100)
    print("Best Dev AUC:", compute_auc(labels, scores))
    print("Best threshold:", th)
    print(classification_report(labels, preds, target_names=["bonafide", "spoof"]))


if __name__ == "__main__":
    main()
