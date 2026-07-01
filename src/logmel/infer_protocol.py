import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from tqdm import tqdm

from .config import LogMelConfig
from .data import (
    load_asv2019_protocol,
    load_asv2021_la_protocol,
    load_asv2021_df_protocol,
    load_add2023_protocol,
    attach_audio_paths,
    LogMelDataset,
)
from .model import build_logmel_resnet18
from .metrics import compute_eer, compute_auc


def load_protocol(protocol_type, protocol_path):
    if protocol_type == "asv2019":
        df = load_asv2019_protocol(protocol_path)
        df["subset"] = "eval"
        return df

    if protocol_type == "asv2021_la":
        return load_asv2021_la_protocol(protocol_path)

    if protocol_type == "asv2021_df":
        return load_asv2021_df_protocol(protocol_path)

    if protocol_type == "add2023":
        df = load_add2023_protocol(protocol_path)
        df["subset"] = "eval"
        return df

    raise ValueError(f"Unsupported protocol type: {protocol_type}")


def evaluate(model, loader, device):
    model.eval()

    scores = []
    labels = []
    files = []

    with torch.no_grad():
        for x, y, file_ids in tqdm(loader, desc="Inference"):
            x = x.to(device)

            logits = model(x)
            prob_spoof = torch.softmax(logits, dim=1)[:, 1]

            scores.extend(prob_spoof.cpu().numpy())
            labels.extend(y.numpy())
            files.extend(file_ids)

    return np.array(labels), np.array(scores), files


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--protocol_type",
        required=True,
        choices=["asv2019", "asv2021_la", "asv2021_df", "add2023"],
    )

    parser.add_argument("--protocol_path", required=True)
    parser.add_argument("--audio_dirs", required=True, nargs="+")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--threshold_path", default=None)
    parser.add_argument("--score_path", required=True)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.score_path), exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    config = LogMelConfig()

    df = load_protocol(args.protocol_type, args.protocol_path)

    extensions = (".wav",) if args.protocol_type == "add2023" else (".flac", ".wav")

    df = attach_audio_paths(
        df,
        args.audio_dirs,
        file_col="file",
        extensions=extensions,
    )

    print("Protocol trials:", len(df))
    print("Path found:", df["path"].notna().sum())
    print("Path missing:", df["path"].isna().sum())

    missing = df[df["path"].isna()]
    if len(missing) > 0:
        missing_path = args.score_path.replace(".csv", "_missing_paths.csv")
        missing.to_csv(missing_path, index=False)
        print("Saved missing paths to:", missing_path)

    df = df[df["path"].notna()].reset_index(drop=True)

    print("\nLabel distribution:")
    print(df["label_str"].value_counts())

    if "subset" in df.columns:
        print("\nSubset distribution:")
        print(df["subset"].value_counts())

    loader = DataLoader(
        LogMelDataset(df, train=False, config=config),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = build_logmel_resnet18(
        num_classes=2,
        dropout=config.dropout,
    ).to(device)

    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    labels, scores, files = evaluate(model, loader, device)

    out_df = pd.DataFrame(
        {
            "file": files,
            "label": labels,
            "score_spoof": scores,
        }
    )

    if "label_str" in df.columns:
        out_df["label_str"] = df["label_str"].values

    if "subset" in df.columns:
        out_df["subset"] = df["subset"].values

    out_df.to_csv(args.score_path, index=False)
    print("\nSaved scores to:", args.score_path)

    eer = compute_eer(labels, scores)
    auc = compute_auc(labels, scores)

    print("\n========== RESULTS ==========")
    print(f"EER: {eer * 100:.6f}%")
    print(f"AUC: {auc:.6f}")
    print(f"N: {len(labels)}")

    if "subset" in out_df.columns:
        print("\n========== EER BY SUBSET ==========")

        for subset in sorted(out_df["subset"].unique()):
            sub = out_df[out_df["subset"] == subset]

            if sub["label"].nunique() == 2:
                sub_eer = compute_eer(sub["label"].values, sub["score_spoof"].values)
                sub_auc = compute_auc(sub["label"].values, sub["score_spoof"].values)

                print(
                    f"{subset}: "
                    f"EER={sub_eer * 100:.6f}% | "
                    f"AUC={sub_auc:.6f} | "
                    f"N={len(sub)}"
                )

    if args.threshold_path is not None and os.path.exists(args.threshold_path):
        th = float(np.load(args.threshold_path))
        preds = (scores > th).astype(int)

        print("\n========== CLASSIFICATION REPORT ==========")
        print("Threshold:", th)
        print(
            classification_report(
                labels,
                preds,
                target_names=["bonafide/genuine", "spoof/fake"],
            )
        )


if __name__ == "__main__":
    main()
