"""
Compute EER for ADD2023 Track 1.2 R1 using XLS-R + SLSClassifier scores.

Score file format:
    <utt_id.wav> <score>

Label file format:
    <utt_id.wav> <label>

Labels:
    genuine -> bonafide
    fake    -> spoof
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve


def compute_eer(bonafide_scores: np.ndarray, spoof_scores: np.ndarray):
    y_true = np.concatenate(
        [
            np.ones(len(bonafide_scores), dtype=np.int32),
            np.zeros(len(spoof_scores), dtype=np.int32),
        ]
    )
    y_score = np.concatenate([bonafide_scores, spoof_scores])

    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    fnr = 1.0 - tpr

    eer_index = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[eer_index] + fnr[eer_index]) / 2.0
    threshold = thresholds[eer_index]

    return eer, threshold


def load_and_compute(score_file: Path, label_file: Path) -> None:
    if not score_file.exists():
        raise FileNotFoundError(f"Score file not found: {score_file}")

    if not label_file.exists():
        raise FileNotFoundError(f"Label file not found: {label_file}")

    scores = pd.read_csv(
        score_file,
        sep=r"\s+",
        header=None,
        names=["utt_id", "score"],
        engine="python",
    )

    labels = pd.read_csv(
        label_file,
        sep=r"\s+",
        header=None,
        names=["utt_id", "label"],
        engine="python",
    )

    df = scores.merge(labels, on="utt_id", how="inner")

    print(f"Score file: {score_file}")
    print(f"Label file: {label_file}")
    print(f"Number of scores: {len(scores)}")
    print(f"Number of labels: {len(labels)}")
    print(f"Number of matched trials: {len(df)}")
    print()
    print("Label distribution:")
    print(df["label"].value_counts())

    bonafide_scores = df[df["label"] == "genuine"]["score"].to_numpy()
    spoof_scores = df[df["label"] == "fake"]["score"].to_numpy()

    eer, threshold = compute_eer(bonafide_scores, spoof_scores)

    print()
    print(f"EER: {eer * 100:.4f}%")
    print(f"Threshold: {threshold:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--score_file",
        required=True,
        type=Path,
        help="Path to ADD2023 score file",
    )
    parser.add_argument(
        "--label_file",
        required=True,
        type=Path,
        help="Path to ADD2023 label.txt",
    )

    args = parser.parse_args()
    load_and_compute(args.score_file, args.label_file)


if __name__ == "__main__":
    main()
