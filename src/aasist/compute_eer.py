import argparse
import pandas as pd
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d


def compute_eer(y_true, y_score):
    """
    Compute Equal Error Rate.

    Parameters
    ----------
    y_true : array-like
        Ground truth labels. Bonafide should be 1 and spoof should be 0.
    y_score : array-like
        Prediction score. Higher score should indicate bonafide.

    Returns
    -------
    eer : float
        Equal Error Rate in percentage.
    threshold : float
        Decision threshold at EER.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    threshold = float(interp1d(fpr, thresholds)(eer))

    return eer * 100.0, threshold


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--score_file", required=True)
    parser.add_argument("--label_col", default="label")
    parser.add_argument("--score_col", default="score")
    parser.add_argument("--bonafide_label", default="bonafide")
    args = parser.parse_args()

    df = pd.read_csv(args.score_file, sep=r"\s+", engine="python")

    y_true = (df[args.label_col] == args.bonafide_label).astype(int)
    y_score = df[args.score_col].astype(float)

    eer, threshold = compute_eer(y_true, y_score)

    print(f"EER: {eer:.4f}%")
    print(f"Threshold: {threshold:.6f}")


if __name__ == "__main__":
    main()
