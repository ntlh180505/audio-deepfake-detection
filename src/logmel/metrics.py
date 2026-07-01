import numpy as np
from sklearn.metrics import roc_curve, f1_score, roc_auc_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d


def compute_eer(y_true, y_score):
    """
    Compute Equal Error Rate.

    y_true:
        0 = bonafide / genuine
        1 = spoof / fake

    y_score:
        spoof probability or spoof score
    """
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)

    eer = brentq(
        lambda x: 1.0 - x - interp1d(fpr, tpr)(x),
        0.0,
        1.0,
    )

    return eer


def compute_auc(y_true, y_score):
    return roc_auc_score(y_true, y_score)


def find_best_threshold(y_true, y_score):
    """
    Find threshold maximizing macro F1 on development set.
    This threshold is used only for classification report.
    EER itself is threshold-independent.
    """
    best_th = 0.5
    best_f1 = 0.0

    for th in np.linspace(0, 1, 100):
        pred = (y_score > th).astype(int)
        f1 = f1_score(y_true, pred, average="macro")

        if f1 > best_f1:
            best_f1 = f1
            best_th = th

    return best_th
