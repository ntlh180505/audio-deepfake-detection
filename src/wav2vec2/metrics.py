import numpy as np
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d


def compute_eer(labels, scores):
    """
    Compute Equal Error Rate.

    labels:
        0 = bonafide
        1 = spoof

    scores:
        Higher score means more likely to be spoof.
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores)

    if len(np.unique(labels)) < 2:
        return None

    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)

    eer = brentq(
        lambda x: 1.0 - x - interp1d(fpr, tpr)(x),
        0.0,
        1.0
    )

    return eer * 100
