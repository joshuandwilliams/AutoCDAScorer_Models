"""Evaluation metrics for ordinal CDA score prediction (integer labels, e.g. 0-6)."""

import numpy as np
from sklearn.metrics import cohen_kappa_score


def near_miss_accuracy(y_true, y_pred, tolerance: int = 1) -> float:
    """Fraction of predictions within ``tolerance`` of the true ordinal label.

    With the default tolerance of 1 this is the existing "near-miss" metric: a
    prediction one score either side of the truth counts as correct.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.size == 0:
        return 0.0
    return float(np.mean(np.abs(y_true - y_pred) <= tolerance))


def quadratic_weighted_kappa(y_true, y_pred, labels=None) -> float:
    """Quadratic Weighted Kappa (QWK): agreement on an ordinal scale.

    Larger disagreements are penalised quadratically, so predicting 5 for a true 6
    costs far less than predicting 0. This is the standard metric for ordinal
    human-graded scores and is directly comparable to inter-rater agreement between
    human scorers. Pass ``labels`` (e.g. ``[0, 1, ..., 6]``) so the score is computed
    over the full scale even when a fold's predictions miss some classes.
    """
    return float(cohen_kappa_score(y_true, y_pred, labels=labels, weights="quadratic"))
