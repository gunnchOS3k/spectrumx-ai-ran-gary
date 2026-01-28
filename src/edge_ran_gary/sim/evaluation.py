"""
Evaluation utilities for detection models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class Evaluator:
    """
    Compute evaluation metrics for binary occupancy detection.
    """

    n_bins: int = 10

    def evaluate(
        self,
        y_true: Iterable[int],
        y_pred: Optional[Iterable[int]] = None,
        y_proba: Optional[Iterable[float]] = None,
    ) -> dict:
        y_true_arr = np.asarray(list(y_true)).astype(int)
        y_pred_arr = None if y_pred is None else np.asarray(list(y_pred)).astype(int)
        y_proba_arr = None if y_proba is None else np.asarray(list(y_proba)).astype(float)

        if y_pred_arr is None and y_proba_arr is None:
            raise ValueError("Provide y_pred or y_proba (or both) to evaluate.")
        if y_pred_arr is None and y_proba_arr is not None:
            y_pred_arr = (y_proba_arr >= 0.5).astype(int)

        results = {
            "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
            "f1": float(f1_score(y_true_arr, y_pred_arr)),
            "precision": float(precision_score(y_true_arr, y_pred_arr)),
            "recall": float(recall_score(y_true_arr, y_pred_arr)),
        }

        if y_proba_arr is not None:
            results["roc_auc"] = float(roc_auc_score(y_true_arr, y_proba_arr))
            results["pr_auc"] = float(average_precision_score(y_true_arr, y_proba_arr))
            results["brier"] = float(brier_score_loss(y_true_arr, y_proba_arr))
            results["ece"] = float(self._expected_calibration_error(y_true_arr, y_proba_arr))
        else:
            results["roc_auc"] = None
            results["pr_auc"] = None
            results["brier"] = None
            results["ece"] = None

        return results

    def _expected_calibration_error(self, y_true: np.ndarray, y_proba: np.ndarray) -> float:
        if y_proba.size == 0:
            return 0.0
        bins = np.linspace(0.0, 1.0, self.n_bins + 1)
        bin_ids = np.digitize(y_proba, bins) - 1
        ece = 0.0
        for i in range(self.n_bins):
            mask = bin_ids == i
            if not np.any(mask):
                continue
            bin_confidence = np.mean(y_proba[mask])
            bin_accuracy = np.mean(y_true[mask])
            ece += (np.sum(mask) / y_proba.size) * abs(bin_accuracy - bin_confidence)
        return float(ece)

