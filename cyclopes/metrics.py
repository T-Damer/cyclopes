from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


THRESHOLD = 0.65


@dataclass(frozen=True)
class BinaryMetrics:
    threshold: float
    count: int
    ai_count: int
    real_count: int
    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int
    ai_precision: float
    ai_recall: float
    real_specificity: float
    balanced_accuracy: float
    accuracy: float
    roc_auc: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(labels.sum())
    negatives = int(labels.size - positives)
    if positives == 0 or negatives == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=np.float64)
    start = 0
    while start < scores.size:
        end = start + 1
        while end < scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end

    positive_rank_sum = float(ranks[labels == 1].sum())
    return (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def binary_metrics(
    labels: np.ndarray | list[int],
    scores: np.ndarray | list[float],
    threshold: float = THRESHOLD,
) -> BinaryMetrics:
    labels_array = np.asarray(labels, dtype=np.int8)
    scores_array = np.asarray(scores, dtype=np.float64)
    if labels_array.ndim != 1 or scores_array.ndim != 1 or labels_array.size != scores_array.size:
        raise ValueError("labels and scores must be equally sized one-dimensional arrays")
    if labels_array.size == 0 or not np.isin(labels_array, [0, 1]).all():
        raise ValueError("labels must be a non-empty binary array")
    if not np.isfinite(scores_array).all() or ((scores_array < 0) | (scores_array > 1)).any():
        raise ValueError("scores must be finite probabilities in [0, 1]")

    predictions = scores_array >= threshold
    positives = labels_array == 1
    negatives = ~positives
    tp = int((predictions & positives).sum())
    tn = int((~predictions & negatives).sum())
    fp = int((predictions & negatives).sum())
    fn = int((~predictions & positives).sum())
    ai_count = tp + fn
    real_count = tn + fp
    ai_recall = tp / ai_count if ai_count else float("nan")
    ai_precision = tp / (tp + fp) if tp + fp else float("nan")
    specificity = tn / real_count if real_count else float("nan")
    balanced = (ai_recall + specificity) / 2 if ai_count and real_count else float("nan")

    return BinaryMetrics(
        threshold=threshold,
        count=int(labels_array.size),
        ai_count=ai_count,
        real_count=real_count,
        true_positive=tp,
        true_negative=tn,
        false_positive=fp,
        false_negative=fn,
        ai_precision=ai_precision,
        ai_recall=ai_recall,
        real_specificity=specificity,
        balanced_accuracy=balanced,
        accuracy=(tp + tn) / labels_array.size,
        roc_auc=roc_auc(labels_array, scores_array),
    )


def best_balanced_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, BinaryMetrics]:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    candidates = np.unique(np.concatenate(([0.0], scores, [1.0])))
    best = max(
        (binary_metrics(labels, scores, float(threshold)) for threshold in candidates),
        key=lambda result: (result.balanced_accuracy, min(result.ai_recall, result.real_specificity)),
    )
    return best.threshold, best
