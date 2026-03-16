from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Sequence


class DifficultyLabel(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass(frozen=True)
class DifficultyThresholds:
    low: float
    high: float


def extract_complexity_scores(record: dict) -> list[float]:
    """Extract DEITA complexity scores from a raw data record."""
    annotation = record.get("annotation") or {}
    deita = annotation.get("deita") or {}
    scores = deita.get("complexity_scores") or []

    out: list[float] = []
    for score in scores:
        try:
            out.append(float(score))
        except (TypeError, ValueError):
            continue
    return out


def compute_difficulty_score(record: dict) -> float:
    """Aggregate one record's complexity scores into a scalar difficulty score."""
    scores = extract_complexity_scores(record)
    if not scores:
        raise ValueError("Record does not contain valid annotation.deita.complexity_scores.")
    return sum(scores) / len(scores)


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile with q in [0, 1]."""
    if not sorted_values:
        raise ValueError("Cannot compute percentile for empty values.")
    if q <= 0.0:
        return float(sorted_values[0])
    if q >= 1.0:
        return float(sorted_values[-1])

    pos = (len(sorted_values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(sorted_values[lo])
    frac = pos - lo
    return float(sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac)


def fit_difficulty_thresholds(scores: Sequence[float]) -> DifficultyThresholds:
    """Fit easy/medium/hard thresholds from 33% and 66% quantiles."""
    clean_scores = sorted(float(s) for s in scores)
    if not clean_scores:
        raise ValueError("Cannot fit thresholds on empty score list.")

    low = _percentile(clean_scores, 1.0 / 3.0)
    high = _percentile(clean_scores, 2.0 / 3.0)
    return DifficultyThresholds(low=low, high=high)


def assign_difficulty_label(score: float, thresholds: DifficultyThresholds) -> DifficultyLabel:
    """Assign easy/medium/hard using fitted thresholds."""
    value = float(score)
    if value <= thresholds.low:
        return DifficultyLabel.EASY
    if value <= thresholds.high:
        return DifficultyLabel.MEDIUM
    return DifficultyLabel.HARD


def annotate_record_difficulty(record: dict, thresholds: DifficultyThresholds) -> dict:
    """Attach difficulty score/label to one raw record in-place."""
    score = compute_difficulty_score(record)
    label = assign_difficulty_label(score, thresholds)

    annotation = record.setdefault("annotation", {})
    annotation["difficulty"] = {
        "score": score,
        "label": label.value,
    }
    return record


def collect_difficulty_scores(records: Iterable[dict]) -> list[float]:
    """Collect difficulty scores for all records that contain DEITA complexity scores."""
    out: list[float] = []
    for record in records:
        out.append(compute_difficulty_score(record))
    return out

