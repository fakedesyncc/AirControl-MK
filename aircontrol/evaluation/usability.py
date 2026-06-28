"""SUS and NASA-TLX scoring for assistive-control user studies."""

import csv
import os
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence


SUS_ITEMS = 10
NASA_TLX_DIMENSIONS = (
    "mental",
    "physical",
    "temporal",
    "performance",
    "effort",
    "frustration",
)


@dataclass
class UsabilityResult:
    participant_id: str
    condition: str
    sus_score: float
    nasa_tlx_raw: float
    nasa_tlx_weighted: float | None = None


def score_sus(responses: Sequence[int]) -> float:
    """Return SUS score in [0, 100] from ten Likert answers in [1, 5]."""
    values = [int(x) for x in responses]
    if len(values) != SUS_ITEMS:
        raise ValueError(f"SUS expects {SUS_ITEMS} answers, got {len(values)}")
    if any(x < 1 or x > 5 for x in values):
        raise ValueError("SUS answers must be in the 1..5 range")

    total = 0
    for idx, value in enumerate(values):
        total += value - 1 if idx % 2 == 0 else 5 - value
    return float(total * 2.5)


def score_nasa_tlx(ratings: Dict[str, float],
                   weights: Dict[str, float] | None = None) -> tuple[float, float | None]:
    """Return raw and optional weighted NASA-TLX score in [0, 100]."""
    normalized = _complete_tlx_dict(ratings, "rating")
    raw = sum(normalized.values()) / len(NASA_TLX_DIMENSIONS)
    if not weights:
        return float(raw), None

    weight_values = _complete_tlx_dict(weights, "weight")
    total_weight = sum(weight_values.values())
    if total_weight <= 0:
        raise ValueError("NASA-TLX weights must sum to a positive value")
    weighted = sum(normalized[k] * weight_values[k] for k in NASA_TLX_DIMENSIONS) / total_weight
    return float(raw), float(weighted)


def append_usability_result(path: str, participant_id: str, condition: str,
                            sus_responses: Sequence[int],
                            tlx_ratings: Dict[str, float],
                            tlx_weights: Dict[str, float] | None = None) -> UsabilityResult:
    """Score one study row and append it to a CSV file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sus = score_sus(sus_responses)
    raw_tlx, weighted_tlx = score_nasa_tlx(tlx_ratings, tlx_weights)
    result = UsabilityResult(participant_id, condition, sus, raw_tlx, weighted_tlx)

    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_csv_fields())
        if not exists:
            writer.writeheader()
        row = {
            "timestamp": round(time.time(), 3),
            "participant_id": participant_id,
            "condition": condition,
            "sus_score": round(sus, 2),
            "nasa_tlx_raw": round(raw_tlx, 2),
            "nasa_tlx_weighted": "" if weighted_tlx is None else round(weighted_tlx, 2),
        }
        for idx, value in enumerate(sus_responses, start=1):
            row[f"sus_{idx}"] = int(value)
        for key in NASA_TLX_DIMENSIONS:
            row[f"tlx_{key}"] = float(tlx_ratings[key])
            row[f"tlx_weight_{key}"] = "" if not tlx_weights else float(tlx_weights[key])
        writer.writerow(row)
    return result


def parse_sus_csv(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_tlx_pairs(value: str) -> Dict[str, float]:
    """Parse 'mental=40,physical=60,...' into a NASA-TLX rating dict."""
    result: Dict[str, float] = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Expected key=value, got {item!r}")
        key, raw = item.split("=", 1)
        key = key.strip().lower()
        if key not in NASA_TLX_DIMENSIONS:
            raise ValueError(f"Unknown NASA-TLX dimension: {key}")
        result[key] = float(raw)
    return _complete_tlx_dict(result, "rating")


def summarize_usability_rows(rows: Iterable[Dict[str, str]]) -> Dict[str, Dict[str, float]]:
    """Aggregate SUS and NASA-TLX by condition for reports."""
    buckets: Dict[str, List[tuple[float, float]]] = {}
    for row in rows:
        try:
            condition = row.get("condition") or "unknown"
            buckets.setdefault(condition, []).append((
                float(row["sus_score"]),
                float(row.get("nasa_tlx_weighted") or row["nasa_tlx_raw"]),
            ))
        except (KeyError, ValueError):
            continue
    summary: Dict[str, Dict[str, float]] = {}
    for condition, values in buckets.items():
        sus = [v[0] for v in values]
        tlx = [v[1] for v in values]
        summary[condition] = {
            "n": float(len(values)),
            "sus_mean": sum(sus) / len(sus),
            "nasa_tlx_mean": sum(tlx) / len(tlx),
        }
    return summary


def _complete_tlx_dict(values: Dict[str, float], label: str) -> Dict[str, float]:
    missing = [key for key in NASA_TLX_DIMENSIONS if key not in values]
    if missing:
        raise ValueError(f"NASA-TLX {label} missing dimensions: {', '.join(missing)}")
    normalized = {key: float(values[key]) for key in NASA_TLX_DIMENSIONS}
    if label == "rating":
        bad = [key for key, val in normalized.items() if val < 0 or val > 100]
        if bad:
            raise ValueError(f"NASA-TLX ratings must be in 0..100: {', '.join(bad)}")
    return normalized


def _csv_fields() -> List[str]:
    fields = [
        "timestamp",
        "participant_id",
        "condition",
        "sus_score",
        "nasa_tlx_raw",
        "nasa_tlx_weighted",
    ]
    fields.extend(f"sus_{idx}" for idx in range(1, SUS_ITEMS + 1))
    for key in NASA_TLX_DIMENSIONS:
        fields.append(f"tlx_{key}")
    for key in NASA_TLX_DIMENSIONS:
        fields.append(f"tlx_weight_{key}")
    return fields
