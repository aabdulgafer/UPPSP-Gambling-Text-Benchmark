#!/usr/bin/env python3
"""
Corrected extension statistical analysis for Research 2.

Purpose
-------
Re-run statistical post-processing for the ModernBERT/NeoBERT extension using
the SAME transformer estimand as the original frozen benchmark:

    mean of seed-specific metrics across seeds 13, 42, and 87

This is deliberately NOT a probability ensemble across seeds.

The script:
- reads existing frozen prediction folders only;
- performs no training, tuning, checkpoint selection, or model inference;
- aggregates monologue probabilities to users separately within each seed;
- uses 10,000 user-blocked bootstrap replicates;
- uses 20,000 paired user-level randomization replicates;
- applies Holm correction across the fixed five extension comparisons;
- preserves the previous extension_statistical_analysis_v1 output untouched.

Expected input directories
--------------------------
contemporary_extension_final_test_results_v1
final_test_results_v1
final_test_baseline_predictions_v1

Default output directory
------------------------
extension_statistical_analysis_seedmean_v1
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import platform
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EXPECTED_DATASET_SHA256 = (
    "421fd5a0f0b8a1f6221a7cee95a41970643bf547038d85db770ca9b979edffff"
)
EXPECTED_TEST_ROWS = 9441
EXPECTED_TEST_USERS = 300
SEEDS = [13, 42, 87]
BOOTSTRAP_REPLICATES = 10_000
PERMUTATION_REPLICATES = 20_000
RANDOM_SEED = 20_260_913

TRAITS = [
    "negative_urgency",
    "positive_urgency",
    "lack_of_premeditation",
    "lack_of_perseverance",
    "sensation_seeking",
]
LEVELS = ["low", "mid", "high"]
LEVEL_TO_INT = {v: i for i, v in enumerate(LEVELS)}

MODEL_ALIASES = {
    "ModernBERT": ["ModernBERT", "ModernBERT-base", "modernbert", "modernbert-base"],
    "NeoBERT": ["NeoBERT", "neobert"],
    "ALBERT": ["ALBERT", "albert"],
}

COMPARISONS = [
    ("modernbert_vs_albert", "ModernBERT", "ALBERT"),
    ("neobert_vs_albert", "NeoBERT", "ALBERT"),
    ("modernbert_vs_tfidf_lr", "ModernBERT", "TF-IDF logistic regression"),
    ("neobert_vs_tfidf_lr", "NeoBERT", "TF-IDF logistic regression"),
    ("modernbert_vs_neobert", "ModernBERT", "NeoBERT"),
]


@dataclass
class RunData:
    label: str
    seed: int
    y_true: np.ndarray
    scores: np.ndarray
    user_ids: np.ndarray
    monologue_ids: np.ndarray


@dataclass
class SufficientStats:
    confusion_by_user: np.ndarray
    exact_by_user: np.ndarray
    observations_by_user: np.ndarray
    absolute_error_by_user: np.ndarray


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_bytes(path: Path, value: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", delete=False, dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp"
    ) as f:
        f.write(value)
        tmp = Path(f.name)
    os.replace(tmp, path)


def atomic_json(path: Path, value: Any):
    atomic_bytes(
        path,
        (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8"),
    )


def atomic_csv(path: Path, frame: pd.DataFrame):
    buf = io.StringIO()
    frame.to_csv(buf, index=False, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
    atomic_bytes(path, ("\ufeff" + buf.getvalue()).encode("utf-8"))


def read_json(path: Path):
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def find_unique(root: Path, filename: str) -> Path:
    matches = list(root.rglob(filename))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one {filename!r} under {root}; found {len(matches)}: {matches}"
        )
    return matches[0]


def find_prediction_files(root: Path) -> list[Path]:
    return sorted(root.rglob("test_predictions.csv"))


def normalize_label(value: str) -> str | None:
    s = str(value).lower().replace("_", "").replace("-", "").replace(" ", "")
    if "modernbert" in s:
        return "ModernBERT"
    if "neobert" in s:
        return "NeoBERT"
    if s == "albert" or "albertbase" in s:
        return "ALBERT"
    return None


def infer_seed(path: Path, config: dict | None = None) -> int:
    if config is not None and "seed" in config:
        return int(config["seed"])
    for part in reversed(path.parts):
        low = part.lower()
        for seed in SEEDS:
            if f"seed{seed}" in low or f"seed_{seed}" in low or low == str(seed):
                return seed
    raise RuntimeError(f"Could not infer seed for {path}")


def transformer_frame_to_run(frame: pd.DataFrame, label: str, seed: int) -> RunData:
    required = {"monologue_id", "user_id"}
    for trait in TRAITS:
        required.update({f"{trait}_true", f"{trait}_predicted"})
        required.update({f"{trait}_prob_{level}" for level in LEVELS})
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"Missing columns for {label}/{seed}: {missing}")
    if len(frame) != EXPECTED_TEST_ROWS:
        raise RuntimeError(f"Unexpected rows for {label}/{seed}: {len(frame)}")
    if frame["monologue_id"].nunique() != EXPECTED_TEST_ROWS:
        raise RuntimeError(f"Duplicate monologue IDs for {label}/{seed}")
    if frame["user_id"].nunique() != EXPECTED_TEST_USERS:
        raise RuntimeError(f"Unexpected users for {label}/{seed}")

    y_true = np.column_stack(
        [frame[f"{t}_true"].map(LEVEL_TO_INT).to_numpy() for t in TRAITS]
    ).astype(np.int8)

    score_sets = []
    for trait in TRAITS:
        cols = [f"{trait}_prob_{level}" for level in LEVELS]
        values = frame[cols].to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise RuntimeError(f"Non-finite probabilities: {label}/{seed}/{trait}")
        if not np.allclose(values.sum(axis=1), 1.0, atol=1e-6):
            raise RuntimeError(f"Invalid probabilities: {label}/{seed}/{trait}")
        saved = frame[f"{trait}_predicted"].map(LEVEL_TO_INT).to_numpy()
        if not np.array_equal(saved, values.argmax(axis=1)):
            raise RuntimeError(f"Argmax mismatch: {label}/{seed}/{trait}")
        score_sets.append(values)

    return RunData(
        label=label,
        seed=seed,
        y_true=y_true,
        scores=np.stack(score_sets, axis=1),
        user_ids=frame["user_id"].astype(str).to_numpy(),
        monologue_ids=frame["monologue_id"].astype(str).to_numpy(),
    )


def baseline_frame_to_run(frame: pd.DataFrame) -> RunData:
    label = "TF-IDF logistic regression"
    required = {"monologue_id", "user_id"}
    for trait in TRAITS:
        required.update({f"{trait}_true", f"{trait}_predicted"})
        required.update({f"{trait}_probability_{level}" for level in LEVELS})
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"Missing TF-IDF columns: {missing}")
    if len(frame) != EXPECTED_TEST_ROWS:
        raise RuntimeError(f"Unexpected TF-IDF rows: {len(frame)}")

    y_true = np.column_stack(
        [frame[f"{t}_true"].map(LEVEL_TO_INT).to_numpy() for t in TRAITS]
    ).astype(np.int8)
    score_sets = []
    for trait in TRAITS:
        cols = [f"{trait}_probability_{level}" for level in LEVELS]
        values = frame[cols].to_numpy(dtype=np.float64)
        if not np.allclose(values.sum(axis=1), 1.0, atol=1e-6):
            raise RuntimeError(f"Invalid TF-IDF probabilities: {trait}")
        saved = frame[f"{trait}_predicted"].map(LEVEL_TO_INT).to_numpy()
        if not np.array_equal(saved, values.argmax(axis=1)):
            raise RuntimeError(f"TF-IDF argmax mismatch: {trait}")
        score_sets.append(values)
    return RunData(
        label=label,
        seed=42,
        y_true=y_true,
        scores=np.stack(score_sets, axis=1),
        user_ids=frame["user_id"].astype(str).to_numpy(),
        monologue_ids=frame["monologue_id"].astype(str).to_numpy(),
    )


def load_transformers(root: Path, wanted: set[str]) -> dict[str, list[RunData]]:
    out: dict[str, list[RunData]] = {label: [] for label in wanted}
    for pred_path in find_prediction_files(root):
        config_path = pred_path.parent / "test_run_config.json"
        config = read_json(config_path) if config_path.exists() else None

        candidate = ""
        if config:
            candidate = " ".join(
                str(config.get(k, ""))
                for k in ["model_label", "model_name", "model", "checkpoint"]
            )
        candidate += " " + str(pred_path.parent)
        label = normalize_label(candidate)
        if label not in wanted:
            continue
        seed = infer_seed(pred_path, config)
        frame = pd.read_csv(pred_path, encoding="utf-8-sig")
        out[label].append(transformer_frame_to_run(frame, label, seed))

    for label in wanted:
        out[label].sort(key=lambda x: x.seed)
        seeds = [r.seed for r in out[label]]
        if seeds != SEEDS:
            raise RuntimeError(f"Expected seeds {SEEDS} for {label}; found {seeds}")
    return out


def load_tfidf(root: Path) -> RunData:
    candidates = list(root.rglob("tfidf_logistic_regression_test_predictions.csv"))
    if len(candidates) != 1:
        raise RuntimeError(
            "Expected exactly one tfidf_logistic_regression_test_predictions.csv "
            f"under {root}; found {candidates}"
        )
    frame = pd.read_csv(candidates[0], encoding="utf-8-sig")
    return baseline_frame_to_run(frame)


def sufficient_stats(run: RunData) -> SufficientStats:
    users = np.array(sorted(np.unique(run.user_ids)))
    if len(users) != EXPECTED_TEST_USERS:
        raise RuntimeError("Unexpected unique-user count")
    positions = {u: i for i, u in enumerate(users)}
    user_index = np.array([positions[u] for u in run.user_ids], dtype=np.int16)

    user_true = np.zeros((len(users), len(TRAITS)), dtype=np.int8)
    user_scores = np.zeros((len(users), len(TRAITS), 3), dtype=np.float64)
    for i in range(len(users)):
        sel = user_index == i
        targets = run.y_true[sel]
        if not np.all(targets == targets[0]):
            raise RuntimeError(f"Inconsistent target for user {users[i]}")
        user_true[i] = targets[0]
        user_scores[i] = run.scores[sel].mean(axis=0)

    user_pred = user_scores.argmax(axis=2).astype(np.int8)
    conf = np.zeros((len(users), len(TRAITS), 3, 3), dtype=np.int16)
    for trait in range(len(TRAITS)):
        np.add.at(
            conf,
            (np.arange(len(users)), trait, user_true[:, trait], user_pred[:, trait]),
            1,
        )
    return SufficientStats(
        confusion_by_user=conf,
        exact_by_user=np.all(user_true == user_pred, axis=1).astype(np.int16),
        observations_by_user=np.ones(len(users), dtype=np.int16),
        absolute_error_by_user=np.abs(user_true - user_pred).sum(axis=1),
    )


def metrics_from_confusion(confusion: np.ndarray) -> np.ndarray:
    # shape: batch x run x trait x true x pred
    diagonal = np.diagonal(confusion, axis1=-2, axis2=-1)
    true_support = confusion.sum(axis=-1)
    pred_support = confusion.sum(axis=-2)
    denom = true_support + pred_support
    f1 = np.divide(
        2.0 * diagonal,
        denom,
        out=np.zeros_like(diagonal, dtype=np.float64),
        where=denom != 0,
    )
    return f1.mean(axis=-1).mean(axis=-1)  # batch x run


def point_seed_f1(stats: list[SufficientStats]) -> np.ndarray:
    # Match the original benchmark exactly: first sum each run's
    # per-user confusion matrices over users, then compute one metric
    # value per independently evaluated seed/run.
    conf = np.stack([s.confusion_by_user for s in stats], axis=0).sum(axis=1)[None, ...]
    return metrics_from_confusion(conf)[0]


def bootstrap_distribution(
    stats: list[SufficientStats], counts: np.ndarray
) -> np.ndarray:
    conf_by_user = np.stack([s.confusion_by_user for s in stats], axis=0)
    conf = np.einsum("bu,rutij->brtij", counts, conf_by_user, optimize=True)
    per_run = metrics_from_confusion(conf)
    return per_run.mean(axis=1)


def align_run_count(left, right):
    if len(left) == len(right):
        return left, right
    if len(left) == 1:
        return left * len(right), right
    if len(right) == 1:
        return left, right * len(left)
    raise RuntimeError("Cannot align run counts")


def permutation_p_value(left, right, observed, rng, replicates):
    left, right = align_run_count(left, right)
    left_conf = np.stack([s.confusion_by_user for s in left], axis=0)
    right_conf = np.stack([s.confusion_by_user for s in right], axis=0)
    left_total = left_conf.sum(axis=1)
    combined_total = left_total + right_conf.sum(axis=1)
    delta = right_conf - left_conf

    exceed = 0
    completed = 0
    chunk = 1000
    while completed < replicates:
        size = min(chunk, replicates - completed)
        swaps = rng.integers(0, 2, size=(size, EXPECTED_TEST_USERS), dtype=np.int8)
        pleft = left_total[None, ...] + np.einsum(
            "bu,rutij->brtij", swaps, delta, optimize=True
        )
        pright = combined_total[None, ...] - pleft
        lmetric = metrics_from_confusion(pleft).mean(axis=1)
        rmetric = metrics_from_confusion(pright).mean(axis=1)
        null = lmetric - rmetric
        exceed += int(np.sum(np.abs(null) >= abs(observed)))
        completed += size
    return float((exceed + 1) / (replicates + 1))


def percentile_interval(values):
    lo, hi = np.quantile(values, [0.025, 0.975])
    return float(lo), float(hi)


def holm_adjust(p_values):
    n = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(n, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        candidate = (n - rank) * p_values[idx]
        running = max(running, candidate)
        adjusted[idx] = min(1.0, running)
    return adjusted.tolist()


def verify_alignment(runs: dict[str, list[RunData]]):
    flat = [(label, r) for label, rr in runs.items() for r in rr]
    ref_label, ref = flat[0]
    for label, run in flat[1:]:
        if not np.array_equal(run.monologue_ids, ref.monologue_ids):
            raise RuntimeError(f"Monologue alignment mismatch: {label}/{run.seed}")
        if not np.array_equal(run.user_ids, ref.user_ids):
            raise RuntimeError(f"User alignment mismatch: {label}/{run.seed}")
        if not np.array_equal(run.y_true, ref.y_true):
            raise RuntimeError(f"Target alignment mismatch: {label}/{run.seed}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--extension-dir", type=Path,
        default=Path("contemporary_extension_final_test_results_v1")
    )
    ap.add_argument(
        "--original-transformer-dir", type=Path,
        default=Path("final_test_results_v1")
    )
    ap.add_argument(
        "--baseline-dir", type=Path,
        default=Path("final_test_baseline_predictions_v1")
    )
    ap.add_argument(
        "--output-dir", type=Path,
        default=Path("extension_statistical_analysis_seedmean_v1")
    )
    args = ap.parse_args()

    if args.output_dir.exists():
        raise RuntimeError(f"Refusing to overwrite existing output: {args.output_dir}")

    extension = load_transformers(args.extension_dir, {"ModernBERT", "NeoBERT"})
    original = load_transformers(args.original_transformer_dir, {"ALBERT"})
    tfidf = load_tfidf(args.baseline_dir)

    runs = {
        "ModernBERT": extension["ModernBERT"],
        "NeoBERT": extension["NeoBERT"],
        "ALBERT": original["ALBERT"],
        "TF-IDF logistic regression": [tfidf],
    }
    verify_alignment(runs)

    stats = {label: [sufficient_stats(r) for r in rr] for label, rr in runs.items()}

    # Same user bootstrap seed and construction as original benchmark.
    rng = np.random.default_rng(RANDOM_SEED)
    counts = rng.multinomial(
        EXPECTED_TEST_USERS,
        np.full(EXPECTED_TEST_USERS, 1.0 / EXPECTED_TEST_USERS),
        size=BOOTSTRAP_REPLICATES,
    ).astype(np.int16)

    points = {}
    boot = {}
    rows = []
    for label, ss in stats.items():
        seed_values = point_seed_f1(ss)
        dist = bootstrap_distribution(ss, counts)
        points[label] = float(seed_values.mean())
        boot[label] = dist
        lo, hi = percentile_interval(dist)
        rows.append({
            "model": label,
            "runs": len(ss),
            "user_macro_f1": points[label],
            "ci_95_low": lo,
            "ci_95_high": hi,
            "seed_sd": float(seed_values.std(ddof=1)) if len(ss) > 1 else None,
            "seed_13": float(seed_values[0]) if len(ss) == 3 else None,
            "seed_42": float(seed_values[1]) if len(ss) == 3 else float(seed_values[0]),
            "seed_87": float(seed_values[2]) if len(ss) == 3 else None,
        })

    # Sanity checks against already frozen descriptive results.
    expected = {
        "ModernBERT": 0.5674,
        "NeoBERT": 0.5551,
        "ALBERT": 0.5625,
        "TF-IDF logistic regression": 0.6296,
    }
    for label, approx in expected.items():
        if abs(points[label] - approx) > 0.001:
            raise RuntimeError(
                f"Point-estimate sanity check failed for {label}: "
                f"{points[label]:.8f} vs expected approximately {approx:.4f}"
            )

    comparison_rows = []
    raw_p = []
    prng = np.random.default_rng(RANDOM_SEED + 1)
    for cid, left, right in COMPARISONS:
        diffs = boot[left] - boot[right]
        lo, hi = percentile_interval(diffs)
        observed = points[left] - points[right]
        p = permutation_p_value(
            stats[left], stats[right], observed, prng, PERMUTATION_REPLICATES
        )
        raw_p.append(p)
        comparison_rows.append({
            "comparison_id": cid,
            "model_a": left,
            "model_b": right,
            "score_a": points[left],
            "score_b": points[right],
            "difference_a_minus_b": observed,
            "bootstrap_ci_95_low": lo,
            "bootstrap_ci_95_high": hi,
            "permutation_p_two_sided": p,
            "permutation_replicates": PERMUTATION_REPLICATES,
        })

    adj = holm_adjust(raw_p)
    for row, p_adj in zip(comparison_rows, adj):
        row["holm_adjusted_p"] = p_adj
        excludes = row["bootstrap_ci_95_low"] > 0 or row["bootstrap_ci_95_high"] < 0
        row["bootstrap_ci_excludes_zero"] = bool(excludes)
        row["supported_at_0_05"] = bool(excludes and p_adj < 0.05)

    args.output_dir.mkdir(parents=True)
    model_df = pd.DataFrame(rows)
    comp_df = pd.DataFrame(comparison_rows)
    atomic_csv(args.output_dir / "extension_model_confidence_intervals_seedmean_v1.csv", model_df)
    atomic_csv(args.output_dir / "extension_pairwise_statistics_seedmean_v1.csv", comp_df)

    summary = {
        "status": "PASS",
        "created_utc": utc_now(),
        "analysis_scope": (
            "Frozen predictions only; corrected extension statistical post-processing; "
            "no training, inference, fitting, tuning, or checkpoint selection."
        ),
        "test_monologues": EXPECTED_TEST_ROWS,
        "test_users": EXPECTED_TEST_USERS,
        "bootstrap": {
            "unit": "user",
            "replicates": BOOTSTRAP_REPLICATES,
            "interval": "percentile 95%",
            "random_seed": RANDOM_SEED,
            "transformer_estimand": "mean metric across seeds 13, 42, and 87",
            "baseline_estimand": "single frozen seed-42 run",
        },
        "paired_tests": {
            "method": "Monte Carlo paired user-level randomization",
            "replicates": PERMUTATION_REPLICATES,
            "random_seed": RANDOM_SEED + 1,
            "comparisons": 5,
            "multiplicity": "Holm adjustment across the fixed five extension comparisons",
        },
        "comparison_family": [
            {"model_a": a, "model_b": b} for _, a, b in COMPARISONS
        ],
        "important_note": (
            "This analysis matches the original benchmark estimand. It does not average "
            "seed probabilities before prediction. The earlier probability-ensemble "
            "extension analysis is retained only as a diagnostic/non-primary artifact."
        ),
        "inputs": {
            "extension_dir": str(args.extension_dir.resolve()),
            "original_transformer_dir": str(args.original_transformer_dir.resolve()),
            "baseline_dir": str(args.baseline_dir.resolve()),
        },
        "output_sha256": {},
        "software": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }

    # Write summary once, then record hashes of CSV outputs.
    for name in [
        "extension_model_confidence_intervals_seedmean_v1.csv",
        "extension_pairwise_statistics_seedmean_v1.csv",
    ]:
        summary["output_sha256"][name] = sha256(args.output_dir / name)
    atomic_json(
        args.output_dir / "extension_statistical_analysis_seedmean_summary_v1.json",
        summary,
    )

    print("\nSTATUS: PASS")
    print("Correct estimand: mean seed-specific metrics (13, 42, 87)")
    print("\nModel estimates:")
    print(model_df.to_string(index=False))
    print("\nFixed extension comparisons:")
    print(comp_df.to_string(index=False))
    print(f"\nOutput: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
