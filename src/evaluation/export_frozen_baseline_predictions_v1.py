#!/usr/bin/env python3
"""Export prediction-level outputs for the already-frozen Research 2 baselines.

This script does not tune or select models. It refits the five prespecified
baseline pipelines on the frozen training partition, evaluates the frozen test
partition, compares the complete metric tree with the canonical server result,
and saves prediction/score rows required for user-blocked statistical analysis.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata as importlib_metadata
import importlib.util
import json
import math
import os
import platform
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_DATASET_SHA256 = (
    "421fd5a0f0b8a1f6221a7cee95a41970643bf547038d85db770ca9b979edffff"
)
EXPECTED_RUNNER_SHA256 = (
    "dfe9f6b30a2c6135509ae8659f3ba9856a38657c540a525db0c789df50c3ad9a"
)
EXPECTED_CANONICAL_RESULTS_SHA256 = (
    "69a7af24bf50f4c09182fca225cd274ac9d7e498f5ded6e6541900d421a20895"
)
EXPECTED_TEST_ROWS = 9_441
EXPECTED_TEST_USERS = 300
EXPECTED_TRAIN_ROWS = 44_029
EXPECTED_TFIDF_FEATURES = 48_701
SEED = 42
CONFIRMATION = "EXPORT_ONLY_NO_MODEL_CHANGES"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n", delete=False,
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    ) as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8-sig", newline="", delete=False,
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    ) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_csv(
            temporary, index=False, encoding="utf-8-sig",
            quoting=csv.QUOTE_MINIMAL, lineterminator="\n"
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def import_runner(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("frozen_baseline_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import baseline runner from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    required = {
        "TRAITS", "LEVELS", "encoded_targets", "aligned_scores",
        "compute_metrics", "aggregate_by_user", "subgroup_metrics"
    }
    missing = sorted(name for name in required if not hasattr(module, name))
    if missing:
        raise RuntimeError(f"Baseline runner lacks required definitions: {missing}")
    return module


def evaluate_scores(
    runner: Any,
    name: str,
    score_matrices: list[Any],
    y_test: Any,
    test_users: Any,
    test_risk: Any,
    test_prompt_type: Any,
) -> dict[str, Any]:
    user_true, user_scores = runner.aggregate_by_user(
        test_users, y_test, score_matrices
    )
    return {
        "model": name,
        "monologue_level": runner.compute_metrics(y_test, score_matrices),
        "user_level": runner.compute_metrics(user_true, user_scores),
        "by_risk_level": runner.subgroup_metrics(
            y_test, score_matrices, test_users, test_risk
        ),
        "by_prompt_type": runner.subgroup_metrics(
            y_test, score_matrices, test_users, test_prompt_type
        ),
    }


def fit_estimators(
    runner: Any,
    estimators: list[Any],
    train_features: Any,
    test_features: Any,
    y_train: Any,
) -> list[Any]:
    scores = []
    for trait_index, estimator in enumerate(estimators):
        estimator.fit(train_features, y_train[:, trait_index])
        scores.append(runner.aligned_scores(estimator, test_features))
    return scores


def prediction_frame(
    runner: Any,
    test: Any,
    y_test: Any,
    score_matrices: list[Any],
    score_kind: str,
) -> Any:
    import numpy as np

    frame = test[["monologue_id", "user_id", "risk_level", "prompt_type"]].copy()
    for index, trait in enumerate(runner.TRAITS):
        scores = score_matrices[index]
        predictions = scores.argmax(axis=1)
        frame[f"{trait}_true"] = [runner.LEVELS[value] for value in y_test[:, index]]
        frame[f"{trait}_predicted"] = [runner.LEVELS[value] for value in predictions]
        for level_index, level in enumerate(runner.LEVELS):
            frame[f"{trait}_{score_kind}_{level}"] = scores[:, level_index]
        if score_kind == "probability":
            sums = scores.sum(axis=1)
            if not np.all(np.isfinite(scores)) or not np.allclose(sums, 1.0, atol=1e-10):
                raise RuntimeError(f"Invalid probability matrix for {trait}")
    return frame


def compare_values(
    canonical: Any,
    reproduced: Any,
    path: str = "root",
    differences: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if differences is None:
        differences = []
    if isinstance(canonical, dict) and isinstance(reproduced, dict):
        if set(canonical) != set(reproduced):
            differences.append({
                "path": path,
                "type": "key_mismatch",
                "canonical_only": sorted(set(canonical) - set(reproduced)),
                "reproduced_only": sorted(set(reproduced) - set(canonical)),
            })
        for key in sorted(set(canonical).intersection(reproduced)):
            compare_values(
                canonical[key], reproduced[key], f"{path}.{key}", differences
            )
    elif isinstance(canonical, list) and isinstance(reproduced, list):
        if len(canonical) != len(reproduced):
            differences.append({
                "path": path, "type": "length_mismatch",
                "canonical": len(canonical), "reproduced": len(reproduced)
            })
        for index, (left, right) in enumerate(zip(canonical, reproduced)):
            compare_values(left, right, f"{path}[{index}]", differences)
    elif (
        isinstance(canonical, (int, float))
        and isinstance(reproduced, (int, float))
        and not isinstance(canonical, bool)
        and not isinstance(reproduced, bool)
    ):
        delta = float(reproduced) - float(canonical)
        if not math.isclose(float(canonical), float(reproduced), rel_tol=1e-12, abs_tol=1e-12):
            differences.append({
                "path": path,
                "type": "numeric",
                "canonical": float(canonical),
                "reproduced": float(reproduced),
                "absolute_delta": abs(delta),
            })
    elif canonical != reproduced:
        differences.append({
            "path": path, "type": "value_mismatch",
            "canonical": canonical, "reproduced": reproduced
        })
    return differences


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export fixed baseline predictions for statistical analysis"
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--canonical-results", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-exporter-sha256", required=True)
    parser.add_argument("--confirmation", required=True)
    args = parser.parse_args()
    args.expected_exporter_sha256 = args.expected_exporter_sha256.lower()
    if len(args.expected_exporter_sha256) != 64:
        parser.error("--expected-exporter-sha256 must have 64 hexadecimal characters")
    return args


def main() -> None:
    args = parse_args()
    if args.confirmation != CONFIRMATION:
        raise RuntimeError(f"Confirmation must be exactly {CONFIRMATION}")
    dataset = args.dataset.resolve()
    canonical_path = args.canonical_results.resolve()
    runner_path = args.runner.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise RuntimeError(f"Refusing to overwrite existing output directory: {output_dir}")

    expected_hashes = {
        "dataset": (dataset, EXPECTED_DATASET_SHA256),
        "runner": (runner_path, EXPECTED_RUNNER_SHA256),
        "canonical results": (canonical_path, EXPECTED_CANONICAL_RESULTS_SHA256),
        "exporter": (Path(__file__).resolve(), args.expected_exporter_sha256),
    }
    for label, (path, expected) in expected_hashes.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(
                f"{label} SHA-256 mismatch: expected {expected}, found {actual}"
            )

    import numpy as np
    import pandas as pd
    from scipy import sparse
    from sklearn.dummy import DummyClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
    from sklearn.svm import LinearSVC

    runner = import_runner(runner_path)
    canonical = read_json(canonical_path)
    expected_top = {
        "status": "final_test",
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "evaluation_split": "test",
        "seed": SEED,
        "train_monologues": EXPECTED_TRAIN_ROWS,
        "evaluation_monologues": EXPECTED_TEST_ROWS,
        "evaluation_users": EXPECTED_TEST_USERS,
        "tfidf_vocabulary_size": EXPECTED_TFIDF_FEATURES,
    }
    for key, expected in expected_top.items():
        if canonical.get(key) != expected:
            raise RuntimeError(
                f"Canonical result has unexpected {key}: {canonical.get(key)!r}"
            )

    data = pd.read_csv(dataset, encoding="utf-8-sig", keep_default_na=False)
    train = data.loc[data["split"] == "train"].reset_index(drop=True)
    test = data.loc[data["split"] == "test"].reset_index(drop=True)
    if len(train) != EXPECTED_TRAIN_ROWS or len(test) != EXPECTED_TEST_ROWS:
        raise RuntimeError("Frozen train/test cardinality mismatch")
    if test["user_id"].nunique() != EXPECTED_TEST_USERS:
        raise RuntimeError("Frozen test-user cardinality mismatch")
    if set(train["user_id"]).intersection(test["user_id"]):
        raise RuntimeError("User leakage detected between train and test")

    y_train = runner.encoded_targets(train)
    y_test = runner.encoded_targets(test)
    test_users = test["user_id"].to_numpy()
    test_risk = test["risk_level"].to_numpy()
    test_prompt = test["prompt_type"].to_numpy()
    results: list[dict[str, Any]] = []
    predictions: dict[str, Any] = {}

    constant_train = np.zeros((len(train), 1), dtype=float)
    constant_test = np.zeros((len(test), 1), dtype=float)
    dummy_scores = fit_estimators(
        runner,
        [DummyClassifier(strategy="most_frequent") for _ in runner.TRAITS],
        constant_train, constant_test, y_train,
    )
    results.append(evaluate_scores(
        runner, "dummy_most_frequent", dummy_scores, y_test,
        test_users, test_risk, test_prompt
    ))
    predictions["dummy_most_frequent"] = prediction_frame(
        runner, test, y_test, dummy_scores, "probability"
    )

    risk_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    risk_train = risk_encoder.fit_transform(train[["risk_level"]])
    risk_test = risk_encoder.transform(test[["risk_level"]])
    risk_scores = fit_estimators(
        runner,
        [
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
            for _ in runner.TRAITS
        ],
        risk_train, risk_test, y_train,
    )
    results.append(evaluate_scores(
        runner, "risk_only_logistic_regression", risk_scores, y_test,
        test_users, test_risk, test_prompt
    ))
    predictions["risk_only_logistic_regression"] = prediction_frame(
        runner, test, y_test, risk_scores, "probability"
    )

    metadata_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    metadata_train = metadata_encoder.fit_transform(train[["risk_level", "prompt_type"]])
    metadata_test = metadata_encoder.transform(test[["risk_level", "prompt_type"]])
    train_ogdq = train[["ogdq_dim_1", "ogdq_dim_2", "ogdq_dim_3"]].apply(
        lambda row: sorted({value for value in row if value}), axis=1
    )
    test_ogdq = test[["ogdq_dim_1", "ogdq_dim_2", "ogdq_dim_3"]].apply(
        lambda row: sorted({value for value in row if value}), axis=1
    )
    ogdq_encoder = MultiLabelBinarizer(sparse_output=True)
    ogdq_train = ogdq_encoder.fit_transform(train_ogdq)
    ogdq_test = ogdq_encoder.transform(test_ogdq)
    metadata_train = sparse.hstack([metadata_train, ogdq_train], format="csr")
    metadata_test = sparse.hstack([metadata_test, ogdq_test], format="csr")
    metadata_scores = fit_estimators(
        runner,
        [
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
            for _ in runner.TRAITS
        ],
        metadata_train, metadata_test, y_train,
    )
    results.append(evaluate_scores(
        runner, "risk_ogdq_logistic_regression", metadata_scores, y_test,
        test_users, test_risk, test_prompt
    ))
    predictions["risk_ogdq_logistic_regression"] = prediction_frame(
        runner, test, y_test, metadata_scores, "probability"
    )

    vectorizer = TfidfVectorizer(
        lowercase=True, ngram_range=(1, 2), min_df=2, max_df=0.98,
        max_features=100000, sublinear_tf=True, strip_accents=None
    )
    text_train = vectorizer.fit_transform(train["main_post"])
    text_test = vectorizer.transform(test["main_post"])
    if len(vectorizer.vocabulary_) != EXPECTED_TFIDF_FEATURES:
        raise RuntimeError(
            f"Expected {EXPECTED_TFIDF_FEATURES} TF-IDF features, "
            f"found {len(vectorizer.vocabulary_)}"
        )

    tfidf_lr_scores = fit_estimators(
        runner,
        [
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
            for _ in runner.TRAITS
        ],
        text_train, text_test, y_train,
    )
    results.append(evaluate_scores(
        runner, "tfidf_logistic_regression", tfidf_lr_scores, y_test,
        test_users, test_risk, test_prompt
    ))
    predictions["tfidf_logistic_regression"] = prediction_frame(
        runner, test, y_test, tfidf_lr_scores, "probability"
    )

    tfidf_svm_scores = fit_estimators(
        runner,
        [
            LinearSVC(C=1.0, class_weight="balanced", random_state=SEED)
            for _ in runner.TRAITS
        ],
        text_train, text_test, y_train,
    )
    results.append(evaluate_scores(
        runner, "tfidf_linear_svm", tfidf_svm_scores, y_test,
        test_users, test_risk, test_prompt
    ))
    predictions["tfidf_linear_svm"] = prediction_frame(
        runner, test, y_test, tfidf_svm_scores, "decision_score"
    )

    reproduced = {
        "status": "final_test",
        "dataset": canonical["dataset"],
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "evaluation_split": "test",
        "seed": SEED,
        "train_monologues": len(train),
        "evaluation_monologues": len(test),
        "evaluation_users": int(test["user_id"].nunique()),
        "tfidf_vocabulary_size": int(len(vectorizer.vocabulary_)),
        "models": results,
    }
    differences = compare_values(canonical, reproduced)
    numeric_differences = [item for item in differences if item["type"] == "numeric"]
    structural_differences = [item for item in differences if item["type"] != "numeric"]
    maximum_delta = max(
        (item["absolute_delta"] for item in numeric_differences), default=0.0
    )
    headline_metric_names = {
        "overall_macro_f1",
        "overall_balanced_accuracy",
        "exact_match_ratio",
        "hamming_loss",
        "mean_ordinal_mae",
    }
    headline_differences = [
        item for item in numeric_differences
        if (
            len(item["path"].split(".")) == 4
            and item["path"].split(".")[0] == "root"
            and item["path"].split(".")[1].startswith("models[")
            and item["path"].split(".")[2] in {"monologue_level", "user_level"}
            and item["path"].split(".")[3] in headline_metric_names
        )
    ]
    maximum_headline_delta = max(
        (item["absolute_delta"] for item in headline_differences), default=0.0
    )
    # Cross-platform LBFGS arithmetic can change a few decisions near class
    # boundaries. No structural difference is allowed; a small numeric drift is
    # retained transparently and never used to change a model specification.
    if structural_differences or maximum_headline_delta > 0.005:
        raise RuntimeError(
            "Reproduced baselines materially differ from the canonical result: "
            f"structural={len(structural_differences)}, "
            f"max_headline_delta={maximum_headline_delta}"
        )

    output_dir.mkdir(parents=True)
    file_hashes: dict[str, str] = {}
    for model_name, frame in predictions.items():
        path = output_dir / f"{model_name}_test_predictions.csv"
        atomic_csv(path, frame)
        file_hashes[path.name] = sha256(path)
    reproduced_path = output_dir / "reproduced_baseline_metrics.json"
    atomic_json(reproduced_path, reproduced)
    file_hashes[reproduced_path.name] = sha256(reproduced_path)
    audit = {
        "status": "PASS",
        "created_utc": utc_now(),
        "purpose": "prediction export for fixed statistical analysis",
        "no_model_selection_or_tuning": True,
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "runner_sha256": EXPECTED_RUNNER_SHA256,
        "canonical_results_sha256": EXPECTED_CANONICAL_RESULTS_SHA256,
        "exporter_sha256": sha256(Path(__file__).resolve()),
        "seed": SEED,
        "train_monologues": len(train),
        "test_monologues": len(test),
        "test_users": int(test["user_id"].nunique()),
        "models": list(predictions),
        "canonical_comparison": {
            "exact_match": len(differences) == 0,
            "difference_count": len(differences),
            "structural_difference_count": len(structural_differences),
            "maximum_numeric_absolute_delta": maximum_delta,
            "maximum_headline_absolute_delta": maximum_headline_delta,
            "differences": differences,
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": package_version("scipy"),
            "scikit_learn": package_version("scikit-learn"),
        },
        "files_sha256": file_hashes,
    }
    atomic_json(output_dir / "baseline_prediction_export_audit.json", audit)
    print(json.dumps({
        "status": audit["status"],
        "output_dir": str(output_dir),
        "canonical_exact_match": audit["canonical_comparison"]["exact_match"],
        "difference_count": len(differences),
        "maximum_numeric_absolute_delta": maximum_delta,
        "maximum_headline_absolute_delta": maximum_headline_delta,
        "files": len(file_hashes) + 1,
    }, indent=2))


if __name__ == "__main__":
    main()
