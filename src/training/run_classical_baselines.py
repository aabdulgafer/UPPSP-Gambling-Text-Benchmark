#!/usr/bin/env python3
"""Run leakage-safe validation baselines for the Research 2 UPPS-P task."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    mean_absolute_error,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
from sklearn.svm import LinearSVC


TRAITS = [
    "negative_urgency",
    "positive_urgency",
    "lack_of_premeditation",
    "lack_of_perseverance",
    "sensation_seeking",
]
LEVELS = ["low", "mid", "high"]
LEVEL_TO_INT = {name: index for index, name in enumerate(LEVELS)}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def encoded_targets(frame: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [frame[f"{trait}_level"].map(LEVEL_TO_INT).to_numpy() for trait in TRAITS]
    )


def aligned_scores(estimator, features) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        raw = estimator.predict_proba(features)
    elif hasattr(estimator, "decision_function"):
        raw = estimator.decision_function(features)
    else:
        prediction = estimator.predict(features)
        raw = np.eye(len(LEVELS), dtype=float)[prediction]

    raw = np.asarray(raw)
    if raw.ndim == 1:
        raw = np.column_stack([-raw, raw])
    scores = np.full((raw.shape[0], len(LEVELS)), -np.inf, dtype=float)
    for source_column, class_id in enumerate(estimator.classes_):
        scores[:, int(class_id)] = raw[:, source_column]
    return scores


def compute_metrics(y_true: np.ndarray, score_matrices: list[np.ndarray]) -> dict:
    y_pred = np.column_stack([scores.argmax(axis=1) for scores in score_matrices])
    per_trait = {}
    for index, trait in enumerate(TRAITS):
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true[:, index],
            y_pred[:, index],
            labels=[0, 1, 2],
            zero_division=0,
        )
        per_trait[trait] = {
            "macro_f1": float(f1_score(y_true[:, index], y_pred[:, index], average="macro")),
            "balanced_accuracy": float(balanced_accuracy_score(y_true[:, index], y_pred[:, index])),
            "quadratic_weighted_kappa": float(
                cohen_kappa_score(y_true[:, index], y_pred[:, index], weights="quadratic")
            ),
            "ordinal_mae": float(mean_absolute_error(y_true[:, index], y_pred[:, index])),
            "per_level": {
                LEVELS[level]: {
                    "precision": float(precision[level]),
                    "recall": float(recall[level]),
                    "f1": float(f1[level]),
                    "support": int(support[level]),
                }
                for level in range(3)
            },
        }

    return {
        "n_records": int(len(y_true)),
        "overall_macro_f1": float(np.mean([item["macro_f1"] for item in per_trait.values()])),
        "overall_balanced_accuracy": float(
            np.mean([item["balanced_accuracy"] for item in per_trait.values()])
        ),
        "exact_match_ratio": float(np.mean(np.all(y_true == y_pred, axis=1))),
        # For multiclass-multioutput targets, Hamming loss is the fraction of
        # individual trait labels that are incorrect.
        "hamming_loss": float(np.mean(y_true != y_pred)),
        "mean_ordinal_mae": float(np.mean(np.abs(y_true - y_pred))),
        "per_trait": per_trait,
    }


def aggregate_by_user(
    user_ids: np.ndarray,
    y_true: np.ndarray,
    score_matrices: list[np.ndarray],
) -> tuple[np.ndarray, list[np.ndarray]]:
    order = np.argsort(user_ids, kind="stable")
    sorted_users = user_ids[order]
    boundaries = np.r_[0, np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1, len(order)]
    user_true = []
    user_scores = [[] for _ in TRAITS]
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        indices = order[start:end]
        first_target = y_true[indices[0]]
        if not np.all(y_true[indices] == first_target):
            raise ValueError(f"Target inconsistency within user {user_ids[indices[0]]}")
        user_true.append(first_target)
        for trait_index, scores in enumerate(score_matrices):
            user_scores[trait_index].append(scores[indices].mean(axis=0))
    return np.asarray(user_true), [np.asarray(scores) for scores in user_scores]


def subgroup_metrics(
    y_true: np.ndarray,
    score_matrices: list[np.ndarray],
    user_ids: np.ndarray,
    subgroup_values: np.ndarray,
) -> dict:
    output = {}
    for value in sorted(pd.unique(subgroup_values)):
        selected = np.flatnonzero(subgroup_values == value)
        selected_scores = [scores[selected] for scores in score_matrices]
        user_true, user_scores = aggregate_by_user(
            user_ids[selected], y_true[selected], selected_scores
        )
        output[str(value)] = {
            "monologue_level": compute_metrics(y_true[selected], selected_scores),
            "user_level": compute_metrics(user_true, user_scores),
        }
    return output


def evaluate_estimators(
    name: str,
    estimators: list,
    train_features,
    eval_features,
    y_train: np.ndarray,
    y_eval: np.ndarray,
    eval_users: np.ndarray,
    eval_risk: np.ndarray,
    eval_prompt_type: np.ndarray,
) -> dict:
    score_matrices = []
    for trait_index, estimator in enumerate(estimators):
        estimator.fit(train_features, y_train[:, trait_index])
        score_matrices.append(aligned_scores(estimator, eval_features))

    user_true, user_scores = aggregate_by_user(eval_users, y_eval, score_matrices)
    return {
        "model": name,
        "monologue_level": compute_metrics(y_eval, score_matrices),
        "user_level": compute_metrics(user_true, user_scores),
        "by_risk_level": subgroup_metrics(
            y_eval, score_matrices, eval_users, eval_risk
        ),
        "by_prompt_type": subgroup_metrics(
            y_eval, score_matrices, eval_users, eval_prompt_type
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eval-split", choices=["validation", "test"], default="validation")
    parser.add_argument("--allow-test-evaluation", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.eval_split == "test" and not args.allow_test_evaluation:
        raise ValueError("Test evaluation is locked; pass --allow-test-evaluation only after protocol freeze")

    data = pd.read_csv(args.dataset, encoding="utf-8-sig", keep_default_na=False)
    train = data.loc[data["split"] == "train"].reset_index(drop=True)
    evaluation = data.loc[data["split"] == args.eval_split].reset_index(drop=True)
    if train.empty or evaluation.empty:
        raise ValueError("Training or evaluation partition is empty")

    y_train = encoded_targets(train)
    y_eval = encoded_targets(evaluation)
    eval_users = evaluation["user_id"].to_numpy()
    eval_risk = evaluation["risk_level"].to_numpy()
    eval_prompt_type = evaluation["prompt_type"].to_numpy()

    results = []

    constant_train = np.zeros((len(train), 1), dtype=float)
    constant_eval = np.zeros((len(evaluation), 1), dtype=float)
    results.append(evaluate_estimators(
        "dummy_most_frequent",
        [DummyClassifier(strategy="most_frequent") for _ in TRAITS],
        constant_train,
        constant_eval,
        y_train,
        y_eval,
        eval_users,
        eval_risk,
        eval_prompt_type,
    ))

    risk_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    x_risk_train = risk_encoder.fit_transform(train[["risk_level"]])
    x_risk_eval = risk_encoder.transform(evaluation[["risk_level"]])
    results.append(evaluate_estimators(
        "risk_only_logistic_regression",
        [
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=args.seed)
            for _ in TRAITS
        ],
        x_risk_train,
        x_risk_eval,
        y_train,
        y_eval,
        eval_users,
        eval_risk,
        eval_prompt_type,
    ))

    categorical_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    x_meta_train = categorical_encoder.fit_transform(train[["risk_level", "prompt_type"]])
    x_meta_eval = categorical_encoder.transform(evaluation[["risk_level", "prompt_type"]])
    train_ogdq = train[["ogdq_dim_1", "ogdq_dim_2", "ogdq_dim_3"]].apply(
        lambda row: sorted({value for value in row if value}), axis=1
    )
    eval_ogdq = evaluation[["ogdq_dim_1", "ogdq_dim_2", "ogdq_dim_3"]].apply(
        lambda row: sorted({value for value in row if value}), axis=1
    )
    ogdq_encoder = MultiLabelBinarizer(sparse_output=True)
    x_ogdq_train = ogdq_encoder.fit_transform(train_ogdq)
    x_ogdq_eval = ogdq_encoder.transform(eval_ogdq)
    x_meta_train = sparse.hstack([x_meta_train, x_ogdq_train], format="csr")
    x_meta_eval = sparse.hstack([x_meta_eval, x_ogdq_eval], format="csr")
    results.append(evaluate_estimators(
        "risk_ogdq_logistic_regression",
        [
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=args.seed)
            for _ in TRAITS
        ],
        x_meta_train,
        x_meta_eval,
        y_train,
        y_eval,
        eval_users,
        eval_risk,
        eval_prompt_type,
    ))

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        max_features=100000,
        sublinear_tf=True,
        strip_accents=None,
    )
    x_text_train = vectorizer.fit_transform(train["main_post"])
    x_text_eval = vectorizer.transform(evaluation["main_post"])

    results.append(evaluate_estimators(
        "tfidf_logistic_regression",
        [
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=args.seed)
            for _ in TRAITS
        ],
        x_text_train,
        x_text_eval,
        y_train,
        y_eval,
        eval_users,
        eval_risk,
        eval_prompt_type,
    ))
    results.append(evaluate_estimators(
        "tfidf_linear_svm",
        [LinearSVC(C=1.0, class_weight="balanced", random_state=args.seed) for _ in TRAITS],
        x_text_train,
        x_text_eval,
        y_train,
        y_eval,
        eval_users,
        eval_risk,
        eval_prompt_type,
    ))

    payload = {
        "status": "development_validation_only" if args.eval_split == "validation" else "final_test",
        "dataset": str(args.dataset),
        "dataset_sha256": file_sha256(args.dataset),
        "evaluation_split": args.eval_split,
        "seed": args.seed,
        "train_monologues": len(train),
        "evaluation_monologues": len(evaluation),
        "evaluation_users": int(evaluation["user_id"].nunique()),
        "tfidf_vocabulary_size": int(len(vectorizer.vocabulary_)),
        "models": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "status": payload["status"],
        "summary": {
            item["model"]: {
                "monologue_macro_f1": round(item["monologue_level"]["overall_macro_f1"], 4),
                "user_macro_f1": round(item["user_level"]["overall_macro_f1"], 4),
                "user_exact_match": round(item["user_level"]["exact_match_ratio"], 4),
            }
            for item in results
        },
    }, indent=2))


if __name__ == "__main__":
    main()
