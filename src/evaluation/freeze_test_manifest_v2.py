#!/usr/bin/env python3
"""Freeze the 21 validation-selected checkpoints before test evaluation.

This utility is read-only with respect to the run directories. It scans the
approved balanced-classification runs, verifies the frozen protocol, hashes
each retained checkpoint, and writes a manifest for independent review.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


APPROVED_MODELS = {
    "bert-base-uncased": "BERT",
    "roberta-base": "RoBERTa",
    "albert-base-v1": "ALBERT",
    "google/electra-small-discriminator": "ELECTRA",
    "xlnet-base-cased": "XLNet",
    "nghuyong/ernie-2.0-en": "ERNIE",
    "microsoft/deberta-v3-base": "DeBERTa-v3-base",
}
APPROVED_SEEDS = {13, 42, 87}
EXPECTED_DATASET_SHA256 = "421fd5a0f0b8a1f6221a7cee95a41970643bf547038d85db770ca9b979edffff"
LEGACY_BERT_RUNNER_SHA256 = "2585509982f5ee0ccbbc51aa5b39fbc43dbfe624cf998ec6ee2cfd6dca03155a"
APPROVED_RUNNER_SHA256 = {
    LEGACY_BERT_RUNNER_SHA256,
    "9f1d347b124dfee9457a5373d11f33b7907b242c8eca0bbd2370299fa2418b00",
    "d21d1ae434f0003c21bbe4233ef60fd060c8a59c9b7de0648d7fb7f9fe1ac981",
}
EXPECTED_PROTOCOL = {
    "task": "classification",
    "eval_split": "validation",
    "allow_test_evaluation": False,
    "max_length": 64,
    "batch_size": 16,
    "gradient_accumulation_steps": 2,
    "epochs": 5,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.10,
    "dropout": 0.30,
    "max_grad_norm": 1.0,
    "patience": 2,
    "num_workers": 0,
    "freeze_encoder": False,
    "disable_fp16": False,
    "dataset_sha256": EXPECTED_DATASET_SHA256,
    "status": "development_validation_only",
    "train_monologues": 44029,
    "evaluation_monologues": 9425,
    "evaluation_users": 300,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve())).replace("/", "\\")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--output", type=Path, default=Path("frozen_test_manifest.csv"))
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("frozen_test_manifest_summary.json"),
    )
    args = parser.parse_args()

    if not args.runs_dir.is_dir():
        raise FileNotFoundError(f"Runs directory not found: {args.runs_dir}")

    project_root = args.runs_dir.resolve().parent
    accepted = {}
    rejected = []

    for config_path in sorted(args.runs_dir.rglob("run_config.json")):
        run_dir = config_path.parent
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception as error:
            rejected.append({"run_dir": str(run_dir), "reason": f"invalid run_config.json: {error}"})
            continue

        model_name = config.get("model_name")
        seed = config.get("seed")
        if model_name not in APPROVED_MODELS or seed not in APPROVED_SEEDS:
            continue
        legacy_bert_seed42 = (
            model_name == "bert-base-uncased"
            and seed == 42
            and run_dir.name == "bert_base_uncased_classification_seed42"
            and config.get("class_weighting") in (None, "")
            and config.get("runner_sha256") in (None, "")
        )
        if config.get("task") != "classification":
            continue
        if config.get("class_weighting") != "balanced" and not legacy_bert_seed42:
            continue

        effective_class_weighting = "balanced"
        effective_runner_sha256 = (
            LEGACY_BERT_RUNNER_SHA256
            if legacy_bert_seed42
            else config.get("runner_sha256")
        )

        deviations = [
            f"{field}={config.get(field)!r}, expected {expected!r}"
            for field, expected in EXPECTED_PROTOCOL.items()
            if config.get(field) != expected
        ]
        if effective_runner_sha256 not in APPROVED_RUNNER_SHA256:
            deviations.append(
                f"runner_sha256={effective_runner_sha256!r}, expected an audited runner hash"
            )
        if deviations:
            rejected.append({"run_dir": str(run_dir), "reason": "; ".join(deviations)})
            continue

        checkpoint = run_dir / "best_model.pt"
        metrics_path = run_dir / "validation_metrics.json"
        history_path = run_dir / "training_history.json"
        if not checkpoint.is_file() or not metrics_path.is_file() or not history_path.is_file():
            rejected.append({"run_dir": str(run_dir), "reason": "missing checkpoint, metrics, or history"})
            continue

        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        history = json.loads(history_path.read_text(encoding="utf-8"))
        if not history:
            rejected.append({"run_dir": str(run_dir), "reason": "empty training history"})
            continue
        best_row = max(history, key=lambda row: row["selection_score"])
        user_macro_f1 = float(metrics["user_level"]["overall_macro_f1"])
        if abs(float(best_row["selection_score"]) - user_macro_f1) > 1e-10:
            rejected.append({"run_dir": str(run_dir), "reason": "checkpoint-selection score mismatch"})
            continue

        key = (model_name, int(seed))
        if key in accepted:
            raise RuntimeError(
                f"Duplicate approved run for {model_name}, seed {seed}: "
                f"{accepted[key]['run_dir']} and {run_dir}"
            )

        accepted[key] = {
            "model_label": APPROVED_MODELS[model_name],
            "model_name": model_name,
            "seed": int(seed),
            "run_dir": relative(run_dir, project_root),
            "checkpoint_path": relative(checkpoint, project_root),
            "checkpoint_sha256": sha256(checkpoint),
            "checkpoint_size_bytes": checkpoint.stat().st_size,
            "best_epoch": int(best_row["epoch"]),
            "validation_user_macro_f1": user_macro_f1,
            "validation_monologue_macro_f1": float(
                metrics["monologue_level"]["overall_macro_f1"]
            ),
            "dataset_sha256": config["dataset_sha256"],
            "runner_sha256": effective_runner_sha256,
            "runner_sha256_recorded": config.get("runner_sha256", ""),
            "legacy_protocol_exception": (
                "BERT seed 42 used the archived initial runner, which applied balanced "
                "classification weights unconditionally before class_weighting and "
                "runner_sha256 were added to run_config.json."
                if legacy_bert_seed42
                else ""
            ),
            "task": config["task"],
            "class_weighting": effective_class_weighting,
            "class_weighting_recorded": config.get("class_weighting", ""),
            "max_length": int(config["max_length"]),
            "batch_size": int(config["batch_size"]),
            "gradient_accumulation_steps": int(config["gradient_accumulation_steps"]),
            "epochs_allowed": int(config["epochs"]),
            "learning_rate": float(config["learning_rate"]),
            "weight_decay": float(config["weight_decay"]),
            "warmup_ratio": float(config["warmup_ratio"]),
            "dropout": float(config["dropout"]),
            "validation_status": config["status"],
        }

    expected_keys = {
        (model_name, seed) for model_name in APPROVED_MODELS for seed in APPROVED_SEEDS
    }
    missing = sorted(expected_keys - set(accepted))
    unexpected = sorted(set(accepted) - expected_keys)
    if missing or unexpected or rejected:
        diagnostic = {
            "status": "FAIL",
            "accepted_runs": len(accepted),
            "missing": [{"model_name": model, "seed": seed} for model, seed in missing],
            "unexpected": [{"model_name": model, "seed": seed} for model, seed in unexpected],
            "rejected": rejected,
        }
        args.summary_output.write_text(json.dumps(diagnostic, indent=2), encoding="utf-8")
        raise RuntimeError(
            f"Manifest freeze failed. Review {args.summary_output}; "
            f"accepted={len(accepted)}, missing={len(missing)}, rejected={len(rejected)}"
        )

    rows = sorted(accepted.values(), key=lambda row: (row["model_label"], row["seed"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "status": "PASS",
        "models": len(APPROVED_MODELS),
        "seeds_per_model": len(APPROVED_SEEDS),
        "checkpoints": len(rows),
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "runner_sha256_values": sorted({row["runner_sha256"] for row in rows}),
        "test_evaluated": False,
        "manifest_sha256": sha256(args.output),
        "manifest": str(args.output),
    }
    args.summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
