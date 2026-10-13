#!/usr/bin/env python3
"""Guarded, inference-only evaluation of the frozen Research 2 contemporary-model extension test split.

The script has two modes:

1. ``--preflight-only`` verifies the frozen manifest, dataset, runner, run
   configurations, tokenizers, and all 6 extension checkpoint hashes. It does not read
   the dataset contents or load any model.
2. Final evaluation requires two explicit unlock arguments. It loads each
   frozen checkpoint exactly once, performs inference on the already-frozen
   test split, and writes predictions and metrics outside the training run
   directories. It contains no training, optimization, model selection, or
   checkpoint-writing code.
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
import re
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


EXPECTED_MANIFEST_SHA256 = None  # supplied and pinned at runtime via --expected-manifest-sha256
EXPECTED_DATASET_SHA256 = (
    "421fd5a0f0b8a1f6221a7cee95a41970643bf547038d85db770ca9b979edffff"
)
EXPECTED_EVALUATION_RUNNER_SHA256 = None  # supplied and pinned at runtime via --expected-runner-sha256
EXPECTED_MODELS = {
    "ModernBERT": "answerdotai/ModernBERT-base",
    "NeoBERT": "chandar-lab/NeoBERT",
}
EXPECTED_SEEDS = {13, 42, 87}
EXPECTED_TEST_ROWS = 9_441
EXPECTED_TEST_USERS = 300
UNLOCK_PHRASE = "I_ACCEPT_ONE_FINAL_TEST_EVALUATION"
HEX64 = re.compile(r"^[0-9a-f]{64}$")


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


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n", delete=False, dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp"
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
            temporary,
            index=False,
            encoding="utf-8-sig",
            quoting=csv.QUOTE_MINIMAL,
            lineterminator="\n",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def resolve_project_path(project_root: Path, raw: str) -> Path:
    """Resolve a manifest path safely on Windows or POSIX."""
    normalized = raw.strip().replace("\\", "/")
    relative = PurePosixPath(normalized)
    if not normalized or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe or invalid relative path in manifest: {raw!r}")
    return project_root.joinpath(*relative.parts)


def require_equal(actual: Any, expected: Any, label: str, errors: list[str]) -> None:
    if actual != expected:
        errors.append(f"{label}: expected {expected!r}, found {actual!r}")


def require_float(
    raw: str, expected: float, label: str, errors: list[str], tolerance: float = 1e-12
) -> None:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        errors.append(f"{label}: not a valid number: {raw!r}")
        return
    if not math.isclose(value, expected, rel_tol=0.0, abs_tol=tolerance):
        errors.append(f"{label}: expected {expected}, found {value}")


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def validate_manifest_structure(
    rows: list[dict[str, str]], summary: dict[str, Any]
) -> list[str]:
    errors: list[str] = []
    require_equal(summary.get("status"), "PASS", "summary.status", errors)
    require_equal(summary.get("models"), 2, "summary.models", errors)
    require_equal(summary.get("seeds_per_model"), 3, "summary.seeds_per_model", errors)
    require_equal(summary.get("checkpoints"), 6, "summary.checkpoints", errors)
    require_equal(
        summary.get("dataset_sha256"), EXPECTED_DATASET_SHA256,
        "summary.dataset_sha256", errors
    )
    require_equal(summary.get("test_evaluated"), False, "summary.test_evaluated", errors)
    require_equal(set(summary.get("runner_sha256_values", [])), {EXPECTED_EVALUATION_RUNNER_SHA256}, "summary.runner_sha256_values", errors)
    require_equal(len(rows), 6, "manifest row count", errors)

    pairs: set[tuple[str, int]] = set()
    checkpoint_paths: set[str] = set()
    checkpoint_hashes: set[str] = set()
    for index, row in enumerate(rows, start=2):
        label = row.get("model_label", "")
        try:
            seed = int(row.get("seed", ""))
        except ValueError:
            errors.append(f"row {index}: invalid seed {row.get('seed')!r}")
            continue
        tag = f"row {index} ({label}, seed {seed})"
        pair = (label, seed)
        if pair in pairs:
            errors.append(f"{tag}: duplicate model/seed pair")
        pairs.add(pair)

        expected_model_name = EXPECTED_MODELS.get(label)
        if expected_model_name is None:
            errors.append(f"{tag}: unexpected model label")
        else:
            require_equal(row.get("model_name"), expected_model_name, f"{tag}.model_name", errors)
        if seed not in EXPECTED_SEEDS:
            errors.append(f"{tag}: unexpected seed")

        checkpoint_path = row.get("checkpoint_path", "").replace("\\", "/").lower()
        if checkpoint_path in checkpoint_paths:
            errors.append(f"{tag}: duplicate checkpoint path")
        checkpoint_paths.add(checkpoint_path)
        checkpoint_hash = row.get("checkpoint_sha256", "")
        if checkpoint_hash in checkpoint_hashes:
            errors.append(f"{tag}: duplicate checkpoint hash")
        checkpoint_hashes.add(checkpoint_hash)
        if not HEX64.fullmatch(checkpoint_hash):
            errors.append(f"{tag}: invalid checkpoint SHA-256")

        require_equal(row.get("dataset_sha256"), EXPECTED_DATASET_SHA256, f"{tag}.dataset", errors)
        require_equal(row.get("runner_sha256"), EXPECTED_EVALUATION_RUNNER_SHA256, f"{tag}.runner_sha256", errors)
        require_equal(row.get("task"), "classification", f"{tag}.task", errors)
        require_equal(row.get("class_weighting"), "balanced", f"{tag}.weighting", errors)
        require_equal(row.get("max_length"), "64", f"{tag}.max_length", errors)
        require_equal(row.get("batch_size"), "16", f"{tag}.batch_size", errors)
        require_equal(
            row.get("gradient_accumulation_steps"), "2", f"{tag}.gradient_accumulation", errors
        )
        require_equal(row.get("epochs_allowed"), "5", f"{tag}.epochs", errors)
        require_float(row.get("learning_rate", ""), 2e-5, f"{tag}.learning_rate", errors)
        require_float(row.get("weight_decay", ""), 0.01, f"{tag}.weight_decay", errors)
        require_float(row.get("warmup_ratio", ""), 0.1, f"{tag}.warmup_ratio", errors)
        require_float(row.get("dropout", ""), 0.3, f"{tag}.dropout", errors)
        require_equal(
            row.get("validation_status"), "development_validation_only",
            f"{tag}.validation_status", errors
        )

        try:
            checkpoint_size = int(row.get("checkpoint_size_bytes", ""))
            best_epoch = int(row.get("best_epoch", ""))
            if checkpoint_size <= 0:
                errors.append(f"{tag}: checkpoint size must be positive")
            if not 1 <= best_epoch <= 5:
                errors.append(f"{tag}: best epoch must be from 1 to 5")
            for metric in ("validation_user_macro_f1", "validation_monologue_macro_f1"):
                value = float(row.get(metric, ""))
                if not 0.0 <= value <= 1.0:
                    errors.append(f"{tag}: {metric} is outside [0, 1]")
        except (TypeError, ValueError):
            errors.append(f"{tag}: invalid numeric manifest field")

        require_equal(row.get("runner_sha256_recorded"), row.get("runner_sha256"), f"{tag}.recorded_runner", errors)
        require_equal(row.get("class_weighting_recorded"), "balanced", f"{tag}.recorded_weighting", errors)


    expected_pairs = {
        (model_label, seed) for model_label in EXPECTED_MODELS for seed in EXPECTED_SEEDS
    }
    require_equal(pairs, expected_pairs, "complete model/seed coverage", errors)
    return errors


def validate_run_config(
    row: dict[str, str], run_dir: Path, errors: list[str]
) -> None:
    label, seed = row["model_label"], int(row["seed"])
    tag = f"{label}/seed{seed}"
    config_path = run_dir / "run_config.json"
    history_path = run_dir / "training_history.json"
    metrics_path = run_dir / "validation_metrics.json"
    tokenizer_dir = run_dir / "tokenizer"
    for required in (config_path, history_path, metrics_path):
        if not required.is_file():
            errors.append(f"{tag}: missing {required}")
    if not tokenizer_dir.is_dir():
        errors.append(f"{tag}: missing tokenizer directory {tokenizer_dir}")
    if not config_path.is_file():
        return

    try:
        config = read_json(config_path)
    except Exception as exc:
        errors.append(f"{tag}: cannot read run_config.json: {exc}")
        return
    expected = {
        "model_name": row["model_name"],
        "task": "classification",
        "seed": seed,
        "eval_split": "validation",
        "allow_test_evaluation": False,
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "max_length": 64,
        "batch_size": 16,
        "gradient_accumulation_steps": 2,
        "epochs": 5,
        "class_weighting": "balanced",
        "status": "development_validation_only",
    }
    for key, expected_value in expected.items():
        require_equal(config.get(key), expected_value, f"{tag}.run_config.{key}", errors)
    require_equal(config.get("runner_sha256"), row["runner_sha256"], f"{tag}.run_config.runner_sha256", errors)
    require_equal(config.get("extension_phase"), "contemporary_encoder_extension_2026", f"{tag}.run_config.extension_phase", errors)
    require_equal(config.get("original_seven_retrained"), False, f"{tag}.run_config.original_seven_retrained", errors)
    if label == "NeoBERT":
        require_equal(config.get("trust_remote_code"), True, f"{tag}.run_config.trust_remote_code", errors)
    expected_pooling = "last_token" if label == "XLNet" else "pooler_output_or_first_token"
    if "pooling_strategy" in config:
        require_equal(
            config.get("pooling_strategy"), expected_pooling,
            f"{tag}.run_config.pooling_strategy", errors
        )
    for key, expected_value in {
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "dropout": 0.3,
    }.items():
        try:
            if not math.isclose(float(config.get(key)), expected_value, rel_tol=0, abs_tol=1e-12):
                errors.append(f"{tag}.run_config.{key}: unexpected value {config.get(key)!r}")
        except (TypeError, ValueError):
            errors.append(f"{tag}.run_config.{key}: invalid value {config.get(key)!r}")


def runtime_probe() -> dict[str, Any]:
    import torch
    import transformers

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "numpy": package_version("numpy"),
        "pandas": package_version("pandas"),
        "scipy": package_version("scipy"),
        "scikit_learn": package_version("scikit-learn"),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_runtime": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def run_preflight(args: argparse.Namespace) -> tuple[list[dict[str, str]], dict[str, Any]]:
    global EXPECTED_MANIFEST_SHA256, EXPECTED_EVALUATION_RUNNER_SHA256
    EXPECTED_MANIFEST_SHA256 = args.expected_manifest_sha256
    EXPECTED_EVALUATION_RUNNER_SHA256 = args.expected_runner_sha256
    project_root = args.project_root.resolve()
    manifest = args.manifest.resolve()
    summary_path = args.manifest_summary.resolve()
    dataset = args.dataset.resolve()
    runner = args.runner.resolve()
    errors: list[str] = []

    required_files = {
        "manifest": manifest,
        "manifest summary": summary_path,
        "dataset": dataset,
        "runner": runner,
    }
    for label, path in required_files.items():
        if not path.is_file():
            errors.append(f"Missing {label}: {path}")
    if errors:
        report = {"status": "FAIL", "created_utc": utc_now(), "errors": errors}
        atomic_json(args.preflight_report.resolve(), report)
        return [], report

    manifest_hash = sha256(manifest)
    dataset_hash = sha256(dataset)
    runner_hash = sha256(runner)
    evaluator_hash = sha256(Path(__file__).resolve())
    require_equal(
        evaluator_hash, args.expected_evaluator_sha256,
        "evaluator SHA-256", errors
    )
    require_equal(manifest_hash, args.expected_manifest_sha256, "manifest SHA-256", errors)
    require_equal(dataset_hash, EXPECTED_DATASET_SHA256, "dataset SHA-256", errors)
    require_equal(
        runner_hash, args.expected_runner_sha256,
        "evaluation runner SHA-256", errors
    )
    try:
        compile(runner.read_text(encoding="utf-8-sig"), str(runner), "exec")
    except Exception as exc:
        errors.append(f"Runner does not compile: {exc}")

    try:
        rows = read_manifest(manifest)
        summary = read_json(summary_path)
    except Exception as exc:
        rows, summary = [], {}
        errors.append(f"Cannot parse frozen manifest or summary: {exc}")
    if rows and summary:
        errors.extend(validate_manifest_structure(rows, summary))
        require_equal(
            summary.get("manifest_sha256"), manifest_hash,
            "summary/manifest SHA-256 agreement", errors
        )

    checkpoint_bytes = 0
    verified_checkpoints = 0
    if not errors:
        for row in rows:
            tag = f"{row['model_label']}/seed{row['seed']}"
            try:
                run_dir = resolve_project_path(project_root, row["run_dir"])
                checkpoint = resolve_project_path(project_root, row["checkpoint_path"])
            except ValueError as exc:
                errors.append(f"{tag}: {exc}")
                continue
            if not run_dir.is_dir():
                errors.append(f"{tag}: missing run directory {run_dir}")
                continue
            validate_run_config(row, run_dir, errors)
            if not checkpoint.is_file():
                errors.append(f"{tag}: missing checkpoint {checkpoint}")
                continue
            actual_size = checkpoint.stat().st_size
            checkpoint_bytes += actual_size
            require_equal(
                actual_size, int(row["checkpoint_size_bytes"]),
                f"{tag}.checkpoint_size", errors
            )
            actual_hash = sha256(checkpoint)
            require_equal(
                actual_hash, row["checkpoint_sha256"],
                f"{tag}.checkpoint_sha256", errors
            )
            if actual_hash == row["checkpoint_sha256"]:
                verified_checkpoints += 1

    probe: dict[str, Any] | None = None
    if not errors:
        try:
            probe = runtime_probe()
            if not probe["cuda_available"]:
                errors.append("CUDA is not available in the active Python environment")
        except Exception as exc:
            errors.append(f"Runtime probe failed: {type(exc).__name__}: {exc}")

    report = {
        "status": "PASS" if not errors else "FAIL",
        "created_utc": utc_now(),
        "mode": "read_only_preflight",
        "dataset_bytes_hashed": True,
        "dataset_records_parsed": False,
        "test_metrics_computed": False,
        "models": 2,
        "seeds_per_model": 3,
        "manifest_rows": len(rows),
        "verified_checkpoints": verified_checkpoints,
        "verified_checkpoint_bytes": checkpoint_bytes,
        "manifest_sha256": manifest_hash,
        "dataset_sha256": dataset_hash,
        "evaluation_runner_sha256": runner_hash,
        "evaluator_sha256": evaluator_hash,
        "runtime": probe,
        "errors": errors,
    }
    atomic_json(args.preflight_report.resolve(), report)
    return rows, report


def import_runner(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("frozen_research2_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import runner from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    required = {
        "TRAITS", "LEVELS", "EncodedTextDataset", "MultiTaskEncoder",
        "validate_dataset", "full_metrics", "prediction_frame", "set_seed"
    }
    missing = sorted(name for name in required if not hasattr(module, name))
    if missing:
        raise RuntimeError(f"Runner is missing required definitions: {missing}")
    return module


def load_state_dict_safely(torch: Any, checkpoint: Path, device: Any) -> dict[str, Any]:
    try:
        state = torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(checkpoint, map_location=device)
    if not isinstance(state, dict) or not state:
        raise ValueError(f"Checkpoint does not contain a nonempty state dictionary: {checkpoint}")
    return state


def infer_logits(
    runner: Any,
    torch: Any,
    frame: Any,
    model_name: str,
    tokenizer_dir: Path,
    checkpoint: Path,
    max_length: int,
    dropout: float,
    batch_size: int,
    num_workers: int,
    local_files_only: bool,
    disable_fp16: bool,
    trust_remote_code: bool,
) -> tuple[Any, Any, dict[str, Any]]:
    import numpy as np
    from torch.utils.data import DataLoader

    device = torch.device("cuda")
    use_amp = not disable_fp16
    tokenizer = runner.AutoTokenizer.from_pretrained(
        tokenizer_dir, local_files_only=True, trust_remote_code=trust_remote_code
    )
    encoded = runner.EncodedTextDataset(frame, tokenizer, max_length, "classification")
    loader = DataLoader(
        encoded,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    model = runner.MultiTaskEncoder(
        model_name,
        "classification",
        dropout,
        False,
        local_files_only,
        trust_remote_code,
    ).to(device)
    if model.model_type == "xlnet":
        if tokenizer.cls_token_id is None:
            raise ValueError("XLNet tokenizer does not define a CLS token")
        if not torch.all(encoded.input_ids[:, -1] == tokenizer.cls_token_id).item():
            raise ValueError("XLNet test encoding does not place CLS at the final token")

    state = load_state_dict_safely(torch, checkpoint, device)
    incompatibility = model.load_state_dict(state, strict=True)
    if incompatibility.missing_keys or incompatibility.unexpected_keys:
        raise ValueError(
            "Strict checkpoint load reported incompatible keys: "
            f"missing={incompatibility.missing_keys}, unexpected={incompatibility.unexpected_keys}"
        )
    del state

    model.eval()
    logits_all, targets_all = [], []
    with torch.inference_mode():
        for batch in loader:
            inputs = {
                key: value.to(device, non_blocking=True)
                for key, value in batch.items() if key != "targets"
            }
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(**inputs)
            logits_all.append(logits.detach().float().cpu().numpy())
            targets_all.append(batch["targets"].numpy())

    logits_array = np.concatenate(logits_all)
    targets_array = np.concatenate(targets_all)
    metadata = {
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0),
        "cuda_runtime": torch.version.cuda,
        "fp16": use_amp,
        "model_type": model.model_type,
        "pooling_strategy": model.pooling_strategy,
        "tokenizer_source": str(tokenizer_dir),
        "tokenizer_padding_side": tokenizer.padding_side,
        "tokenizer_cls_token_id": tokenizer.cls_token_id,
    }
    del model, encoded, loader, tokenizer
    torch.cuda.empty_cache()
    return logits_array, targets_array, metadata


def metric_summary_row(row: dict[str, str], metrics: dict[str, Any]) -> dict[str, Any]:
    mono = metrics["monologue_level"]
    user = metrics["user_level"]
    result: dict[str, Any] = {
        "model_label": row["model_label"],
        "model_name": row["model_name"],
        "seed": int(row["seed"]),
        "checkpoint_sha256": row["checkpoint_sha256"],
        "validation_user_macro_f1": float(row["validation_user_macro_f1"]),
        "validation_monologue_macro_f1": float(row["validation_monologue_macro_f1"]),
        "test_user_macro_f1": user["overall_macro_f1"],
        "test_user_balanced_accuracy": user["overall_balanced_accuracy"],
        "test_user_exact_match": user["exact_match_ratio"],
        "test_user_hamming_loss": user["hamming_loss"],
        "test_user_ordinal_mae": user["mean_ordinal_mae"],
        "test_monologue_macro_f1": mono["overall_macro_f1"],
        "test_monologue_balanced_accuracy": mono["overall_balanced_accuracy"],
        "test_monologue_exact_match": mono["exact_match_ratio"],
        "test_monologue_hamming_loss": mono["hamming_loss"],
        "test_monologue_ordinal_mae": mono["mean_ordinal_mae"],
    }
    for level_name, level_values in (("user", user), ("monologue", mono)):
        for trait, values in level_values["per_trait"].items():
            result[f"test_{level_name}_{trait}_macro_f1"] = values["macro_f1"]
            result[f"test_{level_name}_{trait}_balanced_accuracy"] = values["balanced_accuracy"]
            result[f"test_{level_name}_{trait}_quadratic_kappa"] = values["quadratic_weighted_kappa"]
            result[f"test_{level_name}_{trait}_ordinal_mae"] = values["ordinal_mae"]
    return result


def subgroup_summary_rows(
    manifest_row: dict[str, str], metrics: dict[str, Any]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family in ("by_risk_level", "by_prompt_type"):
        for subgroup, levels in metrics[family].items():
            for evaluation_level in ("monologue_level", "user_level"):
                values = levels[evaluation_level]
                rows.append({
                    "model_label": manifest_row["model_label"],
                    "model_name": manifest_row["model_name"],
                    "seed": int(manifest_row["seed"]),
                    "subgroup_family": family,
                    "subgroup": subgroup,
                    "evaluation_level": evaluation_level,
                    "n_records": values["n_records"],
                    "overall_macro_f1": values["overall_macro_f1"],
                    "overall_balanced_accuracy": values["overall_balanced_accuracy"],
                    "exact_match_ratio": values["exact_match_ratio"],
                    "hamming_loss": values["hamming_loss"],
                    "mean_ordinal_mae": values["mean_ordinal_mae"],
                })
    return rows


def valid_completion(marker: Path, row: dict[str, str], dataset_hash: str) -> bool:
    if not marker.is_file():
        return False
    try:
        value = read_json(marker)
        parent = marker.parent
        metrics = parent / value["metrics_file"]
        predictions = parent / value["predictions_file"]
        return all([
            value.get("status") == "COMPLETE",
            value.get("manifest_sha256") == EXPECTED_MANIFEST_SHA256,
            value.get("dataset_sha256") == dataset_hash,
            value.get("checkpoint_sha256") == row["checkpoint_sha256"],
            value.get("evaluator_sha256") == sha256(Path(__file__).resolve()),
            metrics.is_file(), predictions.is_file(),
            sha256(metrics) == value.get("metrics_sha256"),
            sha256(predictions) == value.get("predictions_sha256"),
        ])
    except Exception:
        return False


def evaluate_one(
    args: argparse.Namespace,
    runner: Any,
    torch: Any,
    test_frame: Any,
    row: dict[str, str],
    session_id: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    project_root = args.project_root.resolve()
    run_dir = resolve_project_path(project_root, row["run_dir"])
    checkpoint = resolve_project_path(project_root, row["checkpoint_path"])
    output_name = f"{row['model_label'].lower().replace(' ', '_').replace('/', '_')}_seed{row['seed']}"
    final_dir = args.output_dir.resolve() / output_name
    marker = final_dir / "completion.json"
    if final_dir.exists():
        if args.resume and valid_completion(marker, row, EXPECTED_DATASET_SHA256):
            metrics = read_json(final_dir / "test_metrics.json")
            return metric_summary_row(row, metrics), subgroup_summary_rows(row, metrics)
        raise RuntimeError(
            f"Refusing to overwrite incomplete or unaudited output directory: {final_dir}"
        )

    runner.set_seed(int(row["seed"]))
    started = time.time()
    with tempfile.TemporaryDirectory(
        dir=args.output_dir.resolve(), prefix=f".{output_name}.partial."
    ) as temporary_name:
        temporary = Path(temporary_name)
        logits, targets, inference_metadata = infer_logits(
            runner=runner,
            torch=torch,
            frame=test_frame,
            model_name=row["model_name"],
            tokenizer_dir=run_dir / "tokenizer",
            checkpoint=checkpoint,
            max_length=int(row["max_length"]),
            dropout=float(row["dropout"]),
            batch_size=args.inference_batch_size,
            num_workers=args.num_workers,
            local_files_only=not args.allow_model_download,
            disable_fp16=args.disable_fp16,
            trust_remote_code=(row["model_label"] == "NeoBERT"),
        )
        metrics = runner.full_metrics("classification", test_frame, targets, logits)
        predictions = runner.prediction_frame(
            "classification", test_frame, targets, logits
        )
        if len(predictions) != EXPECTED_TEST_ROWS:
            raise RuntimeError(
                f"Unexpected prediction count for {output_name}: {len(predictions)}"
            )
        metrics_path = temporary / "test_metrics.json"
        predictions_path = temporary / "test_predictions.csv"
        config_path = temporary / "test_run_config.json"
        atomic_json(metrics_path, metrics)
        atomic_csv(predictions_path, predictions)
        test_config = {
            "status": "FINAL_TEST_INFERENCE_ONLY",
            "session_id": session_id,
            "created_utc": utc_now(),
            "model_label": row["model_label"],
            "model_name": row["model_name"],
            "seed": int(row["seed"]),
            "manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "dataset_sha256": EXPECTED_DATASET_SHA256,
            "checkpoint_sha256": row["checkpoint_sha256"],
            "evaluator_sha256": sha256(Path(__file__).resolve()),
            "checkpoint_path": row["checkpoint_path"],
            "training_runner_sha256": row["runner_sha256"],
            "evaluation_runner_sha256": EXPECTED_EVALUATION_RUNNER_SHA256,
            "evaluator_sha256": sha256(Path(__file__)),
            "split": "test",
            "test_monologues": len(test_frame),
            "test_users": int(test_frame["user_id"].nunique()),
            "inference_batch_size": args.inference_batch_size,
            "num_workers": args.num_workers,
            "allow_model_download": args.allow_model_download,
            "elapsed_seconds": time.time() - started,
            "inference": inference_metadata,
            "environment": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "torch": torch.__version__,
                "transformers": package_version("transformers"),
                "numpy": package_version("numpy"),
                "pandas": package_version("pandas"),
                "scipy": package_version("scipy"),
                "scikit_learn": package_version("scikit-learn"),
            },
        }
        atomic_json(config_path, test_config)
        completion = {
            "status": "COMPLETE",
            "session_id": session_id,
            "model_label": row["model_label"],
            "model_name": row["model_name"],
            "seed": int(row["seed"]),
            "manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "dataset_sha256": EXPECTED_DATASET_SHA256,
            "checkpoint_sha256": row["checkpoint_sha256"],
            "metrics_file": "test_metrics.json",
            "metrics_sha256": sha256(metrics_path),
            "predictions_file": "test_predictions.csv",
            "predictions_sha256": sha256(predictions_path),
            "config_file": "test_run_config.json",
            "config_sha256": sha256(config_path),
        }
        atomic_json(temporary / "completion.json", completion)
        temporary.replace(final_dir)
    return metric_summary_row(row, metrics), subgroup_summary_rows(row, metrics)


def aggregate_model_results(run_frame: Any) -> Any:
    import pandas as pd

    identity = ["model_label", "model_name"]
    numeric = [
        column for column in run_frame.columns
        if column not in identity + ["seed", "checkpoint_sha256"]
    ]
    grouped = run_frame.groupby(identity, sort=True)[numeric].agg(["mean", "std"])
    grouped.columns = [f"{column}_{stat}" for column, stat in grouped.columns]
    return grouped.reset_index()


def aggregate_subgroup_results(subgroup_frame: Any) -> Any:
    identity = [
        "model_label", "model_name", "subgroup_family", "subgroup", "evaluation_level"
    ]
    numeric = [
        "n_records", "overall_macro_f1", "overall_balanced_accuracy",
        "exact_match_ratio", "hamming_loss", "mean_ordinal_mae"
    ]
    grouped = subgroup_frame.groupby(identity, sort=True)[numeric].agg(["mean", "std"])
    grouped.columns = [f"{column}_{stat}" for column, stat in grouped.columns]
    return grouped.reset_index()


def load_or_create_session(output_dir: Path, resume: bool) -> dict[str, Any]:
    session_path = output_dir / "evaluation_session.json"
    if output_dir.exists():
        if not resume:
            raise RuntimeError(
                f"Output directory already exists; use --resume only for the same session: {output_dir}"
            )
        if not session_path.is_file():
            raise RuntimeError(f"Existing output lacks evaluation_session.json: {output_dir}")
        session = read_json(session_path)
        require = {
            "manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "dataset_sha256": EXPECTED_DATASET_SHA256,
            "evaluation_runner_sha256": EXPECTED_EVALUATION_RUNNER_SHA256,
            "evaluator_sha256": sha256(Path(__file__).resolve()),
        }
        for key, expected in require.items():
            if session.get(key) != expected:
                raise RuntimeError(f"Existing session has incompatible {key}")
        return session

    output_dir.mkdir(parents=True)
    session = {
        "status": "IN_PROGRESS",
        "session_id": str(uuid.uuid4()),
        "started_utc": utc_now(),
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "evaluation_runner_sha256": EXPECTED_EVALUATION_RUNNER_SHA256,
        "evaluator_sha256": sha256(Path(__file__)),
        "protocol": "all 6 frozen contemporary-extension checkpoints; inference only; no model selection",
    }
    atomic_json(session_path, session)
    return session


def run_final_evaluation(args: argparse.Namespace, rows: list[dict[str, str]]) -> None:
    if not args.unlock_final_test or args.confirmation != UNLOCK_PHRASE:
        raise RuntimeError(
            "Final test remains locked. Both --unlock-final-test and the exact "
            f"--confirmation {UNLOCK_PHRASE} are required."
        )

    import pandas as pd
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA must be available for the frozen final evaluation")
    runner = import_runner(args.runner.resolve())
    data = pd.read_csv(args.dataset.resolve(), encoding="utf-8-sig", keep_default_na=False)
    validation = runner.validate_dataset(data, "test")
    test_frame = data.loc[data["split"] == "test"].reset_index(drop=True)
    if len(test_frame) != EXPECTED_TEST_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_TEST_ROWS} test monologues, found {len(test_frame)}"
        )
    if test_frame["user_id"].nunique() != EXPECTED_TEST_USERS:
        raise RuntimeError(
            f"Expected {EXPECTED_TEST_USERS} test users, found {test_frame['user_id'].nunique()}"
        )

    session = load_or_create_session(args.output_dir.resolve(), args.resume)
    run_rows: list[dict[str, Any]] = []
    subgroup_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        print(
            f"[{index}/6] Inference only: {row['model_label']} seed {row['seed']}",
            flush=True,
        )
        summary_row, subgroup = evaluate_one(
            args, runner, torch, test_frame, row, session["session_id"]
        )
        run_rows.append(summary_row)
        subgroup_rows.extend(subgroup)

    run_frame = pd.DataFrame(run_rows).sort_values(["model_label", "seed"])
    subgroup_frame = pd.DataFrame(subgroup_rows).sort_values(
        ["model_label", "seed", "subgroup_family", "subgroup", "evaluation_level"]
    )
    model_frame = aggregate_model_results(run_frame)
    subgroup_model_frame = aggregate_subgroup_results(subgroup_frame)
    output_files = {
        "per_run": args.output_dir.resolve() / "final_test_per_run_summary.csv",
        "per_model": args.output_dir.resolve() / "final_test_per_model_summary.csv",
        "subgroups": args.output_dir.resolve() / "final_test_subgroup_summary.csv",
        "subgroups_per_model": (
            args.output_dir.resolve() / "final_test_subgroup_per_model_summary.csv"
        ),
    }
    atomic_csv(output_files["per_run"], run_frame)
    atomic_csv(output_files["per_model"], model_frame)
    atomic_csv(output_files["subgroups"], subgroup_frame)
    atomic_csv(
        output_files["subgroups_per_model"],
        subgroup_model_frame,
    )
    final_summary = {
        "status": "COMPLETE",
        "session_id": session["session_id"],
        "completed_utc": utc_now(),
        "dataset_validation": validation,
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "models": 2,
        "seeds_per_model": 3,
        "completed_checkpoints": len(run_frame),
        "test_monologues": len(test_frame),
        "test_users": int(test_frame["user_id"].nunique()),
        "selection_rule": "No test-based model or checkpoint selection",
        "primary_metric": "mean user-level macro-F1 across five UPPS-P traits",
        "files": {key: path.name for key, path in output_files.items()},
        "file_sha256": {key: sha256(path) for key, path in output_files.items()},
    }
    atomic_json(args.output_dir.resolve() / "final_test_summary.json", final_summary)
    session.update({"status": "COMPLETE", "completed_utc": utc_now()})
    atomic_json(args.output_dir.resolve() / "evaluation_session.json", session)
    print(json.dumps(final_summary, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preflight and inference-only final test evaluation for Research 2"
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-summary", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--expected-evaluator-sha256", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-runner-sha256", required=True)
    parser.add_argument(
        "--preflight-report", type=Path, default=Path("contemporary_extension_final_test_preflight_v1.json")
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("contemporary_extension_final_test_results_v1"))
    parser.add_argument("--inference-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--disable-fp16", action="store_true")
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--unlock-final-test", action="store_true")
    parser.add_argument("--confirmation", default="")
    args = parser.parse_args()
    if args.inference_batch_size < 1:
        parser.error("--inference-batch-size must be at least 1")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")
    for field in ("expected_evaluator_sha256", "expected_manifest_sha256", "expected_runner_sha256"):
        value = getattr(args, field).lower()
        setattr(args, field, value)
        if not HEX64.fullmatch(value):
            parser.error(f"--{field.replace(chr(95), chr(45))} must contain exactly 64 hexadecimal characters")
    return args


def main() -> None:
    args = parse_args()
    rows, report = run_preflight(args)
    print(json.dumps(report, indent=2), flush=True)
    if report["status"] != "PASS":
        raise RuntimeError(
            f"Preflight failed with {len(report['errors'])} error(s); final test remains locked"
        )
    if args.preflight_only:
        return
    run_final_evaluation(args, rows)


if __name__ == "__main__":
    main()
