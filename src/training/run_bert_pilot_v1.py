#!/usr/bin/env python3
"""Fine-tune BERT for five-output UPPS-P recovery on the frozen user split.

The default run evaluates validation data only. Test evaluation requires the
explicit --allow-test-evaluation flag after the experimental protocol is frozen.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata as importlib_metadata
import json
import math
import platform
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
)
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup


TRAITS = [
    "negative_urgency",
    "positive_urgency",
    "lack_of_premeditation",
    "lack_of_perseverance",
    "sensation_seeking",
]
LEVELS = ["low", "mid", "high"]
LEVEL_TO_INT = {name: index for index, name in enumerate(LEVELS)}


def package_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def validate_dataset(data: pd.DataFrame, eval_split: str) -> dict:
    required = {
        "monologue_id",
        "user_id",
        "main_post",
        "risk_level",
        "prompt_type",
        "split",
    }
    required.update(f"{trait}_level" for trait in TRAITS)
    required.update(f"{trait}_score" for trait in TRAITS)
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    if data["monologue_id"].duplicated().any():
        raise ValueError("monologue_id must be unique")
    expected_splits = {"train", "validation", "test"}
    observed_splits = set(data["split"].unique())
    if not observed_splits.issubset(expected_splits):
        raise ValueError(f"Unexpected split values: {sorted(observed_splits - expected_splits)}")
    if "train" not in observed_splits or eval_split not in observed_splits:
        raise ValueError("Training or requested evaluation split is absent")

    for trait in TRAITS:
        level_column = f"{trait}_level"
        score_column = f"{trait}_score"
        invalid_levels = sorted(set(data[level_column].unique()).difference(LEVELS))
        if invalid_levels:
            raise ValueError(f"Unexpected labels in {level_column}: {invalid_levels}")
        scores = pd.to_numeric(data[score_column], errors="coerce")
        if scores.isna().any() or not scores.between(1.0, 4.0).all():
            raise ValueError(f"{score_column} must contain only scores from 1 to 4")
        if data.groupby("user_id")[level_column].nunique().max() != 1:
            raise ValueError(f"{level_column} is not stable within user")
        if data.assign(_score=scores).groupby("user_id")["_score"].nunique().max() != 1:
            raise ValueError(f"{score_column} is not stable within user")

    users_by_split = {
        split: set(group["user_id"])
        for split, group in data.groupby("split", sort=True)
    }
    split_names = sorted(users_by_split)
    for first_index, first_split in enumerate(split_names):
        for second_split in split_names[first_index + 1 :]:
            overlap = users_by_split[first_split].intersection(users_by_split[second_split])
            if overlap:
                raise ValueError(
                    f"User leakage detected between {first_split} and {second_split}: "
                    f"{len(overlap)} users"
                )
    return {
        "rows": int(len(data)),
        "users": int(data["user_id"].nunique()),
        "monologue_ids_unique": True,
        "user_targets_stable": True,
        "pairwise_split_user_overlap": 0,
    }


class EncodedTextDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, tokenizer, max_length: int, task: str):
        encoded = tokenizer(
            frame["main_post"].tolist(),
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.input_ids = encoded["input_ids"]
        self.attention_mask = encoded["attention_mask"]
        self.token_type_ids = encoded.get("token_type_ids")
        if task == "classification":
            self.targets = torch.tensor(
                np.column_stack([
                    frame[f"{trait}_level"].map(LEVEL_TO_INT).to_numpy()
                    for trait in TRAITS
                ]),
                dtype=torch.long,
            )
        else:
            self.targets = torch.tensor(
                frame[[f"{trait}_score" for trait in TRAITS]].to_numpy(dtype=np.float32),
                dtype=torch.float32,
            )

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> dict:
        item = {
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_mask[index],
            "targets": self.targets[index],
        }
        if self.token_type_ids is not None:
            item["token_type_ids"] = self.token_type_ids[index]
        return item


class MultiTaskEncoder(nn.Module):
    def __init__(
        self,
        model_name: str,
        task: str,
        dropout: float,
        freeze_encoder: bool,
        local_files_only: bool,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(
            model_name, local_files_only=local_files_only
        )
        self.task = task
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        if task == "classification":
            self.heads = nn.ModuleList([nn.Linear(hidden_size, 3) for _ in TRAITS])
        else:
            self.regression_head = nn.Linear(hidden_size, len(TRAITS))
        if freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        output = self.encoder(**kwargs)
        pooled = getattr(output, "pooler_output", None)
        if pooled is None:
            pooled = output.last_hidden_state[:, 0]
        pooled = self.dropout(pooled)
        if self.task == "classification":
            return torch.stack([head(pooled) for head in self.heads], dim=1)
        return 1.0 + 3.0 * torch.sigmoid(self.regression_head(pooled))


def classification_weights(frame: pd.DataFrame, device: torch.device) -> torch.Tensor:
    weights = []
    for trait in TRAITS:
        labels = frame[f"{trait}_level"].map(LEVEL_TO_INT).to_numpy()
        counts = np.bincount(labels, minlength=3).astype(float)
        weights.append(len(labels) / (3.0 * counts))
    return torch.tensor(np.asarray(weights), dtype=torch.float32, device=device)


def batch_loss(outputs, targets, task: str, class_weights=None):
    if task == "classification":
        losses = [
            F.cross_entropy(outputs[:, index, :], targets[:, index], weight=class_weights[index])
            for index in range(len(TRAITS))
        ]
        return torch.stack(losses).mean()
    return F.smooth_l1_loss(outputs, targets)


def run_epoch(
    model,
    loader,
    device,
    task,
    class_weights,
    optimizer=None,
    scheduler=None,
    scaler=None,
    gradient_accumulation_steps=1,
    use_amp=False,
    max_grad_norm=1.0,
):
    training = optimizer is not None
    model.train(training)
    outputs_all = []
    targets_all = []
    running_loss = 0.0
    if training:
        optimizer.zero_grad(set_to_none=True)

    for batch_index, batch in enumerate(loader):
        inputs = {
            key: value.to(device, non_blocking=True)
            for key, value in batch.items()
            if key != "targets"
        }
        targets = batch["targets"].to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(**inputs)
                loss = batch_loss(outputs, targets, task, class_weights)
                scaled_loss = loss / gradient_accumulation_steps
            if training:
                scaler.scale(scaled_loss).backward()
                update_now = (
                    (batch_index + 1) % gradient_accumulation_steps == 0
                    or batch_index + 1 == len(loader)
                )
                if update_now:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()

        running_loss += float(loss.detach().cpu()) * len(targets)
        outputs_all.append(outputs.detach().float().cpu().numpy())
        targets_all.append(targets.detach().cpu().numpy())

    return (
        running_loss / len(loader.dataset),
        np.concatenate(outputs_all),
        np.concatenate(targets_all),
    )


def aggregate_classification(users, targets, probabilities):
    frame = pd.DataFrame({"user_id": users})
    user_targets = []
    user_probabilities = []
    for user_id, indices in frame.groupby("user_id", sort=True).groups.items():
        positions = np.asarray(list(indices))
        first = targets[positions[0]]
        if not np.all(targets[positions] == first):
            raise ValueError(f"Inconsistent classification targets for {user_id}")
        user_targets.append(first)
        user_probabilities.append(probabilities[positions].mean(axis=0))
    return np.asarray(user_targets), np.asarray(user_probabilities)


def classification_metrics(targets, probabilities):
    predictions = probabilities.argmax(axis=2)
    per_trait = {}
    for index, trait in enumerate(TRAITS):
        precision, recall, f1, support = precision_recall_fscore_support(
            targets[:, index], predictions[:, index], labels=[0, 1, 2], zero_division=0
        )
        per_trait[trait] = {
            "macro_f1": float(f1_score(targets[:, index], predictions[:, index], average="macro")),
            "balanced_accuracy": float(
                balanced_accuracy_score(targets[:, index], predictions[:, index])
            ),
            "quadratic_weighted_kappa": float(
                cohen_kappa_score(targets[:, index], predictions[:, index], weights="quadratic")
            ),
            "ordinal_mae": float(mean_absolute_error(targets[:, index], predictions[:, index])),
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
        "n_records": int(len(targets)),
        "overall_macro_f1": float(np.mean([value["macro_f1"] for value in per_trait.values()])),
        "overall_balanced_accuracy": float(
            np.mean([value["balanced_accuracy"] for value in per_trait.values()])
        ),
        "exact_match_ratio": float(np.mean(np.all(targets == predictions, axis=1))),
        "hamming_loss": float(np.mean(targets != predictions)),
        "mean_ordinal_mae": float(np.mean(np.abs(targets - predictions))),
        "per_trait": per_trait,
    }


def safe_correlation(function, actual, predicted):
    if np.ptp(actual) < 1e-12 or np.ptp(predicted) < 1e-12:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        value = function(actual, predicted).statistic
    return float(value) if np.isfinite(value) else None


def aggregate_regression(users, targets, predictions):
    frame = pd.DataFrame({"user_id": users})
    user_targets = []
    user_predictions = []
    for user_id, indices in frame.groupby("user_id", sort=True).groups.items():
        positions = np.asarray(list(indices))
        first = targets[positions[0]]
        if not np.allclose(targets[positions], first):
            raise ValueError(f"Inconsistent score targets for {user_id}")
        user_targets.append(first)
        user_predictions.append(predictions[positions].mean(axis=0))
    return np.asarray(user_targets), np.asarray(user_predictions)


def regression_metrics(targets, predictions):
    per_trait = {}
    for index, trait in enumerate(TRAITS):
        per_trait[trait] = {
            "mae": float(mean_absolute_error(targets[:, index], predictions[:, index])),
            "rmse": float(mean_squared_error(targets[:, index], predictions[:, index]) ** 0.5),
            "pearson_r": safe_correlation(pearsonr, targets[:, index], predictions[:, index]),
            "spearman_rho": safe_correlation(spearmanr, targets[:, index], predictions[:, index]),
            "r_squared": float(r2_score(targets[:, index], predictions[:, index])),
        }
    return {
        "n_records": int(len(targets)),
        "mean_mae": float(np.mean([value["mae"] for value in per_trait.values()])),
        "mean_rmse": float(np.mean([value["rmse"] for value in per_trait.values()])),
        "per_trait": per_trait,
    }


def full_metrics(task, frame, targets, outputs):
    users = frame["user_id"].to_numpy()
    if task == "classification":
        probabilities = torch.softmax(torch.tensor(outputs), dim=2).numpy()
        user_targets, user_outputs = aggregate_classification(users, targets, probabilities)
        metric_function = classification_metrics
    else:
        probabilities = None
        user_targets, user_outputs = aggregate_regression(users, targets, outputs)
        metric_function = regression_metrics

    result = {
        "monologue_level": metric_function(targets, probabilities if probabilities is not None else outputs),
        "user_level": metric_function(user_targets, user_outputs),
        "by_risk_level": {},
        "by_prompt_type": {},
    }
    for field, result_key in [("risk_level", "by_risk_level"), ("prompt_type", "by_prompt_type")]:
        for value in sorted(frame[field].unique()):
            selected = np.flatnonzero(frame[field].to_numpy() == value)
            selected_users = users[selected]
            if task == "classification":
                selected_outputs = probabilities[selected]
                grouped_targets, grouped_outputs = aggregate_classification(
                    selected_users, targets[selected], selected_outputs
                )
            else:
                selected_outputs = outputs[selected]
                grouped_targets, grouped_outputs = aggregate_regression(
                    selected_users, targets[selected], selected_outputs
                )
            result[result_key][str(value)] = {
                "monologue_level": metric_function(targets[selected], selected_outputs),
                "user_level": metric_function(grouped_targets, grouped_outputs),
            }
    return result


def prediction_frame(task, frame, targets, outputs):
    result = frame[["monologue_id", "user_id", "risk_level", "prompt_type"]].copy()
    if task == "classification":
        probabilities = torch.softmax(torch.tensor(outputs), dim=2).numpy()
        predictions = probabilities.argmax(axis=2)
        for index, trait in enumerate(TRAITS):
            result[f"{trait}_true"] = [LEVELS[value] for value in targets[:, index]]
            result[f"{trait}_predicted"] = [LEVELS[value] for value in predictions[:, index]]
            for level_index, level in enumerate(LEVELS):
                result[f"{trait}_prob_{level}"] = probabilities[:, index, level_index]
    else:
        for index, trait in enumerate(TRAITS):
            result[f"{trait}_true_score"] = targets[:, index]
            result[f"{trait}_predicted_score"] = outputs[:, index]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="bert-base-uncased")
    parser.add_argument("--task", choices=["classification", "regression"], default="classification")
    parser.add_argument("--eval-split", choices=["validation", "test"], default="validation")
    parser.add_argument("--allow-test-evaluation", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.10)
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--disable-fp16", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    if args.eval_split == "test" and not args.allow_test_evaluation:
        raise ValueError("Test evaluation is locked; do not enable it during development")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.disable_fp16
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    data = pd.read_csv(args.dataset, encoding="utf-8-sig", keep_default_na=False)
    dataset_validation = validate_dataset(data, args.eval_split)
    train = data.loc[data["split"] == "train"].reset_index(drop=True)
    evaluation = data.loc[data["split"] == args.eval_split].reset_index(drop=True)
    if train.empty or evaluation.empty:
        raise ValueError("Training or evaluation partition is empty")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, local_files_only=args.local_files_only
    )
    train_dataset = EncodedTextDataset(train, tokenizer, args.max_length, args.task)
    eval_dataset = EncodedTextDataset(evaluation, tokenizer, args.max_length, args.task)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = MultiTaskEncoder(
        args.model_name,
        args.task,
        args.dropout,
        args.freeze_encoder,
        args.local_files_only,
    ).to(device)
    class_weights = (
        classification_weights(train, device) if args.task == "classification" else None
    )
    optimizer = AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    updates_per_epoch = math.ceil(
        len(train_loader) / args.gradient_accumulation_steps
    )
    total_updates = updates_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_updates * args.warmup_ratio),
        num_training_steps=total_updates,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint = args.output_dir / "best_model.pt"
    history = []
    best_score = -np.inf if args.task == "classification" else np.inf
    epochs_without_improvement = 0
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss, _, _ = run_epoch(
            model,
            train_loader,
            device,
            args.task,
            class_weights,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            use_amp=use_amp,
            max_grad_norm=args.max_grad_norm,
        )
        validation_loss, validation_outputs, validation_targets = run_epoch(
            model,
            eval_loader,
            device,
            args.task,
            class_weights,
            use_amp=use_amp,
        )
        metrics = full_metrics(
            args.task, evaluation, validation_targets, validation_outputs
        )
        selection_score = (
            metrics["user_level"]["overall_macro_f1"]
            if args.task == "classification"
            else metrics["user_level"]["mean_mae"]
        )
        improved = (
            selection_score > best_score
            if args.task == "classification"
            else selection_score < best_score
        )
        if improved:
            best_score = selection_score
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_checkpoint)
        else:
            epochs_without_improvement += 1

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "selection_score": float(selection_score),
            "improved": improved,
        })
        print(json.dumps(history[-1]))
        if epochs_without_improvement >= args.patience:
            break

    model.load_state_dict(torch.load(best_checkpoint, map_location=device))
    final_loss, final_outputs, final_targets = run_epoch(
        model,
        eval_loader,
        device,
        args.task,
        class_weights,
        use_amp=use_amp,
    )
    final_metrics = full_metrics(args.task, evaluation, final_targets, final_outputs)
    predictions = prediction_frame(args.task, evaluation, final_targets, final_outputs)

    run_config = vars(args).copy()
    run_config["dataset"] = str(args.dataset)
    run_config["output_dir"] = str(args.output_dir)
    run_config.update({
        "dataset_sha256": sha256(args.dataset),
        "dataset_validation": dataset_validation,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "cuda_runtime": torch.version.cuda,
        "fp16": use_amp,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": package_version("transformers"),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": package_version("scipy"),
            "scikit_learn": package_version("scikit-learn"),
        },
        "train_monologues": len(train),
        "evaluation_monologues": len(evaluation),
        "evaluation_users": int(evaluation["user_id"].nunique()),
        "elapsed_seconds": time.time() - started,
        "best_selection_score": float(best_score),
        "final_validation_loss": final_loss,
        "status": "development_validation_only" if args.eval_split == "validation" else "final_test",
    })

    (args.output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "training_history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "validation_metrics.json").write_text(
        json.dumps(final_metrics, indent=2) + "\n", encoding="utf-8"
    )
    predictions.to_csv(
        args.output_dir / "validation_predictions.csv",
        index=False,
        encoding="utf-8-sig",
        quoting=csv.QUOTE_MINIMAL,
        lineterminator="\n",
    )
    tokenizer.save_pretrained(args.output_dir / "tokenizer")

    print(json.dumps({
        "status": run_config["status"],
        "task": args.task,
        "best_selection_score": best_score,
        "metrics_file": str(args.output_dir / "validation_metrics.json"),
        "predictions_file": str(args.output_dir / "validation_predictions.csv"),
    }, indent=2))


if __name__ == "__main__":
    main()
