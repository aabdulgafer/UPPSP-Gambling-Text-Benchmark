
# Reproducibility Protocol and Public Artifact Guide

## 1. Purpose

This document summarizes the experimental protocol and identifies the publicly released artifacts supporting the UPPS-P Gambling Text Benchmark.

It distinguishes:

- Inspection of saved experimental results
- Recalculation of metrics from released predictions
- Execution of available research scripts
- Full reproduction of original training and corpus generation

The last two activities may require additional dependencies, local path changes, or artifacts not included in the public release.

This document is not a verified end-to-end execution recipe.

## 2. Dataset and Frozen Partitions

The frozen synthetic corpus is distributed in:

`dataset.zip`

It contains 62,895 synthetic monologues associated with 2,000 synthetic user profiles.

The user-disjoint split manifest is:

`configs/research2_user_split_manifest_v1.csv`

The frozen partitions contain:

| Partition | Users | Monologues |
|---|---:|---:|
| Training | 1,400 | 44,029 |
| Validation | 300 | 9,425 |
| Test | 300 | 9,441 |

The original frozen partitions must be preserved when comparing results with the manuscript.

Regenerating random splits would change the evaluation protocol and could introduce differences unrelated to the evaluated models.

## 3. Prediction Task

The benchmark predicts intended ordinal levels for five UPPS-P facets:

- Negative Urgency (NU)
- Positive Urgency (PU)
- Lack of Premeditation (LPr)
- Lack of Perseverance (LPe)
- Sensation Seeking (SS)

Each facet has three classes:

- low
- mid
- high

The labels represent intended synthetic generation conditions rather than independently measured personality traits.

## 4. Transformer Training Protocol

The manuscript reports the following common training settings:

| Parameter | Setting |
|---|---|
| Random seeds | 13, 42, 87 |
| Maximum sequence length | 64 |
| Dropout | 0.30 |
| Optimizer | AdamW |
| Learning rate | 2e-5 |
| Maximum epochs | 5 |
| Batch size | 16 |
| Gradient accumulation steps | 2 |
| Effective batch size | 32 |
| Loss | Class-weighted |
| Early-stopping patience | 2 |
| Checkpoint selection | Validation user-level macro-F1 |

These settings describe the reported protocol.

The public repository contains selected configurations and evaluation records, but not complete training histories and validation-selection records for every original-pooling run.

Therefore, this table alone is insufficient to guarantee exact reproduction of all trained models.

## 5. Evaluated Model Groups

### Original benchmark

Seven pretrained encoder architectures were evaluated with three random seeds each, producing 21 evaluated checkpoints.

### Contemporary-encoder extension

ModernBERT and NeoBERT were evaluated with the same three seeds, adding six original-pooling checkpoints.

The complete original-pooling benchmark therefore includes 27 evaluated checkpoints across nine architectures.

### Exploratory pooling analysis

Six additional mean-pooling checkpoints were trained for ModernBERT and NeoBERT.

The intended modification was replacement of first-token pooling with attention-mask-weighted mean pooling.

The original data partitions, random seeds, classification task, and evaluation procedure were retained.

Because pooling was investigated after inspection of earlier benchmark results, its test comparisons are exploratory and descriptive rather than independently confirmatory.

## 6. Public Evaluation Artifacts

### Original seven encoders

The archive:

`frozen_test_runs.zip`

contains saved frozen-test predictions, evaluation metrics, completion records, and test configuration records for the 21 original runs.

### Contemporary extension

Saved evaluation artifacts for ModernBERT and NeoBERT are retained in:

`results/original_archives/`

Relevant archives include:

- contemporary_extension_final_test_results_v1.zip
- contemporary_extension_final_test_results_v1_FINAL(1).zip

The existence of multiple archived versions should not be interpreted as evidence of independent additional test partitions.

### Exploratory mean pooling

Saved test results are available in:

`results/pooling_extension/frozen_test_runs/`

Training and validation records for the six exploratory mean-pooling runs are available in:

`results/pooling_extension/validation_runs/`

The latter include run configurations, training histories, validation metrics, and validation predictions.

## 7. User-Level Prediction Aggregation

For each synthetic user and each UPPS-P facet:

1. Collect all monologues belonging to that user.
2. Obtain the predicted class probabilities for each monologue.
3. Average the probabilities across that user's monologues.
4. Select the class with the highest mean probability.

The user-level prediction is therefore based on probability aggregation, not majority voting over monologue-level predicted labels.

The primary endpoint is user-level macro-F1 averaged across the five facets.

The test partition contains 300 independent synthetic users.

Multiple model seeds do not increase the number of independent test users.

## 8. Statistical Analysis

The reported evaluation includes:

- 10,000 user-blocked bootstrap replicates for confidence intervals
- 20,000 Monte Carlo paired randomization replicates
- Seed-mean transformer performance estimates
- Holm adjustment within the manuscript-defined comparison families

The bootstrap resampling unit is the synthetic user, not the individual monologue.

The reported comparison families and exploratory status should be preserved when interpreting statistical outputs.

Released statistical results are available in:

`results/statistical_tests/`

Subgroup summaries are available in:

`results/subgroup_analyses/`

## 9. Classical Baselines

The repository contains:

`src/training/run_classical_baselines.py`

Saved baseline predictions and evaluation records are available in:

`results/baseline_predictions/`

These include majority-class, TF-IDF, and risk-based baseline artifacts.

Not every archived baseline is necessarily included in the final manuscript comparisons.

The manuscript and final reviewer response should identify which baseline results are formally reported.

## 10. Software Environment

The repository provides:

`requirements_transformer.txt`

The listed requirements include:

- PyTorch
- Hugging Face Transformers
- Safetensors
- NumPy
- pandas
- SciPy
- scikit-learn

The requirements file specifies package version ranges rather than a fully frozen environment.

A basic environment can be prepared using:

```bash
python -m venv .venv
```

Activate the environment using the appropriate command for the operating system.

Then install the listed dependencies:

```bash
python -m pip install -r requirements_transformer.txt
```

These commands prepare a basic environment only.

They do not establish compatibility with every script or reproduce the original training environment exactly.

Additional packages, compatible model implementations, or local adjustments may be required.

## 11. Available Research Scripts

The repository includes selected scripts in:

- `src/training/`
- `src/evaluation/`
- `src/analysis/`

These cover baseline experiments, transformer training components, frozen-test evaluation, statistical analysis, and figure generation.

Some scripts were preserved from their original execution environments and may contain hard-coded paths or assumptions about locally stored files.

Before execution:

1. Inspect the script's expected input files.
2. Verify the required package versions.
3. Check for hard-coded absolute paths.
4. Confirm the expected archive structure.
5. Direct outputs to a separate working directory.
6. Compare recalculated results against the saved metrics.

No claim is made that every released script executes unchanged after cloning the repository.

## 12. Reproduction Levels

### Level A: Artifact inspection

Supported by the public release.

Researchers can inspect the frozen dataset, split manifests, saved prediction outputs, evaluation metrics, and selected audit records.

### Level B: Metric verification

Partially supported.

The saved predictions and evaluation records provide inputs for independently recalculating reported metrics.

The exact aggregation and statistical procedures must be followed.

Not every analysis script has been verified in a clean environment.

### Level C: Model retraining

Partially supported.

Selected training scripts and configuration records are available.

However, complete training and validation records for all original-pooling runs are not publicly released.

Exact reproduction may also depend on model revisions, tokenizer versions, software versions, hardware behavior, and other environment details.

### Level D: Original-checkpoint inference

Not supported by the public release alone.

The original trained transformer checkpoint weights are not publicly released.

### Level E: Corpus regeneration

Not fully supported by the public release.

The frozen generated dataset is available, but the complete original production generation scripts are not included in the current release snapshot.

Generation details and prompt-conditioning procedures are described in the manuscript and Supplementary Material.

## 13. Integrity and Traceability

The repository includes:

`SHA256SUMS.json`

This file may be used as a reference for recorded artifact checksums.

Checksum coverage and agreement with the current public files should be verified before describing the entire release as cryptographically validated.

Original experiment archives are retained for traceability.

Overlapping copies should not be counted as independent experiments.

## 14. Scientific Limitations

The dataset consists of synthetic text and synthetic profiles.

The intended UPPS-P levels are generation conditions, not independently measured psychometric scores.

The benchmark does not establish clinical diagnostic validity or reliable psychological profiling of real individuals.

External validation against human-written text paired with independently measured UPPS-P traits has not been completed.

The original benchmark, contemporary extension, and exploratory pooling stages should be distinguished when interpreting the reported findings.

## 15. Additional Documentation

For the current public release status, see:

- [Release status](RELEASE_STATUS.md)
- [Repository contents](REPOSITORY_CONTENTS.md)
- [Dataset documentation](../data/README.md)

For methodological details and the interpretation of results, consult the associated manuscript and Supplementary Material.
