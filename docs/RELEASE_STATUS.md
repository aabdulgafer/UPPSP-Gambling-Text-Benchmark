
# Release Status

## Overview

This document describes the public release status of the research artifacts supporting the manuscript:

"Beyond the Big Five: Benchmarking Pretrained Language Encoders for Predicting Intended UPPS-P Facet Levels from Synthetic Gambling-Related Text."

It distinguishes artifacts included in this repository from materials retained internally.

Public availability does not imply that every original training experiment can be reproduced exactly from the released files.

## 1. Dataset

**Status: Publicly available**

The frozen synthetic dataset is included in the repository root:

- `dataset.zip`

The archive contains:

- 62,895 synthetic monologues
- 2,000 synthetic user profiles
- Generated subtopics
- Participation records
- Behavioral-dimension usage records

The frozen user-disjoint split manifest is available at:

- `configs/research2_user_split_manifest_v1.csv`

Additional information is provided in `data/README.md`.

## 2. Source Code

**Status: Partially released**

The repository contains scripts for transformer training, evaluation, statistical analysis, and baseline experiments.

Relevant directories include:

- `src/training/`
- `src/evaluation/`
- `src/analysis/`

The original production scripts used for synthetic corpus generation are not included in this release snapshot.

The available scripts should not be interpreted as a complete end-to-end reproduction package for all experimental stages.

## 3. Model Configurations

**Status: Partially released**

The repository includes configuration records for evaluated transformer checkpoints within archived test-result packages.

The original benchmark evaluated seven pretrained encoders with three random seeds each (21 checkpoints).

A contemporary-encoder extension added ModernBERT and NeoBERT with three seeds each (six additional checkpoints), yielding 27 original-pooling checkpoints.

A subsequent exploratory analysis evaluated six additional mean-pooling checkpoints.

The repository also contains:

- `configs/example_neobert_seed87_run_config.json`
- `results/training_histories/example_neobert_seed87_training_history.json`

These two files are examples from one run and do not constitute complete training histories for all 27 original-pooling checkpoints.

The complete set of original training configurations, training histories, and validation-selection records is not provided as a standalone public release.

## 4. Test Predictions and Evaluation Results

**Status: Publicly available for multiple experimental stages**

The repository contains:

- `frozen_test_runs.zip`
- `results/original_archives/`

These archives include saved test predictions, evaluation metrics, and configuration records from the original benchmark, contemporary-encoder extension, and exploratory pooling experiments.

The repository also provides selected statistical comparison results in:

- `results/statistical_tests/`

And subgroup evaluation summaries in:

- `results/subgroup_analyses/`

These released outputs support examination of reported evaluation results but do not replace the unavailable trained model checkpoint weights.

## 5. Classical Baseline Artifacts

**Status: Publicly available as archived results**

The directory:

- `results/baseline_predictions/`

contains prediction exports and evaluation summaries for several classical baseline configurations.

The included baseline export audit records a PASS result and a runner SHA-256 matching the released baseline script:

- `src/training/run_classical_baselines.py`

This audit does not establish independent reproduction of all transformer experiments.

The presence of archived baseline results does not mean that every archived baseline is included in the final manuscript comparisons.

## 6. Model Checkpoints

**Status: Not publicly released**

Large trained transformer checkpoint weights are not included in this repository.

The corresponding experimental results and selected configuration records are retained in released evaluation archives.

Reproducing model inference from the original trained weights is not possible using this repository alone.

## 7. Corpus Generation Materials

**Status: Not fully released**

The repository contains the frozen generated dataset but does not currently include the complete original production scripts used to generate the synthetic corpus.

Generation methodology, language models, decoding settings, and prompt-conditioning details are described in the manuscript and accompanying Supplementary Material.

The released dataset should not be interpreted as sufficient for exact regeneration of the corpus.

## 8. Additional Validation and Audit Materials

**Status: Partially released**

The repository includes selected evaluation, statistical-analysis, and baseline audit artifacts.

The presence of these outputs does not establish that all post hoc human-validation records, lexical-diversity audits, and other supplementary analyses are publicly released.

The public availability of individual audit materials should be verified against the repository contents and the associated Supplementary Material.

## 9. Dependencies and Reproduction

The repository includes:

- `requirements_transformer.txt`
- `docs/reproducibility.md`

These files provide dependency information and existing reproduction documentation.

The current release does not guarantee complete end-to-end reproduction of the original corpus generation, all transformer training runs, or inference using the original trained checkpoints.

## 10. Licensing

The applicable code and dataset licenses have not yet been finalized in this release snapshot.

Public accessibility should not be interpreted as unrestricted permission to reuse or redistribute the materials.

Licensing information will be updated after confirmation by the authors and relevant rights holders.

## 11. Experimental and Scientific Limitations

The dataset contains synthetic monologues and synthetic user profiles.

The prediction targets represent intended UPPS-P conditioning levels, not independently measured psychological traits.

The contemporary-encoder extension and exploratory pooling experiments were conducted after earlier test results had been inspected.

Consequently, the exploratory results are descriptive and are not independently confirmed by a new held-out user partition.

External validation against human-written text paired with independently measured UPPS-P scores has not been completed.
