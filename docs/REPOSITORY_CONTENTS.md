
# Repository Contents and Reproducibility Status

## Overview

This document describes the organization of the publicly released research artifacts supporting the UPPS-P Gambling Text Benchmark.

The repository contains the frozen synthetic dataset, user-disjoint split manifests, selected source code, saved test predictions, evaluation results, statistical outputs, and exploratory pooling records.

Some experiment artifacts are distributed as ZIP archives rather than expanded directories.

The repository includes training configurations and histories for all 27 original-pooling runs, but does not contain complete validation records or trained checkpoint weights for every run.

## 1. Dataset

### dataset.zip

Location: Repository root

Contains the frozen synthetic corpus and related generation outputs:

- 1_Generated_Subtopics.csv
- 2_User_profiles_extended.csv
- 3_Generated_monologues.csv
- Participation_Matrix_M.csv
- used_dimensions_log.csv

The dataset contains 62,895 synthetic monologues associated with 2,000 synthetic user profiles.

The frozen user-disjoint split manifest is:

`configs/research2_user_split_manifest_v1.csv`

Additional dataset information is available in `data/README.md`.

## 2. Configuration Files

Location: `configs/`

Includes:

- research2_user_split_manifest_v1.csv
- frozen_test_manifest(2).csv
- contemporary_extension_frozen_manifest_v1(1).csv
- example_neobert_seed87_run_config.json

The NeoBERT configuration is an example from one original-pooling training run, not a complete collection of training configurations for all 27 runs.

Additional test configuration records are contained within the experiment-result archives.

## 3. Original Seven-Encoder Benchmark

Location: `frozen_test_runs.zip`

This archive contains frozen-test evaluation artifacts for the original seven pretrained encoder architectures, evaluated using three random seeds each (21 runs).

The run directories contain:

- completion.json
- test_metrics.json
- test_predictions.csv
- test_run_config.json

These files document completed test evaluations, prediction outputs, and associated test configurations.

They do not contain the trained model checkpoint weights.

## 4. Contemporary-Encoder Extension

Location: `results/original_archives/`

Relevant archives include:

- contemporary_extension_final_test_results_v1.zip
- contemporary_extension_final_test_results_v1_FINAL(1).zip

These archives contain saved evaluation outputs associated with the ModernBERT and NeoBERT extension.

The extension adds six original-pooling runs, corresponding to two encoders evaluated using three seeds each.

The repository retains the original archives for traceability.

The archives should not be interpreted as separate independent experiments merely because multiple versions are preserved.

## 5. Exploratory Mean-Pooling Analysis

Location: `results/pooling_extension/`

### Frozen-test results

`results/pooling_extension/frozen_test_runs/`

Contains six run directories:

- modernbert_base_masked_mean_seed13
- modernbert_base_masked_mean_seed42
- modernbert_base_masked_mean_seed87
- neobert_masked_mean_seed13
- neobert_masked_mean_seed42
- neobert_masked_mean_seed87

Each run includes:

- test_manifest.json
- test_metrics.json
- test_predictions.csv

### Validation and training records

`results/pooling_extension/validation_runs/`

Contains the corresponding six run directories.

Each includes:

- run_config.json
- training_history.json
- validation_metrics.json
- validation_predictions.csv

The mean-pooling experiments were exploratory and were conducted after earlier benchmark test results had been inspected.

They do not constitute independent confirmation on a new test partition.

## 6. Original Training Records

The consolidated archive is available at:

`results/training_runs/transformer_training_audit.zip`

It contains training configurations (`run_config.json`) and training histories (`training_history.json`) for all 27 original-pooling encoder–seed combinations.

The repository also provides an illustrative NeoBERT seed-87 configuration and training history as separately accessible files.

Complete validation metrics and predictions are not available for every original-pooling run. Trained transformer checkpoint weights are not publicly released.

The BERT-base-uncased seed-42 configuration does not explicitly document its class-weighting setting. This limitation is retained for provenance transparency.

## 7. Classical Baseline Results

Location: `results/baseline_predictions/`

Includes:

- dummy_most_frequent_test_predictions.csv
- tfidf_logistic_regression_test_predictions.csv
- tfidf_linear_svm_test_predictions.csv
- risk_only_logistic_regression_test_predictions.csv
- risk_ogdq_logistic_regression_test_predictions.csv
- final_test_baselines_v1.json
- reproduced_baseline_metrics.json
- baseline_prediction_export_audit.json

These files are retained as research artifacts.

Their presence does not imply that every archived baseline is reported in the final manuscript.

## 8. Statistical and Subgroup Analyses

### Statistical results

Location: `results/statistical_tests/`

Includes:

- primary_pairwise_comparisons.csv
- extension_pairwise_statistics_seedmean_v1.csv
- model_confidence_intervals.csv

### Subgroup analyses

Location: `results/subgroup_analyses/`

Includes:

- final_test_subgroup_summary.csv

### Confusion matrices

Location: `results/confusion_matrices/`

Includes:

- uppsp_modernbert_confusion_matrices_data.csv

These outputs support examination of reported evaluation results.

## 9. Source Code

Location: `src/`

The repository includes:

### Training

- src/training/run_bert_pilot_v1.py
- src/training/run_classical_baselines.py
- src/training/run_contemporary_encoder_extension_v1(2).py

### Evaluation

- src/evaluation/evaluate_frozen_test_v1.py
- src/evaluation/evaluate_contemporary_extension_final_test_v1(1).py
- src/evaluation/export_frozen_baseline_predictions_v1.py
- src/evaluation/freeze_test_manifest_v2.py

### Analysis

- src/analysis/create_uppsp_confusion_matrices.py
- src/analysis/run_extension_statistical_analysis_seedmean_v1_3.py

These scripts provide partial reproduction and analysis functionality.

Five production-related corpus-generation scripts are included in `src/generation/`: `config_M.py`, `main_M.py`, `prompts_M.py`, `quality_metrics_M.py`, and `utils_M.py`. Their availability does not, by itself, establish complete reproducibility of corpus generation.

## 10. Figures

Location: `figures/`

Includes:

- aggregation.png
- experimental_framework.png
- uppsp_modernbert_confusion_matrices.png

## 11. Dependencies and Documentation

The repository contains:

- requirements_transformer.txt
- SHA256SUMS.json
- README.md
- data/README.md
- docs/RELEASE_STATUS.md
- docs/REPOSITORY_CONTENTS.md
- docs/reproducibility.md

The dependency file contains version ranges rather than a completely frozen software environment.

The repository has not been verified through an end-to-end execution in a clean environment.

## 12. Original Experiment Archives

Location: `results/original_archives/`

This directory preserves original evaluation archives for traceability.

Some archive contents overlap with files available elsewhere in the repository.

These archives should not be counted as additional independent experimental runs.

## 13. Public Release Limitations

The following materials are not completely available in this repository snapshot:

- Additional inputs, dependencies, and environment details potentially required for complete corpus regeneration
- Complete validation metrics and predictions for every original-pooling run
- Large trained transformer checkpoint weights
- Complete independently verified end-to-end reproduction environment
- All post hoc validation and audit materials

The repository supports inspection and partial reproduction of the benchmark but does not establish exact end-to-end reproduction of every experiment.

The prediction targets represent intended synthetic UPPS-P conditioning levels rather than independently measured psychological traits.

External validation using human-written text paired with independently measured UPPS-P scores has not been completed.

## 14. Licensing

The applicable licenses for the code and dataset require confirmation by the authors and relevant rights holders.

Public accessibility does not itself grant unrestricted permission for reuse or redistribution.
