
# Public Artifact Inventory

## Purpose and Scope

This inventory documents the release status of research artifacts supporting the manuscript:

"Beyond the Big Five: Benchmarking Pretrained Language Encoders for Predicting Intended UPPS-P Facet Levels from Synthetic Gambling-Related Text."

It distinguishes publicly released files from experimental materials retained internally.

The inventory is based on the repository release and should be updated whenever files are added, replaced, or removed.

## Release Status Definitions

- **Public:** The identified artifact is included in the repository.
- **Partial:** Selected files or experiment stages are public, but the complete artifact collection is not.
- **Not public:** The artifact is not distributed in the repository.
- **Pending:** Release or licensing status requires confirmation.

Public release does not necessarily establish independent reproducibility.

## Artifact Inventory

| Artifact category | Public status | Repository location | Scope and limitations |
|---|---|---|---|
| Frozen synthetic corpus | Public | `dataset.zip` | Contains 62,895 synthetic monologues, 2,000 profiles, and related generation outputs |
| User-disjoint split manifest | Public | `configs/research2_user_split_manifest_v1.csv` | Defines the frozen user partitions |
| Frozen-test manifests | Public | `configs/` | Includes original and contemporary-extension manifests |
| Corpus-generation scripts | Public, subject to final verification | `src/generation/` | Five production-related Python scripts uploaded after the original repository snapshot |
| Generation prompt templates | Public, subject to final verification | `src/generation/prompts_M.py` | Contains generation prompt definitions; manuscript Supplementary Material provides methodological context |
| Original topic input and auxiliary generation files | Partial | `dataset.zip`; other inputs not confirmed | Generated subtopics and profiles are released; complete original auxiliary input collection has not been verified |
| Transformer training scripts | Partial | `src/training/` | Selected scripts are available; a fully verified training pipeline for every architecture is not established |
| Original seven-encoder test configurations | Public | `frozen_test_runs.zip` | Contains test configuration records for 21 runs |
| Contemporary-extension test configurations | Public | `results/original_archives/contemporary_extension_final_test_results_v1_FINAL(1).zip` | Contains test configuration records for the six ModernBERT/NeoBERT original-pooling runs |
| Original-pooling training configurations and histories | Partial | results/training_runs/ | Configuration and history records for 26 verified balanced-weighting runs; historical BERT seed-42 configuration requires separate provenance verification |
| Exploratory mean-pooling training records | Public | `results/pooling_extension/validation_runs/` | Six runs with configurations, training histories, validation metrics, and validation predictions |
| Original seven-encoder test predictions | Public | `frozen_test_runs.zip` | Saved frozen-test prediction files for 21 runs |
| Contemporary-extension test predictions | Public | `results/original_archives/` | Saved predictions for six ModernBERT/NeoBERT original-pooling runs |
| Exploratory mean-pooling test predictions | Public | `results/pooling_extension/frozen_test_runs/` | Saved test predictions and metrics for six runs |
| Classical baseline predictions | Public | `results/baseline_predictions/` | Includes majority, TF-IDF, and risk-based baseline outputs; not all archived baselines are necessarily reported in the manuscript |
| Statistical comparison outputs | Public | `results/statistical_tests/` | Includes comparison tables and confidence-interval results |
| Subgroup analysis summaries | Public | `results/subgroup_analyses/` | Includes frozen-test subgroup summaries |
| Confusion-matrix data and figure | Public | `results/confusion_matrices/`; `figures/` | Released figure and underlying CSV |
| Baseline export audit | Public | `results/baseline_predictions/baseline_prediction_export_audit.json` | Documents a baseline export audit; not a full independent experiment reproduction |
| Other evaluation and audit records | Partial | `results/`; `results/original_archives/` | Selected evaluation records are released; not all internally retained audits have been verified as public |
| Human annotation records | Not confirmed as public | No verified repository path | Human-validation methodology and results are described in the manuscript and Supplementary Material |
| Original trained transformer checkpoint weights | Not public | None | Checkpoint weights are retained internally and are not distributed |
| Original model and tokenizer weights | Not public | None | Third-party pretrained model files must be obtained separately, subject to applicable terms |
| Transformer dependency specification | Public | `requirements_transformer.txt` | Package version ranges; not a fully locked environment |
| Artifact checksum records | Public | `SHA256SUMS.json` | Recorded checksums; coverage and agreement with current files require verification |
| Code and dataset licenses | Pending | No verified license files | Applicable licensing terms require author and rights-holder confirmation |

## Experimental Stage Distinctions

### Original benchmark

The original benchmark evaluated seven pretrained encoder architectures using seeds 13, 42, and 87, producing 21 evaluated checkpoints.

Saved frozen-test predictions, metrics, and test configuration records are released.

Complete original training and validation records and checkpoint weights are not publicly available.

### Contemporary-encoder extension

ModernBERT and NeoBERT added six original-pooling checkpoints.

Saved test evaluation artifacts are available in the archived extension results.

The extension used the existing frozen dataset partitions.

### Exploratory pooling analysis

Six additional checkpoints were trained using attention-mask-weighted mean pooling.

Their validation and test records are publicly available.

This analysis was conducted after inspection of earlier test results. Its test comparisons are exploratory and should not be treated as independent confirmation.

## Relationship to Appendix A

Appendix A of the manuscript describes the internally retained reproducibility record.

The internally retained record is broader than the public repository release.

In particular, retention of model checkpoints, full prediction records, or audit materials by the researchers does not imply that every corresponding file is publicly available.

The public repository contains saved predictions for the evaluated transformer stages, selected statistical outputs, and some audit records.

Large trained transformer checkpoint weights are not publicly released.

Complete original-pooling training and validation histories are also not included in the current release.

## Reproducibility Limitations

The released materials support inspection of the frozen corpus and examination of saved evaluation outputs.

Some analyses may be recalculated from the public predictions and metrics.

However, exact end-to-end reproduction is not established because the public release does not include all original training records, trained model weights, auxiliary generation inputs, and fully locked software environments.

The intended UPPS-P facet levels are synthetic conditioning labels, not independently measured psychological traits.

## Licensing

The public repository's licensing terms require confirmation.

Until applicable licenses are published, repository accessibility should not be interpreted as unrestricted permission for reuse or redistribution.

## Maintenance

This inventory should be updated after:

1. Adding or replacing generation materials.
2. Publishing additional training configurations or audit records.
3. Confirming and publishing code and dataset licenses.
4. Creating an archived version or persistent DOI.
5. Making any change to the set of publicly released artifacts.

The manuscript Data Availability Statement should be consistent with the inventory for the version of the repository cited in the article.

### Original-Pooling Training Records

**Repository location:** `results/training_runs/`

The repository contains training configurations (`run_config.json`) and training histories (`training_history.json`) for 26 original-pooling classification runs with explicitly recorded balanced class weighting.

These records cover the available verified encoder–seed combinations from the original benchmark. The historical BERT-base-uncased seed-42 run is tracked separately because its configuration does not explicitly record the class-weighting setting.

The published training records support inspection of training hyperparameters, epoch-level progression, and validation-based checkpoint selection. They do not constitute a complete release of model checkpoints or all intermediate evaluation artifacts.

The BERT seed-42 provenance remains subject to reconciliation with the frozen final-test manifest. No alternative weighting configuration should be attributed to this run without supporting evidence.
