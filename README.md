
# UPPS-P Gambling Text Benchmark

Research artifacts supporting the manuscript:

**Beyond the Big Five: Benchmarking Pretrained Language Encoders for Predicting Intended UPPS-P Facet Levels from Synthetic Gambling-Related Text**

## Overview

This study benchmarks nine pretrained language encoders for predicting five intended UPPS-P impulsivity facet levels from synthetic English social-media-style monologues.

The five facets are:

- Negative Urgency (NU)
- Positive Urgency (PU)
- Lack of Premeditation (LPr)
- Lack of Perseverance (LPe)
- Sensation Seeking (SS)

Each facet is represented by three intended levels: low, mid, and high.

The labels are synthetic generation conditions, not independently measured psychological traits.

## Dataset

The frozen corpus contains:

- 62,895 synthetic monologues
- 2,000 synthetic user profiles
- 41,340 gambling/addiction-context monologues
- 21,555 generic-context monologues
- 150 parent topics and 1,200 subtopics

The dataset is available in the repository:

[Download dataset.zip](dataset.zip)

The frozen user-disjoint partitions are:

| Partition | Users | Monologues |
|---|---:|---:|
| Training | 1,400 | 44,029 |
| Validation | 300 | 9,425 |
| Test | 300 | 9,441 |
| Total | 2,000 | 62,895 |

The user-level split manifest is available at:

`configs/research2_user_split_manifest_v1.csv`

## Experimental Setup

Nine pretrained encoder architectures were evaluated:

1. BERT
2. RoBERTa
3. ALBERT
4. ELECTRA
5. XLNet
6. ERNIE 2.0
7. DeBERTa-v3
8. ModernBERT
9. NeoBERT

Each encoder was evaluated using three random seeds (13, 42, and 87).

The original benchmark comprised seven encoders (21 checkpoints). A subsequent extension added ModernBERT and NeoBERT (six checkpoints).

An exploratory analysis evaluated six additional mean-pooling configurations for ModernBERT and NeoBERT.

All evaluations used the same frozen user-disjoint test partition. The later experimental stages were conducted after earlier test results had been inspected; consequently, exploratory pooling findings are not independently confirmed.

## Repository Structure

- `dataset.zip` — frozen synthetic dataset
- `configs/` — split manifests and configuration records
- `data/` — dataset documentation
- `results/` — evaluation results, predictions, statistical comparisons, and archived experiment outputs
- `frozen_test_runs.zip` — original frozen-test evaluation artifacts
- `figures/` — manuscript figures
- `docs/` — reproducibility and release documentation
- `requirements_transformer.txt` — transformer environment dependencies

## Evaluation

The primary endpoint is user-level macro-F1 averaged across the five intended UPPS-P facets.

Supporting evaluations include monologue-level classification, facet-specific metrics, ordinal error measures, confidence intervals, and analyses across gambling-risk strata and elicitation contexts.

User-level predictions are obtained by averaging the predicted class probabilities across monologues belonging to the same synthetic user.

## Reproducibility and Artifact Availability

The repository includes frozen dataset partitions, archived test predictions, statistical outputs, and training configurations and histories for all 27 original-pooling encoder runs. The training records are provided in results/training_runs/transformer_training_audit.zip.

The repository also includes training and validation records for the six exploratory mean-pooling runs.

Source code for corpus generation, model training, evaluation, and analysis, as well as trained model checkpoint weights, is retained privately and is not included in the public release. The available configurations and saved outputs support artifact inspection and selected metric verification, not exact end-to-end reproduction.

For the current scope and limitations of the public release, see:

- [Release status](docs/RELEASE_STATUS.md)
- [Repository contents](docs/REPOSITORY_CONTENTS.md)
- [Reproducibility documentation](docs/reproducibility.md)

## Research Scope and Limitations

The benchmark investigates the recovery of intended UPPS-P conditioning from synthetically generated text.

Its results do not establish psychometric validity, clinical diagnostic capability, or reliable personality inference for real individuals.

The models and dataset should not be used for nonconsensual psychological profiling, surveillance, or decisions affecting real individuals.

## Citation

Please cite the associated research article when referring to this dataset or benchmark.

## License

The original synthetic dataset and original research documentation in this repository are licensed under the **Creative Commons Attribution–NonCommercial 4.0 International (CC BY-NC 4.0)** license, except where otherwise indicated. This license permits sharing and adaptation with appropriate attribution for noncommercial purposes.

- [License overview](https://creativecommons.org/licenses/by-nc/4.0/)
- [Full legal terms](https://creativecommons.org/licenses/by-nc/4.0/legalcode)
- [Repository license file](LICENSE)

Third-party materials remain subject to their respective licenses and terms of use. Source code for corpus generation, model training, evaluation, and statistical analysis, and trained model checkpoint weights, are retained privately and are not included in the public release or covered by this license.

For academic use, please cite the associated research publication when available and identify the repository version used.
