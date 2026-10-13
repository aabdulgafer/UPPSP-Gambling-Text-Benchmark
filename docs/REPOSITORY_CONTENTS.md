# Repository contents and reproducibility status

This release combines the supplied benchmark resources with recovered frozen-test and pooling experiment outputs. Experiment artifacts are retained byte-for-byte.

## Results layout

- `results/main_benchmark/frozen_test_runs/`: frozen-test artifacts for the seven original encoders (21 runs).
- `results/main_benchmark/contemporary_frozen_test_runs/`: frozen-test artifacts for ModernBERT and NeoBERT (six runs).
- `results/pooling_extension/frozen_test_runs/`: frozen-test artifacts for six masked-mean pooling runs.
- `results/pooling_extension/validation_runs/`: training configurations, histories, validation metrics, and predictions for the six pooling runs.
- `results/training_runs/neobert_classification_balanced_seed87/`: supplied training configuration and history for this one run.
- `results/original_archives/`: original experiment archives preserved for traceability; their contents may duplicate expanded result files.

## Limitations

This package does not establish the availability of full training histories, validation metrics, and validation predictions for all 27 main benchmark runs. Only one main-run training history is available in the assembled resources. Full independent training reproduction has not been verified. The frozen modeling dataset is not included in this ZIP; use the separately released dataset and verify its SHA-256 against the experiment configuration. The manuscript and repository should describe the release accordingly.

The repository has not been executed end-to-end in a clean environment. Exact dependency versions appear in individual run configurations; `requirements_transformer.txt` contains version ranges rather than a fully pinned environment.

Extract the ZIP and upload the *contents* of the top-level `UPPSP-Gambling-Text-Benchmark` folder to the repository root. Do not upload the outer ZIP as a single file if individual-file browsing is desired.
