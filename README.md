# UPPS-P Gambling Text Benchmark

Research artifacts accompanying *Beyond the Big Five: Benchmarking Pretrained Language Encoders for Recovering UPPS-P Impulsivity Traits in Gambling-Related Text* (manuscript under preparation for Electronics).

The study evaluates intended five-facet UPPS-P conditioning in **62,895 synthetic monologues** from **2,000 synthetic user profiles**, with nine pretrained language encoders and three seeds (13, 42, 87). The held-out test set contains 300 users and 9,441 monologues. The post-hoc pooling extension is separately identified. These labels are synthetic generation conditions, not measured psychological traits or diagnostic ground truth.

## Status
This is a **curated draft release**, not yet a fully executable reproduction package. Files under `src/` were recovered from prior research uploads and have not yet been independently checked against the exact manuscript run environment. The training entry point for the seven original encoders, baseline training script, original frozen prediction exports, full dependencies, and some configuration artifacts remain to be added. See `docs/MISSING_FILES.md`.

## Dataset
The dataset ZIP was uploaded separately by the authors. Add its actual filename or download URL here and document its schema and license. Do not silently substitute a different cleaned dataset revision.

## Evaluation
Frozen user-disjoint split: training 1,400 users / 44,029 monologues; validation 300 / 9,425; test 300 / 9,441. Primary endpoint: mean across five facet-level user macro-F1 scores. Original benchmark: nine encoders × three seeds. Exploratory extension: masked mean pooling for ModernBERT and NeoBERT.

## Provenance
See `docs/FILE_MANIFEST.json` for file origins and SHA-256 checksums. Original filenames are retained to avoid implying that recovered files were the only or final production scripts.

## Citation and licensing
Citation metadata and licenses must be approved by the authors before a public release. No license is granted merely by publishing this folder.
