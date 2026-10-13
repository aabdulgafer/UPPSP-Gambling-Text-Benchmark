# Reproducibility protocol (manuscript-derived summary)

- Use the original frozen user-disjoint partitions; do not regenerate random splits.
- Predict five intended UPPS-P facets (negative urgency, positive urgency, lack of premeditation, lack of perseverance, sensation seeking), each with low/mid/high classes.
- Main encoder training: seeds 13, 42, 87; maximum length 64; dropout 0.30; AdamW learning rate 2e-5; up to five epochs; batch size 16 and gradient accumulation 2; class-weighted loss; early stopping patience 2; checkpoint selection using validation user-level macro-F1.
- Average per-monologue predicted class probabilities for each user before taking argmax.
- Bootstrap: 10,000 user-blocked replicates; paired randomization: 20,000 Monte Carlo replicates; Holm-adjust within the comparison families stated in the manuscript.
- Keep the later ModernBERT/NeoBERT mean-pooling results explicitly exploratory; original test results were known before this extension.

These instructions summarize the manuscript, **not** a verified execution recipe. Verify the exact scripts, package versions, output paths and model identifiers before claiming reproduction.
