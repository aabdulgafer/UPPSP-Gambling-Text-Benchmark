
# Dataset Documentation

## Overview

This directory documents the frozen synthetic dataset used in the UPPS-P Gambling Text Benchmark.

The dataset is publicly available in the repository root:

[dataset.zip](../dataset.zip)

The archive contains synthetic English social-media-style monologues generated under intended UPPS-P impulsivity conditioning.

The assigned facet levels are synthetic generation targets, not independently measured psychological traits.

## Dataset Summary

| Property | Value |
|---|---:|
| Synthetic user profiles | 2,000 |
| Monologues | 62,895 |
| Parent topics | 150 |
| Subtopics | 1,200 |
| Gambling/addiction-context monologues | 41,340 |
| Generic-context monologues | 21,555 |

## Archive Contents

The dataset archive contains the following files:

### 1_Generated_Subtopics.csv

Generated subtopics used to guide monologue generation.

### 2_User_profiles_extended.csv

Synthetic user profiles containing intended UPPS-P conditioning variables and additional profile information.

### 3_Generated_monologues.csv

The frozen corpus of 62,895 synthetic monologues used in the benchmark.

### Participation_Matrix_M.csv

Participation records associated with the synthetic generation process.

### used_dimensions_log.csv

Records of behavioral dimensions used during text generation.

## Intended UPPS-P Facets

The benchmark considers five dimensions:

- Negative Urgency (NU)
- Positive Urgency (PU)
- Lack of Premeditation (LPr)
- Lack of Perseverance (LPe)
- Sensation Seeking (SS)

Each facet is represented by three intended ordinal levels: low, mid, and high.

The labels describe intended synthetic conditioning rather than psychometrically validated measurements.

## Frozen Data Partitions

The benchmark uses user-disjoint training, validation, and test partitions.

| Partition | Users | Monologues |
|---|---:|---:|
| Training | 1,400 | 44,029 |
| Validation | 300 | 9,425 |
| Test | 300 | 9,441 |
| Total | 2,000 | 62,895 |

The split manifest is available at:

[research2_user_split_manifest_v1.csv](../configs/research2_user_split_manifest_v1.csv)

The same frozen test partition was used throughout the original benchmark and subsequent exploratory analyses.

## Data Integrity

The released corpus underwent cleaning for duplicated and corrupted records before the benchmark partitions were frozen.

The published benchmark results refer to the frozen dataset and should not be interpreted as results on regenerated or modified data.

## Scope and Limitations

The corpus contains synthetic text and synthetic user profiles.

The intended UPPS-P levels are not clinical diagnoses or independently measured psychological characteristics.

The dataset has not been externally validated against paired human-written texts and independently measured UPPS-P questionnaire scores.

The dataset should not be used for nonconsensual psychological profiling or decisions affecting real individuals.

## Licensing and Reuse

The original synthetic dataset is licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) license, as specified in the repository's [LICENSE](../LICENSE) file.

The license permits sharing and adaptation for noncommercial purposes, provided that appropriate attribution is given, a link to the license is included, and any modifications are indicated.

Third-party materials, if present, remain subject to their respective licenses and terms of use.

## Additional Documentation

For experimental configurations, evaluation artifacts, and reproduction details, see the repository's main README and the documentation in the `docs/` directory.
