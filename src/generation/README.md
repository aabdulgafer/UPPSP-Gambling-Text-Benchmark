
# Synthetic Corpus Generation

## Overview

This directory contains the Python scripts associated with the generation of the synthetic corpus used in the UPPS-P Gambling Text Benchmark.

The corpus was generated using synthetic user profiles, topic and subtopic information, and intended UPPS-P conditioning.

The resulting texts are synthetic English social-media-style monologues. The intended UPPS-P labels are generation conditions rather than independently measured psychological traits.

## Generation Scripts

The directory contains:

- `config_M.py` — generation configuration and input/output settings
- `main_M.py` — generation workflow and model execution
- `prompts_M.py` — prompt templates and conditioning instructions
- `quality_metrics_M.py` — generation quality-assessment functions
- `utils_M.py` — supporting generation utilities

These files document the implemented generation pipeline. They should be interpreted together with the manuscript and its Supplementary Material.

## Language Models

The study used two language-model families for synthetic corpus construction:

- Mistral-7B-Instruct-v0.3 for topic/subtopic and synthetic profile generation
- Meta-Llama-3-8B-Instruct for synthetic monologue generation

Model identifiers, local model paths, loading options, and generation settings should be checked against the corresponding scripts and the manuscript.

The model weights are not distributed in this directory.

## Conditioning

The synthetic profiles specify intended levels for five UPPS-P facets:

- Negative Urgency (NU)
- Positive Urgency (PU)
- Lack of Premeditation (LPr)
- Lack of Perseverance (LPe)
- Sensation Seeking (SS)

The generation prompts use these conditioning variables to guide linguistic expression.

The resulting texts do not establish that the intended psychological constructs were independently measured or psychometrically validated.

## Generation Settings

The monologue-generation protocol reported in the manuscript includes:

- Temperature: 0.6
- Top-p: 0.9
- Maximum generated tokens: 128

Other settings, including model-loading parameters and GPU offloading, may depend on the execution environment.

These parameters describe the reported generation protocol. They do not guarantee identical outputs when rerunning stochastic language-model generation.

## Required Inputs

The generation workflow relies on supporting inputs, including:

- Parent topics and generated subtopics
- Synthetic user profiles
- Prompt templates
- Local language-model files
- Configuration and output paths

The released frozen dataset is available at the repository root:

[dataset.zip](../../dataset.zip)

The archive contains the generated subtopics, user profiles, monologues, participation records, and behavioral-dimension usage records.

The original source topic file and any additional auxiliary input files must be supplied separately if they are required by the published scripts.

## Reproduction Limitations

The generation scripts are provided to improve methodological transparency.

They should not be interpreted as a fully verified, self-contained corpus-regeneration package.

Exact regeneration may depend on:

- The original input files
- The specific language-model weights and quantization formats
- Software and library versions
- Random-number generation and execution environment
- Local model-loading and file-path configurations
- Post-generation filtering and cleaning procedures

The released dataset is the frozen research artifact used for the reported benchmark.

Regenerated text should not be substituted for the frozen corpus when attempting to reproduce the published evaluation results.

## Relationship to the Manuscript

The manuscript and Supplementary Material describe the corpus-generation methodology, prompt-conditioning procedures, decoding settings, and quality-control process.

The public scripts provide implementation-level evidence supporting those descriptions.

Any differences between the published scripts and the original production environment should be documented explicitly rather than silently treated as identical.

## Responsible Use

The generated corpus contains synthetic content and synthetic user profiles.

The intended UPPS-P labels are not clinical diagnoses or independently measured personality characteristics.

The generation pipeline and its outputs should not be used for nonconsensual psychological profiling or decisions affecting real individuals.
