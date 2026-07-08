# Radiology Report Ambiguity Reproducibility Code

This repository provides the public reproduction workflow for a radiology report ambiguity study. The workflow starts from MIMIC-CXR report text, identifies report-level ambiguity with a four-LLM ensemble, evaluates five summarization baselines under ambiguous and not-ambiguous report conditions, assigns 0-10 ambiguity scores, and maps uncertainty expressions to semantic subclasses for report-level analysis.

The public workflow uses the public Step 2 test set with report-level ambiguity labels. Private report text, generated model outputs, manual review sheets, model checkpoints, and metric artifacts are not included in this repository.

## Repository Layout

- `models/`: model download helpers and checkpoint placement notes.
- `data/`: source report corpus placement notes.
- `local_private_data/`: private input and generated output placement tree.
- `step1_Get_impression_and_findings_both_exist_data/`: report extraction and four-LLM ambiguity classification.
- `step2_Model_finetune_and_summarize_evaluate/`: baseline summarization, RaTEScore evaluation, t-tests, and visualization.
- `step2_ambiguity_score_0_to_10/`: fine-grained 0-10 ambiguity scoring.
- `step3_subtype/`: uncertainty expression extraction, cue extraction, semantic subclass mapping, and subclass-level RaTEScore analysis.

## Environment Variables

The scripts resolve paths relative to the repository root by default. These variables can bind external resources:

- `MASTER_PROJECT_ROOT`: repository root.
- `LOCAL_PRIVATE_DATA_DIR`: private data root. Defaults to `local_private_data/`.
- `CLASSIFY_FOUR_LLMS_DIR`: root directory for the four local instruction LLM checkpoints.
- `DEEPSEEK_MODEL_PATH`, `LLAMA_MODEL_PATH`, `MEDGEMMA_MODEL_PATH`, `QWEN_MODEL_PATH`: per-model checkpoint paths.
- `T5_BASE_DIR`, `BART_BASE_DIR`: baseline checkpoint locations for the download helpers.

## Reproduction Order

1. Follow `step1_Get_impression_and_findings_both_exist_data/README.md` to prepare reports and run the four-LLM ambiguity classifier.
2. Follow `step2_Model_finetune_and_summarize_evaluate/README.md` to prepare the public test manifest, run five summarization baselines, compute RaTEScore, and run statistical tests.
3. Follow `step2_ambiguity_score_0_to_10/README.md` to generate 0-10 ambiguity scores for reference impressions and model predictions.
4. Follow `step3_subtype/README.md` to extract expressions and cues, build semantic subclasses, map reviewed labels to report level, and evaluate subclass-level RaTEScore.

## Data Placement

See `DATA_REQUIREMENTS.md` for the exact private file layout. The repository includes README files in data directories so that the directory contract is visible without storing private files.
