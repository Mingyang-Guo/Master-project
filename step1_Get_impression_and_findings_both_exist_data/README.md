# Step 1: Report Extraction and Ambiguity Classification

Step 1 prepares MIMIC-CXR report text, runs a pilot ambiguity classification experiment, estimates four-model ensemble weights, and applies the ensemble to the full valid report set.

## Inputs

- Pilot reports and manual labels under `local_private_data/step1/pilot/`.
- Full valid reports under `local_private_data/step1/full_reports_121254/`.
- Four local instruction LLM checkpoints configured through `CLASSIFY_FOUR_LLMS_DIR` or per-model environment variables.

## Scripts

- `all_report_preprocessing.py`: extracts reports with usable findings and impression sections.
- `prepare_pilot_inputs.py`: builds the pilot input layout.
- `classify_pilot_deepseek.py`, `classify_pilot_llama.py`, `classify_pilot_medgemma.py`, `classify_pilot_qwen.py`: run the four pilot classifiers.
- `evaluate_deepseek.py`, `evaluate_llama.py`, `evaluate_MedGemma.py`, `evaluate_qwen.py`: evaluate pilot classifiers against manual labels.
- `result_summarize_and_weight_distribute.py`: summarize pilot metrics and write ensemble weights.
- `plot_classification_confusion_matrices.py`: create classifier diagnostic figures.
- `Four_ensemble_classify.py`: run full-set four-LLM ambiguity classification.
- `ambiguity_classifier_common.py` and `evaluate_classifier_common.py`: shared helpers.

## Outputs

Generated files are written under `local_private_data/step1/generated/`. The Step 1 label file consumed by Step 2 is `llm_impression_ambiguity_weighted_ensemble_4models.jsonl`.

