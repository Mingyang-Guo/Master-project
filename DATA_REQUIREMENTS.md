# Data and Artifact Placement

This repository stores code and documentation. Place private inputs, generated outputs, manual review sheets, and model checkpoints in the directories below before running the workflow.

## Report Data

- `data/mimic_cxr_reports/`: source MIMIC-CXR report text files.
- `local_private_data/step1/full_reports_121254/`: valid reports with both findings and impression for full-set ambiguity classification.

## Model Checkpoints

- `models/local_llms/DeepSeek/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/`
- `models/local_llms/Llama/LLM-Research/Meta-Llama-3-8B-Instruct/`
- `models/local_llms/MedGemma/google/medgemma-4b-it/`
- `models/local_llms/Qwen/Qwen/Qwen3-8B/`
- `models/huggingface/t5/`
- `models/huggingface/bart/`

Equivalent paths can be supplied through the environment variables documented in `README.md`.

## Step 1 Files

- `local_private_data/step1/pilot/raw_reports/`: sampled pilot reports.
- `local_private_data/step1/pilot/reports_with_impression/`: pilot reports with impression sections.
- `local_private_data/step1/pilot/manual_classify_results/amb/`: manually labelled ambiguous pilot impressions.
- `local_private_data/step1/pilot/manual_classify_results/not_amb/`: manually labelled not-ambiguous pilot impressions.
- `local_private_data/step1/generated/`: Step 1 outputs, including pilot predictions, model weights, and full-set ensemble labels.

## Step 2 Files

- `local_private_data/step2/splits/CXR_train.json`
- `local_private_data/step2/splits/CXR_val.json`
- `local_private_data/step2/splits/CXR_test.json`
- `local_private_data/step2/splits/split_manifest.csv`
- `local_private_data/step2/generated/ready_sum2.csv`: canonical public test manifest.
- `local_private_data/step2/generated/model_outputs/`: generated summaries for T5, BART, RadBARTSum, CSTRL, and Llama.
- `local_private_data/step2/generated/metrics/all_baselines/`: RaTEScore and auxiliary metrics for the public labelled split.

## Step 2 Tail Files

- `local_private_data/step2/generated/ambiguity_score_0_to_10/`: 0-10 ambiguity score outputs and summaries.

## Step 3 Files

- `local_private_data/step3/generated/raw_expression_extraction/`: script `00` outputs.
- `local_private_data/step3/generated/prepared_expressions/`: script `01` outputs.
- `local_private_data/step3/generated/cue_extraction/`: script `02` outputs.
- `local_private_data/step3/generated/sentence_frequency_recount/`: script `03` outputs.
- `local_private_data/step3/generated/semantic_subclass_4llm_2/`: script `04` outputs.
- `local_private_data/step3/generated/expression_labelling/`: script `05` outputs.
- `local_private_data/step3/inputs/manual_review/final_expression_subclass_labels_1668_unified_sorted_manual_check_sorted.csv`: manually reviewed expression labels for script `06`.
- `local_private_data/step3/generated/report_level_semantic_subclasses_17_manual_check/`: script `06` outputs.
- `local_private_data/step3/inputs/manual_review/report_level_semantic_subclasses_augmented.csv`: reviewed report-level subclass sheet for script `07`.
- `local_private_data/step3/generated/ratescore_by_subclass_5models/`: script `07` outputs.
