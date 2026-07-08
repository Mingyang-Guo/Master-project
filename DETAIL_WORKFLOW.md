# Detailed Reproduction Workflow and Outputs

This document expands `WORKFLOW.md` with the expected files produced by each runnable workflow script. The repository intentionally stores code and Markdown only. Runtime artifacts such as CSV, XLSX, JSONL, SVG, model weights, and private data are generated locally under `local_private_data/` and are not committed.

The paths below are relative to the repository root unless stated otherwise. Most scripts also accept CLI arguments that can override the default input and output paths.

## Shared Runtime Roots

- `local_private_data/step1/`: Step 1 private inputs and generated ambiguity-classification outputs.
- `local_private_data/step2/`: Step 2 split files, generated manifests, model predictions, metrics, statistics, and visualizations.
- `local_private_data/step3/`: Step 3 manual-review inputs and generated subtype-analysis outputs.
- `local_private_data/models/`: local fine-tuned summarization checkpoints used by Step 2.
- `models/local_llms/`: local instruction LLM checkpoints used by the four-LLM ambiguity and subtype workflows, unless overridden by environment variables.

## Optional Model Download Helpers

These helpers are not part of the main `WORKFLOW.md` command chain, but they prepare base checkpoints for later training.

### `python models/T5_base_download.py`

Outputs:

- `models/huggingface/t5/`, or the directory set by `T5_BASE_DIR`.

Description:

Downloads or materializes the base T5 checkpoint and tokenizer for local use. The directory is a model checkpoint artifact and should not be committed.

### `python models/BART_large_cnn_download.py`

Outputs:

- `models/huggingface/bart/`, or the directory set by `BART_BASE_DIR`.

Description:

Downloads or materializes the BART-large-CNN checkpoint and tokenizer for local use. The directory is a model checkpoint artifact and should not be committed.

## Step 1: Report Extraction and Four-LLM Ambiguity Classification

Run from:

```bash
cd step1_Get_impression_and_findings_both_exist_data
```

### `python all_report_preprocessing.py`

Primary outputs:

- `local_private_data/step1/full_reports_121254/`: copied report text files that contain both FINDINGS and IMPRESSION sections.
- `local_private_data/step1/full_reports_121254/filter_summary.csv`: per-file filtering record with keep/drop status and drop reason.

Console output:

- Prints the input report root and output root.
- Prints total scanned, kept, dropped, and drop-reason counts.
- Prints the path of `filter_summary.csv`.

Description:

Filters the source report corpus to the subset usable for findings-to-impression summarization and later ambiguity classification.

### `python prepare_pilot_inputs.py`

Primary outputs:

- No new canonical public result file is written by this script.
- It validates and summarizes files already placed under `local_private_data/step1/pilot/`, including raw reports, reports with IMPRESSION, and manual labels.

Console output:

- Prints raw report count.
- Prints reports with IMPRESSION count.
- Prints manual ambiguous and not-ambiguous label counts.
- Prints reports with both FINDINGS and IMPRESSION.
- Prints strict-subset label distributions when applicable.

Description:

Checks that the pilot ambiguity-classification input layout and manual labels are internally consistent before running the pilot classifiers.

### `python classify_pilot_deepseek.py`

Primary outputs:

- `local_private_data/step1/generated/llm_impression_ambiguity_DeepSeek_R1_Distill_Qwen7B_vllm.xlsx`
- `local_private_data/step1/generated/llm_impression_ambiguity_DeepSeek_R1_Distill_Qwen7B_vllm.jsonl`

Console output:

- Prints progress as completed rows accumulate.
- Prints the final saved XLSX and JSONL paths with the number of predictions.

Description:

Runs the DeepSeek local instruction model on pilot impressions and stores structured ambiguity predictions.

### `python classify_pilot_llama.py`

Primary outputs:

- `local_private_data/step1/generated/llm_impression_ambiguity_Llama3_8B_vllm.xlsx`
- `local_private_data/step1/generated/llm_impression_ambiguity_Llama3_8B_vllm.jsonl`

Console output:

- Prints progress as completed rows accumulate.
- Prints the final saved XLSX and JSONL paths with the number of predictions.

Description:

Runs the Llama local instruction model on pilot impressions and stores structured ambiguity predictions.

### `python classify_pilot_medgemma.py`

Primary outputs:

- `local_private_data/step1/generated/llm_impression_ambiguity_MedGemma_vllm.xlsx`
- `local_private_data/step1/generated/llm_impression_ambiguity_MedGemma_vllm.jsonl`

Console output:

- Prints progress as completed rows accumulate.
- Prints the final saved XLSX and JSONL paths with the number of predictions.

Description:

Runs the MedGemma local instruction model on pilot impressions and stores structured ambiguity predictions.

### `python classify_pilot_qwen.py`

Primary outputs:

- `local_private_data/step1/generated/llm_impression_ambiguity_Qwen3_8B_vllm.xlsx`
- `local_private_data/step1/generated/llm_impression_ambiguity_Qwen3_8B_vllm.jsonl`

Console output:

- Prints progress as completed rows accumulate.
- Prints the final saved XLSX and JSONL paths with the number of predictions.

Description:

Runs the Qwen local instruction model on pilot impressions and stores structured ambiguity predictions.

### `python evaluate_deepseek.py`

Primary outputs:

- `local_private_data/step1/generated/deepseek_evaluation_using_manual_GT.txt`
- `local_private_data/step1/generated/deepseek_evaluation_rows.csv`

Console output:

- Prints the path of the saved evaluation text file.

Description:

Compares the DeepSeek pilot predictions against `manual_GT.xlsx` and writes both a human-readable report and row-level evaluation table.

### `python evaluate_llama.py`

Primary outputs:

- `local_private_data/step1/generated/llama_evaluation_using_manual_GT.txt`
- `local_private_data/step1/generated/llama_evaluation_rows.csv`

Console output:

- Prints the path of the saved evaluation text file.

Description:

Compares the Llama pilot predictions against `manual_GT.xlsx` and writes both a human-readable report and row-level evaluation table.

### `python evaluate_MedGemma.py`

Primary outputs:

- `local_private_data/step1/generated/medgemma_evaluation_using_manual_GT.txt`
- `local_private_data/step1/generated/medgemma_evaluation_rows.csv`

Console output:

- Prints the path of the saved evaluation text file.

Description:

Compares the MedGemma pilot predictions against `manual_GT.xlsx` and writes both a human-readable report and row-level evaluation table.

### `python evaluate_qwen.py`

Primary outputs:

- `local_private_data/step1/generated/qwen_evaluation_using_manual_GT.txt`
- `local_private_data/step1/generated/qwen_evaluation_rows.csv`

Console output:

- Prints the path of the saved evaluation text file.

Description:

Compares the Qwen pilot predictions against `manual_GT.xlsx` and writes both a human-readable report and row-level evaluation table.

### `python result_summarize_and_weight_distribute.py`

Primary outputs:

- `local_private_data/step1/generated/ensemble_review_rows.csv`
- `local_private_data/step1/generated/model_weights.csv`

Console output:

- Prints pilot report count.
- Prints how many manual labels changed after five-voter majority review.
- Prints final label distribution.
- Prints eight-decimal ensemble parameters.
- Prints the saved structured-review path.

Description:

Combines pilot predictions and manual labels, derives the ensemble review table, and writes model weights used by the full-set weighted ensemble.

### `python plot_classification_confusion_matrices.py`

Primary outputs:

- `local_private_data/step1/generated/visualizations/classification_cm/<model>_confusion_matrix_counts.csv`
- `local_private_data/step1/generated/visualizations/classification_cm/weighted_ensemble_confusion_matrix_counts.csv`
- `local_private_data/step1/generated/visualizations/classification_cm/classification_confusion_matrix_summary.csv`

Console output:

- Prints the output directory containing classification confusion-matrix tables.

Description:

Creates count tables for single-model and weighted-ensemble confusion matrices. The committed code version writes CSV summaries; downstream plotting can use these tables.

### `python Four_ensemble_classify.py`

Primary outputs:

- `local_private_data/step1/generated/ensemble_cache/<model>.jsonl`: per-model full-set prediction cache.
- `local_private_data/step1/generated/ensemble_cache/<model>.metadata.json`: cache metadata for final assembly checks.
- `local_private_data/step1/generated/llm_impression_ambiguity_weighted_ensemble_4models.xlsx`
- `local_private_data/step1/generated/llm_impression_ambiguity_weighted_ensemble_4models.jsonl`

Console output:

- Prints discovered report count.
- Prints fixed eight-decimal global model weights.
- Prints per-model cached, pending, and total counts.
- Prints per-model completion progress.
- Prints final label distribution.
- Prints final XLSX and JSONL paths.
- If any model cache is incomplete, prints that final assembly is deferred.

Description:

Runs the four local LLMs across the full valid report set, caches per-model predictions, then assembles weighted report-level ambiguity labels. The JSONL output is the canonical Step 1 label input for Step 2.

## Step 2: Summarization, Metrics, Statistics, and Visualization

Run from:

```bash
cd step2_Model_finetune_and_summarize_evaluate
```

### `python prepare_step2_test_manifest.py`

Primary outputs:

- `local_private_data/step2/generated/ready_sum2.csv`
- `local_private_data/step2/generated/ready_sum2_audit.json`

Console output:

- Prints the audit JSON, including input paths, record counts before and after deduplication, removed duplicate count, label counts, and output path.

Description:

Joins the test split to Step 1 ensemble labels, removes duplicate findings/impression pairs, and creates the canonical public Step 2 input manifest used by all baseline inference scripts.

### `python T5/T5_finetun.py`

Primary outputs:

- `local_private_data/step2/generated/training/T5/epoch_<N>_rouge_<score>.bin`
- `local_private_data/step2/generated/training/T5/BEST_epoch_<N>_valid_rouge_<score>_model_weights.bin`
- `local_private_data/step2/generated/training/T5/training_meta.json`
- `local_private_data/models/T5_final/`: exported HuggingFace-format final model directory.

Console output:

- Prints device, loaded train/validation/test sample counts, and model path.
- Prints training loss progress.
- Prints validation ROUGE per epoch.
- Prints saved epoch weights and best-weight paths.
- Prints final exported model directory and metadata path.

Description:

Fine-tunes the T5 baseline and exports the checkpoint consumed by `T5_sum.py`.

### `python T5/T5_sum.py`

Primary outputs:

- `local_private_data/step2/generated/model_outputs/T5/predictions_all.csv`
- `local_private_data/step2/generated/model_outputs/T5/predictions_ambiguous.csv`
- `local_private_data/step2/generated/model_outputs/T5/predictions_not_ambiguous.csv`
- `local_private_data/step2/generated/model_outputs/T5/run_summary.json`

Console output:

- Prints device and fine-tuned model path.
- Prints generation retry warnings if any batch fails temporarily.
- Prints a JSON run summary and `===== DONE =====`.

Description:

Generates T5 summaries for the canonical Step 2 manifest and writes standardized prediction tables for all, ambiguous, and not-ambiguous subsets.

### `python BART/BART_finetune.py`

Primary outputs:

- `local_private_data/step2/generated/training/BART/epoch_<N>_rouge_<score>.bin`
- `local_private_data/step2/generated/training/BART/BEST_epoch_<N>_valid_rouge_<score>_model_weights.bin`
- `local_private_data/step2/generated/training/BART/training_meta.json`
- `local_private_data/models/BART_final/`: exported HuggingFace-format final model directory.

Console output:

- Prints device, loaded train/validation/test sample counts, and model path.
- Prints training loss progress.
- Prints validation ROUGE per epoch.
- Prints saved epoch weights and best-weight paths.
- Prints final exported model directory and metadata path.

Description:

Fine-tunes the BART baseline and exports the checkpoint consumed by `BART_sum.py`.

### `python BART/BART_sum.py`

Primary outputs:

- `local_private_data/step2/generated/model_outputs/BART/predictions_all.csv`
- `local_private_data/step2/generated/model_outputs/BART/predictions_ambiguous.csv`
- `local_private_data/step2/generated/model_outputs/BART/predictions_not_ambiguous.csv`
- `local_private_data/step2/generated/model_outputs/BART/all/sample_level_results.csv`
- `local_private_data/step2/generated/model_outputs/BART/all/summary_metrics.json`

Console output:

- Prints dataset name, input path, valid sample count, local model path, and tokenizer path.
- Prints generation retry warnings if any batch fails temporarily.
- Prints sample-level result path.
- Prints summary metrics path and the JSON summary.
- Prints `All done.` at completion.

Description:

Generates BART summaries, writes standardized prediction subsets, and computes local generation-quality summaries for the full dataset.

### `python RadBARTSum/Fine_Tune_RadBARTSum.py`

Primary outputs:

- `local_private_data/step2/generated/training/RadSumBART/epoch_<N>_rouge_<score>.bin`
- `local_private_data/step2/generated/training/RadSumBART/BEST_epoch_<N>_valid_rouge_<score>_model_weights.bin`
- `local_private_data/step2/generated/training/RadSumBART/training_meta.json`
- `local_private_data/models/RadSumBART_final/`: exported HuggingFace-format final model directory.

Console output:

- Prints device, loaded train/validation/test sample counts, and model path.
- Prints training loss progress.
- Prints validation ROUGE per epoch.
- Prints saved epoch weights and best-weight paths.
- Prints final exported model directory and metadata path.

Description:

Fine-tunes the RadBARTSum baseline and exports the checkpoint consumed by `RadBARTSum_sum.py`.

### `python RadBARTSum/RadBARTSum_sum.py`

Primary outputs:

- `local_private_data/step2/generated/model_outputs/RadSumBART/predictions_all.csv`
- `local_private_data/step2/generated/model_outputs/RadSumBART/predictions_ambiguous.csv`
- `local_private_data/step2/generated/model_outputs/RadSumBART/predictions_not_ambiguous.csv`
- `local_private_data/step2/generated/model_outputs/RadSumBART/all/sample_level_results.csv`
- `local_private_data/step2/generated/model_outputs/RadSumBART/all/summary_metrics.json`

Console output:

- Prints dataset name, input path, valid sample count, local model path, and tokenizer path.
- Prints generation retry warnings if any batch fails temporarily.
- Prints sample-level result path.
- Prints summary metrics path and the JSON summary.
- Prints `All done.` at completion.

Description:

Generates RadBARTSum summaries, writes standardized prediction subsets, and computes local generation-quality summaries for the full dataset.

### `python CSTRL/CSTRL_sum.py`

Primary outputs:

- `local_private_data/step2/generated/model_outputs/CSTRL/predictions_all.csv`
- `local_private_data/step2/generated/model_outputs/CSTRL/predictions_ambiguous.csv`
- `local_private_data/step2/generated/model_outputs/CSTRL/predictions_not_ambiguous.csv`
- `local_private_data/step2/generated/model_outputs/CSTRL/run_summary.json`

Console output:

- Prints device and local CSTRL model path.
- Prints generation retry warnings if any batch fails temporarily.
- Prints a JSON run summary and `===== DONE =====`.

Description:

Generates CSTRL summaries for the canonical Step 2 manifest and writes standardized prediction tables for all, ambiguous, and not-ambiguous subsets.

### `python Llama/Llama_sum/Llama_sum.py`

Primary outputs:

- `local_private_data/step2/generated/model_outputs/Llama/predictions_all.csv`
- `local_private_data/step2/generated/model_outputs/Llama/predictions_ambiguous.csv`
- `local_private_data/step2/generated/model_outputs/Llama/predictions_not_ambiguous.csv`
- `local_private_data/step2/generated/model_outputs/Llama/run_summary.json`

Console output:

- Prints input CSV, output directory, and model path.
- Prints generation retry warnings if any batch fails temporarily.
- Prints a JSON run summary and `===== DONE =====`.

Description:

Generates Llama summaries for the canonical Step 2 manifest and writes standardized prediction tables for all, ambiguous, and not-ambiguous subsets.

### `python evaluate_all_baselines_ratescore.py`

Primary outputs:

- `local_private_data/step2/generated/metrics/all_baselines/<MODEL>_metrics_all.csv`
- `local_private_data/step2/generated/metrics/all_baselines/<MODEL>_metrics_ambiguous.csv`
- `local_private_data/step2/generated/metrics/all_baselines/<MODEL>_metrics_not_ambiguous.csv`
- `local_private_data/step2/generated/metrics/all_baselines/all_baselines_metrics_summary.csv`
- `local_private_data/step2/generated/metrics/all_baselines/all_baselines_metrics_outputs.json`

Console output:

- Prints the summary metric table.
- Prints the saved summary path.
- Prints the saved output-index JSON path.

Description:

Computes RaTEScore and auxiliary metrics for each baseline and subset. The `<MODEL>_metrics_all.csv` files become key inputs for Step 2 Tail and Step 3.

### `python ttest_all_baselines_public.py`

Primary outputs:

- `local_private_data/step2/generated/statistics/all_baselines_ttest_public/all_baselines_welch_ttest.csv`
- `local_private_data/step2/generated/statistics/all_baselines_ttest_public/all_baselines_welch_ttest.json`

Console output:

- Prints the Welch t-test table.
- Prints saved CSV and JSON paths.

Description:

Runs public split Welch t-tests comparing ambiguous and not-ambiguous groups across baseline metrics.

### `python Visualize_RaTEScore.py`

Primary outputs:

- `local_private_data/step2/generated/visualizations/ratescore_visualization_long_data.csv`
- `local_private_data/step2/generated/visualizations/ratescore_visualization_summary.csv`
- `local_private_data/step2/generated/visualizations/ratescore_ttest_for_visualization.csv`
- `local_private_data/step2/generated/visualizations/<dataset>_ratescore_mean_by_group.svg`
- `local_private_data/step2/generated/visualizations/<dataset>_ratescore_box_summary.svg`
- `local_private_data/step2/generated/visualizations/<dataset>_ratescore_delta_ci.svg`

Console output:

- Prints the visualization output directory.

Description:

Builds Step 2 RaTEScore summary tables and SVG visualizations for group means, box-style summaries, and delta confidence intervals.

## Step 2 Tail: 0-10 Ambiguity Scoring

Run from:

```bash
cd step2_ambiguity_score_0_to_10
```

### `python score_0_to_10_ensemble.py --validate-only`

Primary outputs:

- No score output is written in validation-only mode.

Console output:

- For each baseline, prints validated row counts, scoring item counts, and the input metric CSV path.

Description:

Checks that the five baseline metric files contain the required columns and can be converted into reference/prediction scoring items.

### `python score_0_to_10_ensemble.py`

Primary outputs:

- `local_private_data/step2/generated/ambiguity_score_0_to_10/public/<MODEL>_score_0_to_10_ambiguous.csv`
- `local_private_data/step2/generated/ambiguity_score_0_to_10/public/cache/<MODEL>_ambiguous_<LLM>.jsonl`
- `local_private_data/step2/generated/ambiguity_score_0_to_10/public/cache/<MODEL>_ambiguous_<LLM>.errors.jsonl`, only when a model response fails parsing.

Console output:

- Prints per-model cache progress for new scoring items.
- Prints `Saved: <output_path>` for each baseline score CSV.

Description:

Uses the four local LLMs to score ambiguity intensity from 0 to 10 for references and model predictions, then assembles per-baseline score tables.

### `python summarize_score_0_to_10.py`

Primary outputs:

- `local_private_data/step2/generated/ambiguity_score_0_to_10/visualizations/`: summary CSV files and visualization-ready score tables for the 0-10 scoring outputs.

Console output:

- Prints the output root for the saved 0-10 score summaries.

Description:

Summarizes score shifts between references and generated summaries and prepares aggregate data for later reporting or plotting.

## Step 3: Semantic Uncertainty Subtype Analysis

Run from:

```bash
cd step3_subtype
```

### `python 00_extract_report_ambiguity_expressions_4llm.py`

Primary outputs:

- `local_private_data/step3/generated/raw_expression_extraction/predictions_with_metrics_labeled_no_repeat.csv`
- `local_private_data/step3/generated/raw_expression_extraction/ambiguity_expressions_extracted_4_models_text_only.csv`

Console output:

- Prints input CSV path and row counts.
- Prints per-model load path, preview snippets, parse warnings if any, processed counts, extracted-item counts, and parse-failure counts.
- Prints final output path, original row count, ambiguous rows processed, output row count, extraction model counts, and extracted-any counts.

Description:

Prepares a Step 3-compatible input from RadBARTSum metrics and extracts raw ambiguity expressions from ambiguous reports using four local LLMs.

### `python 01_prepare_cleaned_expressions.py`

Primary outputs:

- `local_private_data/step3/generated/prepared_expressions/cleaned_sentences_for_manual_review.csv`
- `local_private_data/step3/generated/prepared_expressions/pattern_inventory_all_deduplicated.csv`
- `local_private_data/step3/generated/prepared_expressions/pattern_inventory_parser_signature_variants.csv`
- `local_private_data/step3/generated/prepared_expressions/pattern_inventory_for_manual_review.csv`
- `local_private_data/step3/generated/prepared_expressions/sentence_pattern_matches_long.csv`
- `local_private_data/step3/generated/prepared_expressions/sentence_patterns_wide.csv`
- `local_private_data/step3/generated/prepared_expressions/unmatched_sentences.csv`
- `local_private_data/step3/generated/prepared_expressions/data_preparation_summary.csv`
- `local_private_data/step3/generated/prepared_expressions/pattern_extraction_config.json`

Console output:

- Prints input loading path.
- Prints cleaned and deduplicated row count.
- Prints the saved cleaned-sentences path.
- Prints spaCy model name.
- Prints parsing progress and pattern-match counts.
- Prints all saved output paths.
- Prints the top deduplicated patterns.

Description:

Cleans raw expression text, deduplicates sentence-level items, extracts parser-based pattern inventories, and prepares the cue-extraction input.

### `python 02_extract_uncertainty_cues_4llm.py`

Primary outputs:

- `local_private_data/step3/generated/cue_extraction/cue_sentence_level.csv`
- `local_private_data/step3/generated/cue_extraction/cue_sentence_level.xlsx`
- `local_private_data/step3/generated/cue_extraction/cue_sentence_level.jsonl`
- `local_private_data/step3/generated/cue_extraction/cue_frequency.csv`
- `local_private_data/step3/generated/cue_extraction/cue_frequency.xlsx`
- `local_private_data/step3/generated/cue_extraction/cue_model_matrix.csv`
- `local_private_data/step3/generated/cue_extraction/cue_model_matrix.xlsx`

Console output:

- Prints input CSV path and valid row count.
- Prints per-model load path and batch progress.
- Prints all saved cue-level, frequency, and matrix output paths.

Description:

Uses four local LLMs to identify target diagnostic uncertainty cues in each cleaned sentence, then writes sentence-level detail, cue frequencies, and model-agreement matrices.

### `python 03_recount_cue_sentence_frequency.py`

Primary outputs:

- `local_private_data/step3/generated/sentence_frequency_recount/cue_sentence_frequency_summary.csv`
- `local_private_data/step3/generated/sentence_frequency_recount/cue_sentence_frequency_long_detail.csv`

Console output:

- Prints input CSV path.
- Prints saved summary and long-detail paths.
- Prints the top 20 cues, or reports that no cues were found.

Description:

Recounts cue occurrence at sentence level after cue extraction and writes both summary and long-format detail tables.

### `python 04_classify_sentence_semantic_subclasses_4llm.py`

Primary outputs:

- `local_private_data/step3/generated/semantic_subclass_4llm_2/cue_semantic_subclasses_4llm.csv`
- `local_private_data/step3/generated/semantic_subclass_4llm_2/sentence_semantic_subclasses_4llm.csv`
- `local_private_data/step3/generated/semantic_subclass_4llm_2/cue_label_sentence_statistics_4llm.csv`
- `local_private_data/step3/generated/semantic_subclass_4llm_2/manual_review_subclasses/<LABEL_ID>.csv`: label-specific manual-review tables.

Console output:

- Prints model loading and pending item counts.
- Prints prompt-token and generation-token diagnostics.
- Prints warnings with raw output if JSON validation fails.
- Prints `Wrote <N> rows: <destination>` for each generated table.
- Prints manual-review table creation messages.

Description:

Classifies each cue-sentence unit into the 17 semantic uncertainty subclasses with four local LLMs and prepares both machine summaries and manual-review tables.

### `python 05_build_expression_labels.py`

Primary outputs:

- `local_private_data/step3/generated/expression_labelling/expression_labels_4llm.csv`
- `local_private_data/step3/generated/expression_labelling/unique_expression_labels_4llm.csv`
- `local_private_data/step3/generated/expression_labelling/expression_label_statistics.csv`
- `local_private_data/step3/generated/expression_labelling/unlabelled_expressions_for_manual_review.csv`
- `local_private_data/step3/generated/expression_labelling/unresolved_expressions_for_manual_review.csv`
- `local_private_data/step3/generated/expression_labelling/expression_labelling_summary.json`

Console output:

- Prints a JSON summary including input paths, row counts, manual-review-required counts, and output paths.

Description:

Aggregates sentence-level semantic subclass predictions into expression-level label tables, highlights unresolved or unlabelled expressions, and records summary statistics for manual review.

### Manual Review Before Script 06

Before running script `06`, place this reviewed file under:

- `local_private_data/step3/inputs/manual_review/final_expression_subclass_labels_1668_unified_sorted_manual_check_sorted.csv`

Description:

This file is the manually checked expression-level subclass table consumed by report-level mapping.

### `python 06_build_report_level_semantic_subclasses.py`

Primary outputs:

- `local_private_data/step3/generated/report_level_semantic_subclasses_17_manual_check/report_level_semantic_subclasses.csv`
- `local_private_data/step3/generated/report_level_semantic_subclasses_17_manual_check/report_subclass_membership_long.csv`
- `local_private_data/step3/generated/report_level_semantic_subclasses_17_manual_check/expression_to_report_mapping.csv`
- `local_private_data/step3/generated/report_level_semantic_subclasses_17_manual_check/nonliteral_expression_matches_for_review.csv`
- `local_private_data/step3/generated/report_level_semantic_subclasses_17_manual_check/excluded_empty_expression_rows.csv`
- `local_private_data/step3/generated/report_level_semantic_subclasses_17_manual_check/final_label_alignment.csv`
- `local_private_data/step3/generated/report_level_semantic_subclasses_17_manual_check/report_level_summary.json`
- `local_private_data/step3/generated/report_level_semantic_subclasses_17_manual_check/README.md`
- `local_private_data/step3/generated/report_level_semantic_subclasses_17_manual_check/subsets/<SUBCLASS>.csv`
- `local_private_data/step3/generated/report_level_semantic_subclasses_17_manual_check/subsets/NO.csv`

Console output:

- Prints a JSON summary with row counts, output paths, subset paths, and alignment diagnostics.

Description:

Maps manually reviewed expression-level labels back to report-level semantic subclass memberships and writes subset-specific report tables.

### Manual Review Before Script 07

Before running script `07`, place this reviewed file under:

- `local_private_data/step3/inputs/manual_review/report_level_semantic_subclasses_augmented.csv`

Description:

This file is the reviewed report-level subclass table consumed by subclass-level RaTEScore analysis.

### `python 07_analyze_ratescore_by_subclass_5models.py`

Primary outputs:

Per-model outputs under `local_private_data/step3/generated/ratescore_by_subclass_5models/<MODEL>/`:

- `selected_reports_with_reviewed_labels.csv`
- `ratescore_report_subclass_membership_long.csv`
- `ratescore_summary_by_subclass.csv`
- `ratescore_summary_by_subclass.md`

Cross-model outputs under `local_private_data/step3/generated/ratescore_by_subclass_5models/`:

- `five_model_ratescore_summary_long.csv`
- `five_model_ratescore_summary_wide.csv`
- `five_model_overall_summary.csv`
- `five_model_report_scores_wide.csv`
- `five_model_ratescore_comparison.xlsx`
- `analysis_summary.json`
- `README.md`

Console output:

- Prints an analysis summary JSON with input paths, selected model counts, and output locations.

Description:

Combines reviewed report-level subclass labels with the five baseline RaTEScore metric files and produces per-subclass performance summaries for all models.


## Optional Runnable Utilities

These scripts are directly runnable, but they are not part of the default command sequence in `WORKFLOW.md`.

### `python step2_Model_finetune_and_summarize_evaluate/ttest_all_baselines.py`

Primary outputs:

- `local_private_data/step2/generated/statistics/all_baselines_ttest/all_baselines_welch_ttest.csv`
- `local_private_data/step2/generated/statistics/all_baselines_ttest/all_baselines_welch_ttest.json`

Console output:

- Prints the Welch t-test table.
- Prints saved CSV and JSON paths.

Description:

Runs the same Welch t-test workflow as the public wrapper, but writes to the non-public default statistics directory unless `--output-dir` is overridden.

### `python step2_Model_finetune_and_summarize_evaluate/CSTRL/CSTRL_sum_directly_using_weights.py`

Primary outputs:

- `local_private_data/step2/generated/model_outputs/CSTRL/predictions_all.csv`
- `local_private_data/step2/generated/model_outputs/CSTRL/predictions_ambiguous.csv`
- `local_private_data/step2/generated/model_outputs/CSTRL/predictions_not_ambiguous.csv`
- `local_private_data/step2/generated/model_outputs/CSTRL/run_summary.json`

Console output:

- Prints device and local fine-tuned CSTRL path.
- Prints generation retry warnings if any batch fails temporarily.
- Prints a JSON run summary and `===== DONE =====`.

Description:

Alternative CSTRL inference entry point that loads an already fine-tuned CSTRL checkpoint directly from local weights and writes the same standardized prediction files as `CSTRL_sum.py`.

## Supporting Python Modules

The workflow also includes shared implementation modules that are imported by the runnable scripts and are not normally executed directly:

- `step1_Get_impression_and_findings_both_exist_data/ambiguity_classifier_common.py`: shared four-LLM pilot classification logic.
- `step1_Get_impression_and_findings_both_exist_data/evaluate_classifier_common.py`: shared pilot evaluation logic.
- `step2_Model_finetune_and_summarize_evaluate/baseline_io.py`: shared Step 2 prediction schema validation and writers.
- `step2_Model_finetune_and_summarize_evaluate/step2_paths.py`: central Step 2 path configuration.
- `step3_subtype/step3_paths.py`: central Step 3 path configuration and path safety checks.
- `step3_subtype/core/*.py`: Step 3 implementation modules used by the public entry scripts.

These modules may write outputs only when called through the entry scripts listed above or when invoked by a developer with custom arguments.
