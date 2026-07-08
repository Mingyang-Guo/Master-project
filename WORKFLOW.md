# Public Reproduction Workflow

This file gives the command order for the public reproduction workflow.

## Step 1

```bash
cd step1_Get_impression_and_findings_both_exist_data
python all_report_preprocessing.py
python prepare_pilot_inputs.py
python classify_pilot_deepseek.py
python classify_pilot_llama.py
python classify_pilot_medgemma.py
python classify_pilot_qwen.py
python evaluate_deepseek.py
python evaluate_llama.py
python evaluate_MedGemma.py
python evaluate_qwen.py
python result_summarize_and_weight_distribute.py
python plot_classification_confusion_matrices.py
python Four_ensemble_classify.py
```

## Step 2

```bash
cd step2_Model_finetune_and_summarize_evaluate
python prepare_step2_test_manifest.py
python T5/T5_finetun.py
python T5/T5_sum.py
python BART/BART_finetune.py
python BART/BART_sum.py
python RadBARTSum/Fine_Tune_RadBARTSum.py
python RadBARTSum/RadBARTSum_sum.py
python CSTRL/CSTRL_sum.py
python Llama/Llama_sum/Llama_sum.py
python evaluate_all_baselines_ratescore.py
python ttest_all_baselines_public.py
python Visualize_RaTEScore.py
```

## Step 2 Tail

```bash
cd step2_ambiguity_score_0_to_10
python score_0_to_10_ensemble.py --validate-only
python score_0_to_10_ensemble.py
python summarize_score_0_to_10.py
```

## Step 3

```bash
cd step3_subtype
python 00_extract_report_ambiguity_expressions_4llm.py
python 01_prepare_cleaned_expressions.py
python 02_extract_uncertainty_cues_4llm.py
python 03_recount_cue_sentence_frequency.py
python 04_classify_sentence_semantic_subclasses_4llm.py
python 05_build_expression_labels.py
python 06_build_report_level_semantic_subclasses.py
python 07_analyze_ratescore_by_subclass_5models.py
```

Manual review files are placed under `local_private_data/step3/inputs/manual_review/` before scripts `06` and `07`.
