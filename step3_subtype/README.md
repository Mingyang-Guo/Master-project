# Step 3: Semantic Subtype Analysis

Step 3 extracts uncertainty expressions from ambiguous reports, builds cue-sentence units, assigns semantic uncertainty subclasses, maps manually reviewed expression labels back to report level, and evaluates five baseline models by semantic subclass.

The public workflow uses the Step 2 public ambiguous subset from `local_private_data/step2/generated/metrics/all_baselines/`.

## Script Order

```bash
python 00_extract_report_ambiguity_expressions_4llm.py
python 01_prepare_cleaned_expressions.py
python 02_extract_uncertainty_cues_4llm.py
python 03_recount_cue_sentence_frequency.py
python 04_classify_sentence_semantic_subclasses_4llm.py
python 05_build_expression_labels.py
python 06_build_report_level_semantic_subclasses.py
python 07_analyze_ratescore_by_subclass_5models.py
```

## Script Roles

- `00_extract_report_ambiguity_expressions_4llm.py`: extracts raw uncertainty expressions from ambiguous RadBARTSum report rows.
- `01_prepare_cleaned_expressions.py`: cleans extracted expressions and prepares sentence-level analysis items.
- `02_extract_uncertainty_cues_4llm.py`: extracts target uncertainty cues with four LLMs.
- `03_recount_cue_sentence_frequency.py`: recounts cue-sentence frequency after cue extraction.
- `04_classify_sentence_semantic_subclasses_4llm.py`: assigns semantic subclasses to cue-sentence units.
- `05_build_expression_labels.py`: aggregates four-model subclass labels into expression-level review tables.
- `06_build_report_level_semantic_subclasses.py`: maps reviewed expression labels back to report-level semantic subclasses.
- `07_analyze_ratescore_by_subclass_5models.py`: evaluates five baseline models across reviewed semantic subclasses.
- `step3_paths.py`: central path configuration.
- `core/`: implementation modules used by the public entry scripts.

## Manual Review Inputs

Before script `06`, place `final_expression_subclass_labels_1668_unified_sorted_manual_check_sorted.csv` under `local_private_data/step3/inputs/manual_review/`.

Before script `07`, place `report_level_semantic_subclasses_augmented.csv` under `local_private_data/step3/inputs/manual_review/`.

## Author-Provided Review Files

Scripts `06` and `07` can start from author-provided manual-review files. This is part of the public workflow: readers may place the released reviewed CSV files in `local_private_data/step3/inputs/manual_review/` and continue directly to report-level mapping and subclass-level evaluation. Script `06` uses the current script `05` output as the bridge between the released reviewed labels and the current run.

## Core Modules

The `core/` directory contains implementation modules used by the public entry scripts:

- `core/__init__.py`: marks the core implementation package.
- `core/extract_report_ambiguity_expressions_text_only_core.py`: implementation for script `00`.
- `core/prepare_cleaned_expressions_core.py`: implementation for script `01`.
- `core/extract_uncertainty_cues_4llm_core.py`: implementation for script `02`.
- `core/recount_cue_sentence_frequency_core.py`: implementation for script `03`.
- `core/classify_sentence_semantic_subclasses_4llm_core.py`: implementation for script `04`.
- `core/build_expression_labels_core.py`: implementation for script `05`.
- `core/build_report_level_semantic_subclasses_core.py`: implementation for script `06`.
- `core/analyze_ratescore_5_models_reviewed_core.py`: implementation for script `07`.
