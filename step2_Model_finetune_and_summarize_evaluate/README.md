# Step 2: Baseline Summarization and RaTEScore Evaluation

Step 2 builds the public test manifest, runs five summarization baselines, evaluates generated summaries with RaTEScore and auxiliary metrics, and performs group-level statistical analysis for the public labelled split.

## Inputs

- Step 1 ensemble labels at `local_private_data/step1/generated/llm_impression_ambiguity_weighted_ensemble_4models.jsonl`.
- Split files under `local_private_data/step2/splits/`.
- Baseline checkpoints under `models/` or the configured environment variables.

## Manifest Preparation

```bash
python prepare_step2_test_manifest.py
```

This writes `local_private_data/step2/generated/ready_sum2.csv`.

## Baseline Generation

```bash
python T5/T5_finetun.py
python T5/T5_sum.py
python BART/BART_finetune.py
python BART/BART_sum.py
python RadBARTSum/Fine_Tune_RadBARTSum.py
python RadBARTSum/RadBARTSum_sum.py
python CSTRL/CSTRL_sum.py
python Llama/Llama_sum/Llama_sum.py
```

`CSTRL/CSTRL_sum_directly_using_weights.py` is available for runs that bind a prepared CSTRL checkpoint directly.

## Evaluation and Statistics

```bash
python evaluate_all_baselines_ratescore.py
python ttest_all_baselines_public.py
python Visualize_RaTEScore.py
```

The shared modules `baseline_io.py`, `step2_paths.py`, and `ttest_all_baselines.py` define canonical data formatting, path resolution, and statistical testing logic.

## Outputs

- Baseline predictions: `local_private_data/step2/generated/model_outputs/`.
- Metrics: `local_private_data/step2/generated/metrics/all_baselines/`.
- Statistics: `local_private_data/step2/generated/statistics/`.
- Figures: `local_private_data/step2/generated/visualizations/`.

