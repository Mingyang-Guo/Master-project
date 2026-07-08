# Step 2 Tail: 0-10 Ambiguity Scoring

This module assigns fine-grained ambiguity scores to reference impressions and generated summaries for the public labelled reproduction split.

## Input

The scorer reads five baseline metric files from `local_private_data/step2/generated/metrics/all_baselines/`. Each file must include `group`, `reference`, and `prediction` columns.

## Commands

```bash
python score_0_to_10_ensemble.py --validate-only
python score_0_to_10_ensemble.py
python summarize_score_0_to_10.py
```

## Outputs

Outputs are written under `local_private_data/step2/generated/ambiguity_score_0_to_10/public/` and `local_private_data/step2/generated/ambiguity_score_0_to_10/visualizations/`.

