# Japan AI Cup Experiments

## v2 baseline (time-aware CV)
- date: 2026-02-12
- script: `competitions/japan-ai-cup/scripts/train_lgbm_v2.py`
- feature set:
  - deterministic latest profile (`arg_max` by `date`)
  - refund-aware features (`refund_txn_count`, `refund_txn_ratio`, `refund_amount_ratio`)
  - recency window features (30/60/90d transaction/spend shares)
  - recent-vs-all diversity (`cat1_recent_ratio`, `cat1_diversity_change_30_90`)
- cv setup:
  - primary: expanding time buckets by `last_purchase_date`
  - reference: random stratified 5-fold
- outputs:
  - `competitions/japan-ai-cup/predictions/submission_v2.csv`
  - `competitions/japan-ai-cup/predictions/oof_v2.csv`
  - `competitions/japan-ai-cup/predictions/feature_importance_v2.csv`
- result:
  - TimeCV fold AUC: [0.646315, 0.707060, 0.767471, 0.867451]
  - TimeCV mean AUC: 0.747074
  - TimeCV OOF AUC: 0.871314 (23,972 rows)
  - RandomCV mean AUC: 0.903360
  - gap note: random CV is much higher than time-aware CV, so time-aware metric is the primary decision metric.
  - Kaggle LB (public/private): 0.903 / pending
  - comparison vs previous `submission.csv` (0.906): -0.003

## Logging rule
- Add one section per run.
- Always include: code revision, CV definition, feature diff, OOF, LB score, and short next action.

## v2 tuned-1 (light parameter search)
- date: 2026-02-12
- tuning script: `competitions/japan-ai-cup/scripts/tune_lgbm_v2.py`
- search space (8 candidates):
  - `num_leaves`: [47, 63, 95]
  - `min_child_samples`: [60, 80, 120]
  - `feature_fraction`: [0.7, 0.8, 0.9]
  - `lambda_l1/lambda_l2`: [(0.1,1.0), (0.1,2.0), (0.5,3.0)]
- best candidate by TimeCV mean AUC:
  - name: `leaf63_child120`
  - params: `num_leaves=63, min_child_samples=120, feature_fraction=0.8, lambda_l1=0.1, lambda_l2=1.0`
  - TimeCV mean AUC: 0.755463
  - TimeCV OOF AUC: 0.871365
  - best RandomCV mean AUC: 0.903706
- retrain output (best params):
  - `competitions/japan-ai-cup/predictions/submission_v2_tuned1.csv`
  - `competitions/japan-ai-cup/predictions/oof_v2_tuned1.csv`
  - `competitions/japan-ai-cup/predictions/feature_importance_v2_tuned1.csv`
- retrain metrics:
  - TimeCV mean AUC: 0.754307
  - TimeCV OOF AUC: 0.873210
  - RandomCV mean AUC: 0.903126
- next:
  - submit `submission_v2_tuned1.csv` and compare LB vs 0.903.

## v2 ablation (feature removal)
- date: 2026-02-12
- script: `competitions/japan-ai-cup/scripts/ablate_features_v2.py`
- fixed params: `num_leaves=63, min_child_samples=120, feature_fraction=0.8, lambda_l1=0.1, lambda_l2=1.0`
- compared settings:
  - full (49 features): TimeCV mean 0.754780 / OOF 0.874581
  - no_refund (45 features): TimeCV mean 0.754780 / OOF 0.874581
  - no_recency_window (35 features): TimeCV mean 0.754908 / OOF 0.875530
- outputs:
  - `competitions/japan-ai-cup/notes/ablation_results_v2.csv`
  - `competitions/japan-ai-cup/predictions/submission_v2_ablation_full.csv`
  - `competitions/japan-ai-cup/predictions/submission_v2_ablation_no_refund.csv`
  - `competitions/japan-ai-cup/predictions/submission_v2_ablation_no_recency_window.csv`
- submission:
  - submitted `submission_v2_ablation_no_recency_window.csv` (LB pending)
