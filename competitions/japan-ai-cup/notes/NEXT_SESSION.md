# Japan AI Cup Next Session (2026-02-13)

## Start Here
1. 提出上限がリセットされたことを Kaggle 画面で確認する。
2. 次のファイルを提出する:
   - `competitions/japan-ai-cup/predictions/submission_stepwise_det_profile_plus_refund.csv`
   - 提出メッセージ: `stepwise: det_profile + drop_null + refund (no recency)`
3. LBを `submission_stepwise_det_profile_plus_refund_recency.csv` (0.906) と比較する。

## Decision Rule
- `plus_refund` が **0.907 以上**: これを新基準にして周辺を深掘り。
- `plus_refund` が **0.906**: recencyの有無で同等。次は seed averaging を試す。
- `plus_refund` が **0.905 以下**: recencyあり版（0.906）を維持して別方向を試す。

## Candidate Next Actions
- A. seed averaging (baseline/stepwise共通)
  - `seed in [42, 2024, 7, 77, 777]` で予測平均
- B. `det_profile_drop_null` 単独提出（切り分け）
- C. baselineパラメータの微調整
  - `num_leaves`: 24/31/40
  - `min_child_samples`: 20/40/80

## Current Best Public LB
- `submission.csv` (baseline): 0.906
- `submission_stepwise_det_profile_plus_refund_recency.csv`: 0.906

## Important Files
- Log: `competitions/japan-ai-cup/notes/experiments.md`
- Stepwise results: `competitions/japan-ai-cup/notes/stepwise_migration_results.csv`
- Submit candidates:
  - `competitions/japan-ai-cup/predictions/submission_stepwise_det_profile_plus_refund.csv`
  - `competitions/japan-ai-cup/predictions/submission_stepwise_det_profile_plus_refund_recency.csv`
