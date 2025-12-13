"""
Store Sales - ベースライン + 店舗Cluster情報

ベースラインモデルに店舗cluster情報を追加。
時系列交差検証で効果を測定する。

追加する理由:
- 構造的な情報で過学習リスクが低い
- 店舗間の類似性を捉える（17のclusterに分類）
- テストデータでも利用可能
- 時間によって変わらない安定した特徴
"""
import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
from sklearn.metrics import mean_squared_log_error
from datetime import datetime, timedelta

# パス設定
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data/raw'
OUTPUT_DIR = PROJECT_ROOT / 'predictions'

# 出力ディレクトリを作成
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 80)
print("Store Sales - ベースライン + 店舗Cluster情報")
print("=" * 80)

# データの読み込み
print("\nデータを読み込み中...")
train_df = pd.read_csv(DATA_DIR / 'train.csv', parse_dates=['date'])
test_df = pd.read_csv(DATA_DIR / 'test.csv', parse_dates=['date'])
stores_df = pd.read_csv(DATA_DIR / 'stores.csv')

print(f"訓練データ: {train_df.shape}")
print(f"テストデータ: {test_df.shape}")
print(f"店舗データ: {stores_df.shape}")

# 店舗情報の確認
print(f"\n店舗cluster情報:")
print(f"  ユニークな店舗数: {stores_df['store_nbr'].nunique()}")
print(f"  cluster数: {stores_df['cluster'].nunique()}")
print(f"  clusterの範囲: {stores_df['cluster'].min()} ～ {stores_df['cluster'].max()}")
print(f"  各clusterの店舗数:")
print(stores_df['cluster'].value_counts().sort_index().to_string())

# 基本的な日付特徴量を作成
def create_date_features(df):
    """日付から基本的な特徴量を生成"""
    df = df.copy()
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_month'] = df['date'].dt.day
    df['week_of_year'] = df['date'].dt.isocalendar().week
    return df

print("\n日付特徴量を生成中...")
train_df = create_date_features(train_df)
test_df = create_date_features(test_df)

# 店舗cluster情報を結合
print("\n店舗cluster情報を結合中...")
train_df = train_df.merge(stores_df[['store_nbr', 'cluster']], on='store_nbr', how='left')
test_df = test_df.merge(stores_df[['store_nbr', 'cluster']], on='store_nbr', how='left')

# 欠損値チェック
print(f"cluster欠損チェック:")
print(f"  訓練データ: {train_df['cluster'].isna().sum()} / {len(train_df)} ({train_df['cluster'].isna().sum()/len(train_df):.2%})")
print(f"  テストデータ: {test_df['cluster'].isna().sum()} / {len(test_df)} ({test_df['cluster'].isna().sum()/len(test_df):.2%})")

# ラグ特徴量の作成
print("\nラグ特徴量を生成中...")

# 訓練データとテストデータを結合してラグを計算
all_data = pd.concat([train_df, test_df], ignore_index=True)
all_data = all_data.sort_values(['store_nbr', 'family', 'date'])

# 7日前と14日前の売上
for lag in [7, 14]:
    all_data[f'sales_lag_{lag}'] = all_data.groupby(['store_nbr', 'family'])['sales'].shift(lag)

# 訓練とテストに再分割
train_with_features = all_data[all_data['date'] <= train_df['date'].max()].copy()
test_with_features = all_data[all_data['date'] > train_df['date'].max()].copy()

print(f"特徴量作成後の訓練データ: {train_with_features.shape}")
print(f"特徴量作成後のテストデータ: {test_with_features.shape}")

# 特徴量リスト（clusterを追加）
feature_cols = [
    'store_nbr', 'family', 'onpromotion',
    'year', 'month', 'day_of_week', 'day_of_month', 'week_of_year',
    'sales_lag_7', 'sales_lag_14',
    'cluster'  # ← 新規追加
]

print(f"\n特徴量数: {len(feature_cols)}")
print(f"追加特徴量: cluster")

# familyをカテゴリコード化
print("\nカテゴリカル変数をエンコード中...")
family_mapping = {family: idx for idx, family in enumerate(train_with_features['family'].unique())}
train_with_features['family'] = train_with_features['family'].map(family_mapping)
test_with_features['family'] = test_with_features['family'].map(family_mapping)

# 欠損値を0で埋める
train_with_features[feature_cols] = train_with_features[feature_cols].fillna(0)
test_with_features[feature_cols] = test_with_features[feature_cols].fillna(0)

# 時系列交差検証の設定
print("\n" + "=" * 80)
print("時系列交差検証")
print("=" * 80)

validation_periods = [
    ('2017-06-16', '2017-06-30'),  # Fold 1
    ('2017-07-01', '2017-07-15'),  # Fold 2
    ('2017-07-16', '2017-07-31'),  # Fold 3
    ('2017-08-01', '2017-08-15'),  # Fold 4
]

print(f"\n検証期間数: {len(validation_periods)}")

# LightGBMのハイパーパラメータ
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'seed': 42
}

# 各Foldで訓練・評価
fold_scores = []
fold_models = []

for fold_idx, (val_start, val_end) in enumerate(validation_periods, 1):
    print(f"\nFold {fold_idx}: 検証期間 {val_start} ～ {val_end}")

    # 訓練データと検証データに分割
    val_start_date = pd.to_datetime(val_start)
    val_end_date = pd.to_datetime(val_end)

    train_fold = train_with_features[train_with_features['date'] < val_start_date].copy()
    valid_fold = train_with_features[
        (train_with_features['date'] >= val_start_date) &
        (train_with_features['date'] <= val_end_date)
    ].copy()

    # 特徴量とターゲットに分割
    X_train_fold = train_fold[feature_cols]
    y_train_fold = train_fold['sales']
    X_valid_fold = valid_fold[feature_cols]
    y_valid_fold = valid_fold['sales']

    # LightGBMデータセット作成
    lgb_train = lgb.Dataset(X_train_fold, y_train_fold)
    lgb_valid = lgb.Dataset(X_valid_fold, y_valid_fold, reference=lgb_train)

    # モデル訓練
    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=1000,
        valid_sets=[lgb_train, lgb_valid],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=0)  # ログを抑制
        ]
    )

    # 検証データで予測
    y_pred_valid = model.predict(X_valid_fold, num_iteration=model.best_iteration)
    y_pred_valid = np.maximum(y_pred_valid, 0)

    # RMSLE計算
    rmsle = np.sqrt(mean_squared_log_error(y_valid_fold, y_pred_valid))
    fold_scores.append(rmsle)
    fold_models.append(model)

    print(f"  RMSLE: {rmsle:.4f} (best iteration: {model.best_iteration})")

# 全Foldの結果をまとめる
print("\n" + "=" * 80)
print("時系列交差検証の結果")
print("=" * 80)

for i, (period, score) in enumerate(zip(validation_periods, fold_scores), 1):
    print(f"Fold {i} ({period[0]} ～ {period[1]}): RMSLE = {score:.4f}")

mean_score = np.mean(fold_scores)
std_score = np.std(fold_scores)

print(f"\n{'='*80}")
print(f"【ベースライン + Cluster】")
print(f"平均RMSLE: {mean_score:.4f} (+/- {std_score:.4f})")
print(f"最終Fold RMSLE: {fold_scores[-1]:.4f}")
print(f"{'='*80}")

# ベースラインとの比較
baseline_mean = 0.9387
baseline_std = 0.2882
baseline_last = 0.7462

print(f"\n【ベースライン（参考）】")
print(f"平均RMSLE: {baseline_mean:.4f} (+/- {baseline_std:.4f})")
print(f"最終Fold RMSLE: {baseline_last:.4f}")

print(f"\n【改善度】")
print(f"平均スコア: {baseline_mean:.4f} → {mean_score:.4f} ({(baseline_mean - mean_score):.4f})")
print(f"標準偏差: {baseline_std:.4f} → {std_score:.4f} ({(baseline_std - std_score):.4f})")
print(f"最終Fold: {baseline_last:.4f} → {fold_scores[-1]:.4f} ({(baseline_last - fold_scores[-1]):.4f})")

if mean_score < baseline_mean:
    improvement_pct = (baseline_mean - mean_score) / baseline_mean * 100
    print(f"\n✅ 平均スコアが {improvement_pct:.2f}% 改善しました！")
else:
    decline_pct = (mean_score - baseline_mean) / baseline_mean * 100
    print(f"\n⚠️ 平均スコアが {decline_pct:.2f}% 悪化しました")

if std_score < baseline_std:
    stability_improvement = (baseline_std - std_score) / baseline_std * 100
    print(f"✅ スコアのばらつきが {stability_improvement:.2f}% 減少（より安定）")
else:
    stability_decline = (std_score - baseline_std) / baseline_std * 100
    print(f"⚠️ スコアのばらつきが {stability_decline:.2f}% 増加（不安定化）")

# 全訓練データで最終モデルを訓練
print("\n" + "=" * 80)
print("全訓練データで最終モデルを訓練")
print("=" * 80)

X_train_full = train_with_features[feature_cols]
y_train_full = train_with_features['sales']

lgb_train_full = lgb.Dataset(X_train_full, y_train_full)

final_model = lgb.train(
    params,
    lgb_train_full,
    num_boost_round=int(np.mean([m.best_iteration for m in fold_models])),
    valid_sets=[lgb_train_full],
    valid_names=['train'],
    callbacks=[lgb.log_evaluation(period=200)]
)

# 特徴量の重要度
print("\n特徴量の重要度:")
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': final_model.feature_importance(importance_type='gain')
}).sort_values('importance', ascending=False)
print(feature_importance.to_string(index=False))

# clusterの順位を確認
cluster_rank = feature_importance[feature_importance['feature'] == 'cluster'].index[0] + 1
print(f"\ncluster の重要度順位: {cluster_rank} / {len(feature_cols)}")

# テストデータで予測
print("\n" + "=" * 80)
print("テストデータで予測を生成")
print("=" * 80)

X_test = test_with_features[feature_cols]
y_pred_test = final_model.predict(X_test, num_iteration=final_model.best_iteration)
y_pred_test = np.maximum(y_pred_test, 0)

# 提出用ファイルを作成
submission_df = pd.DataFrame({
    'id': test_with_features['id'],
    'sales': y_pred_test
})

output_path = OUTPUT_DIR / 'submission_baseline_plus_cluster.csv'
submission_df.to_csv(output_path, index=False)

print(f"\n予測結果を保存しました: {output_path}")
print(f"予測売上統計:")
print(f"  平均: {y_pred_test.mean():.2f}")
print(f"  中央値: {np.median(y_pred_test):.2f}")

# まとめ
print("\n" + "=" * 80)
print("まとめ")
print("=" * 80)
print(f"\n追加特徴量: cluster")
print(f"時系列CV平均RMSLE: {mean_score:.4f} (+/- {std_score:.4f})")
print(f"最終Fold RMSLE: {fold_scores[-1]:.4f}")
print(f"\nベースラインからの改善:")
print(f"  平均スコア: {(baseline_mean - mean_score):.4f}")
print(f"  標準偏差: {(baseline_std - std_score):.4f}")
print(f"\n提出ファイル: {output_path}")

if mean_score < baseline_mean and fold_scores[-1] < baseline_last:
    print(f"\n✅ この特徴量は効果的です！Kaggleに提出しましょう")
    print(f"予想Public Score: {fold_scores[-1] * 1.5:.2f} 前後")
elif mean_score < baseline_mean:
    print(f"\n⚠️ 平均スコアは改善しましたが、最終Foldが悪化しています")
    print(f"慎重に判断してください")
else:
    print(f"\n⚠️ この特徴量は効果が薄いようです")
