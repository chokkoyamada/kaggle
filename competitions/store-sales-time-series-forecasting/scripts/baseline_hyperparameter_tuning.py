"""
Store Sales - ベースラインモデルのハイパーパラメータチューニング

特徴量はベースラインのまま、ハイパーパラメータを最適化する。
過学習を防ぎ、汎化性能を向上させる。

チューニング戦略:
1. 複数のパラメータセットを試す
2. 時系列CVで評価
3. 最良の設定を選択
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
print("Store Sales - ハイパーパラメータチューニング")
print("=" * 80)

# データの読み込み
print("\nデータを読み込み中...")
train_df = pd.read_csv(DATA_DIR / 'train.csv', parse_dates=['date'])
test_df = pd.read_csv(DATA_DIR / 'test.csv', parse_dates=['date'])

print(f"訓練データ: {train_df.shape}")
print(f"テストデータ: {test_df.shape}")

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

# 特徴量リスト（ベースラインと同じ）
feature_cols = [
    'store_nbr', 'family', 'onpromotion',
    'year', 'month', 'day_of_week', 'day_of_month', 'week_of_year',
    'sales_lag_7', 'sales_lag_14'
]

# familyをカテゴリコード化
family_mapping = {family: idx for idx, family in enumerate(train_with_features['family'].unique())}
train_with_features['family'] = train_with_features['family'].map(family_mapping)
test_with_features['family'] = test_with_features['family'].map(family_mapping)

# 欠損値を0で埋める
train_with_features[feature_cols] = train_with_features[feature_cols].fillna(0)
test_with_features[feature_cols] = test_with_features[feature_cols].fillna(0)

# 時系列交差検証の設定
validation_periods = [
    ('2017-06-16', '2017-06-30'),  # Fold 1
    ('2017-07-01', '2017-07-15'),  # Fold 2
    ('2017-07-16', '2017-07-31'),  # Fold 3
    ('2017-08-01', '2017-08-15'),  # Fold 4
]

# ハイパーパラメータの候補
print("\n" + "=" * 80)
print("ハイパーパラメータチューニング")
print("=" * 80)

param_sets = {
    'baseline': {
        'name': 'ベースライン（現在の設定）',
        'params': {
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
    },
    'conservative': {
        'name': '保守的（過学習防止重視）',
        'params': {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 20,  # ← 減らす
            'learning_rate': 0.03,  # ← 減らす
            'min_data_in_leaf': 50,  # ← 追加
            'feature_fraction': 0.7,  # ← 減らす
            'bagging_fraction': 0.7,  # ← 減らす
            'bagging_freq': 5,
            'lambda_l1': 0.1,  # ← L1正則化追加
            'lambda_l2': 0.1,  # ← L2正則化追加
            'verbose': -1,
            'seed': 42
        }
    },
    'aggressive': {
        'name': '積極的（表現力重視）',
        'params': {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 50,  # ← 増やす
            'learning_rate': 0.07,  # ← 増やす
            'min_data_in_leaf': 20,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.9,
            'bagging_freq': 3,
            'verbose': -1,
            'seed': 42
        }
    },
    'balanced': {
        'name': 'バランス型',
        'params': {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 25,
            'learning_rate': 0.04,
            'min_data_in_leaf': 30,
            'feature_fraction': 0.75,
            'bagging_fraction': 0.75,
            'bagging_freq': 5,
            'lambda_l1': 0.05,
            'lambda_l2': 0.05,
            'verbose': -1,
            'seed': 42
        }
    }
}

print(f"\n試すパラメータセット数: {len(param_sets)}")
for key, config in param_sets.items():
    print(f"  - {config['name']}")

# 各パラメータセットで評価
all_results = {}

for param_name, param_config in param_sets.items():
    print(f"\n{'='*80}")
    print(f"【{param_config['name']}】")
    print(f"{'='*80}")

    params = param_config['params']

    # 主要なパラメータを表示
    print(f"  num_leaves: {params['num_leaves']}")
    print(f"  learning_rate: {params['learning_rate']}")
    print(f"  min_data_in_leaf: {params.get('min_data_in_leaf', 'デフォルト')}")
    print(f"  lambda_l1: {params.get('lambda_l1', 0)}")
    print(f"  lambda_l2: {params.get('lambda_l2', 0)}")

    fold_scores = []
    fold_models = []

    for fold_idx, (val_start, val_end) in enumerate(validation_periods, 1):
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
                lgb.log_evaluation(period=0)
            ]
        )

        # 検証データで予測
        y_pred_valid = model.predict(X_valid_fold, num_iteration=model.best_iteration)
        y_pred_valid = np.maximum(y_pred_valid, 0)

        # RMSLE計算
        rmsle = np.sqrt(mean_squared_log_error(y_valid_fold, y_pred_valid))
        fold_scores.append(rmsle)
        fold_models.append(model)

    # 結果を保存
    mean_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)

    all_results[param_name] = {
        'name': param_config['name'],
        'mean': mean_score,
        'std': std_score,
        'last_fold': fold_scores[-1],
        'fold_scores': fold_scores,
        'models': fold_models
    }

    print(f"\n  結果:")
    for i, score in enumerate(fold_scores, 1):
        print(f"    Fold {i}: {score:.4f}")
    print(f"  平均RMSLE: {mean_score:.4f} (+/- {std_score:.4f})")
    print(f"  最終Fold: {fold_scores[-1]:.4f}")

# 全パラメータセットの比較
print("\n" + "=" * 80)
print("全パラメータセットの比較")
print("=" * 80)

# 結果を表形式で表示
print(f"\n{'設定名':<25} {'平均RMSLE':<12} {'標準偏差':<12} {'最終Fold':<12}")
print("-" * 80)

baseline_mean = all_results['baseline']['mean']

for param_name in ['baseline', 'conservative', 'aggressive', 'balanced']:
    result = all_results[param_name]
    marker = "✅" if result['mean'] < baseline_mean and param_name != 'baseline' else ""
    print(f"{result['name']:<25} {result['mean']:<12.4f} {result['std']:<12.4f} {result['last_fold']:<12.4f} {marker}")

# 最良のパラメータセットを選択
best_param_name = min(all_results.keys(), key=lambda k: all_results[k]['mean'])
best_result = all_results[best_param_name]

print(f"\n{'='*80}")
print(f"最良のパラメータセット: {best_result['name']}")
print(f"平均RMSLE: {best_result['mean']:.4f} (+/- {best_result['std']:.4f})")
print(f"最終Fold: {best_result['last_fold']:.4f}")
print(f"{'='*80}")

# ベースラインとの比較
baseline_result = all_results['baseline']
if best_param_name != 'baseline':
    improvement = baseline_result['mean'] - best_result['mean']
    improvement_pct = improvement / baseline_result['mean'] * 100
    print(f"\nベースラインからの改善: {improvement:.4f} ({improvement_pct:.2f}%)")

# 最良モデルでテストデータを予測
print("\n" + "=" * 80)
print("最良モデルでテストデータを予測")
print("=" * 80)

best_models = best_result['models']
best_params = param_sets[best_param_name]['params']

# 全訓練データで最終モデルを訓練
X_train_full = train_with_features[feature_cols]
y_train_full = train_with_features['sales']

lgb_train_full = lgb.Dataset(X_train_full, y_train_full)

final_model = lgb.train(
    best_params,
    lgb_train_full,
    num_boost_round=int(np.mean([m.best_iteration for m in best_models])),
    valid_sets=[lgb_train_full],
    valid_names=['train'],
    callbacks=[lgb.log_evaluation(period=200)]
)

# テストデータで予測
X_test = test_with_features[feature_cols]
y_pred_test = final_model.predict(X_test, num_iteration=final_model.best_iteration)
y_pred_test = np.maximum(y_pred_test, 0)

# 提出用ファイルを作成
submission_df = pd.DataFrame({
    'id': test_with_features['id'],
    'sales': y_pred_test
})

output_path = OUTPUT_DIR / 'submission_tuned.csv'
submission_df.to_csv(output_path, index=False)

print(f"\n予測結果を保存しました: {output_path}")
print(f"予測売上統計:")
print(f"  平均: {y_pred_test.mean():.2f}")
print(f"  中央値: {np.median(y_pred_test):.2f}")

# まとめ
print("\n" + "=" * 80)
print("まとめ")
print("=" * 80)
print(f"\n最良パラメータセット: {best_result['name']}")
print(f"時系列CV平均RMSLE: {best_result['mean']:.4f} (+/- {best_result['std']:.4f})")
print(f"最終Fold RMSLE: {best_result['last_fold']:.4f}")
print(f"\n予想Public Score: {best_result['last_fold'] * 1.5:.2f} 前後")
print(f"\n提出ファイル: {output_path}")

if best_result['mean'] < baseline_result['mean']:
    print(f"\n✅ ハイパーパラメータチューニングで改善しました！")
else:
    print(f"\n⚠️ ベースラインの設定が最適でした")
