"""
Japan AI Cup - ベースラインモデル
LightGBMを使用した顧客再訪予測
"""
import duckdb
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# パス設定
DATA_DIR = "competitions/japan-ai-cup/data"
OUTPUT_DIR = "competitions/japan-ai-cup/predictions"

print("=" * 60)
print("Japan AI Cup - ベースラインモデル")
print("=" * 60)

# =============================================================================
# 1. 特徴量エンジニアリング（DuckDBで高速処理）
# =============================================================================
print("\n📊 特徴量エンジニアリング...")

con = duckdb.connect()

# 参照日（予測対象期間の開始日）
REFERENCE_DATE = 20250203

# ユーザーごとの特徴量を集約
features_query = f"""
WITH user_features AS (
    SELECT
        user_id,

        -- 購買行動の統計
        COUNT(*) as purchase_count,
        SUM(total_price) as total_spent,
        AVG(total_price) as avg_spent,
        MAX(total_price) as max_spent,
        MIN(total_price) as min_spent,
        STDDEV(total_price) as std_spent,

        -- 商品数・点数
        SUM(amount) as total_items,
        AVG(amount) as avg_items,

        -- 来店パターン
        COUNT(DISTINCT date) as visit_days,
        MIN(date) as first_purchase_date,
        MAX(date) as last_purchase_date,
        {REFERENCE_DATE} - MAX(date) as recency,

        -- 商品カテゴリの多様性
        COUNT(DISTINCT item_category_cd_1) as unique_cat1,
        COUNT(DISTINCT item_category_cd_2) as unique_cat2,
        COUNT(DISTINCT item_category_cd_3) as unique_cat3,
        COUNT(DISTINCT jan_cd) as unique_products,

        -- 顧客属性（最新の値を取得）
        FIRST(age_category) as age_category,
        FIRST(sex) as sex,
        FIRST(user_stage) as user_stage,
        FIRST(membership_start_ym) as membership_start_ym,
        FIRST(user_flag_ec) as user_flag_ec,
        FIRST(user_flag_1) as user_flag_1,
        FIRST(user_flag_2) as user_flag_2,
        FIRST(user_flag_3) as user_flag_3,
        FIRST(user_flag_4) as user_flag_4,
        FIRST(user_flag_5) as user_flag_5,
        FIRST(user_flag_6) as user_flag_6

    FROM read_csv_auto('{DATA_DIR}/data.csv')
    GROUP BY user_id
)
SELECT
    f.*,
    -- 派生特徴量
    f.total_spent / NULLIF(f.visit_days, 0) as avg_spent_per_visit,
    f.purchase_count / NULLIF(f.visit_days, 0) as avg_purchases_per_visit,
    f.last_purchase_date - f.first_purchase_date as purchase_span,
    CASE
        WHEN f.last_purchase_date - f.first_purchase_date > 0
        THEN f.visit_days * 1.0 / (f.last_purchase_date - f.first_purchase_date)
        ELSE 0
    END as visit_frequency
FROM user_features f
"""

df_features = con.execute(features_query).fetchdf()
print(f"  特徴量データ: {len(df_features):,} ユーザー, {len(df_features.columns)} カラム")

# Train/Test ラベルの読み込み
df_train_labels = con.execute(f"""
    SELECT user_id, churn as target
    FROM read_csv_auto('{DATA_DIR}/train_flag.csv')
""").fetchdf()

df_test_users = con.execute(f"""
    SELECT user_id
    FROM read_csv_auto('{DATA_DIR}/sample_submission.csv')
""").fetchdf()

con.close()

# =============================================================================
# 2. Train/Test データの準備
# =============================================================================
print("\n🔄 Train/Test データの準備...")

# TrainとTestに分割
df_train = df_features.merge(df_train_labels, on='user_id', how='inner')
df_test = df_features.merge(df_test_users, on='user_id', how='inner')

print(f"  Train: {len(df_train):,} ユーザー")
print(f"  Test: {len(df_test):,} ユーザー")

# 特徴量カラムの定義
feature_cols = [col for col in df_features.columns if col != 'user_id']

# カテゴリカル特徴量
categorical_cols = ['age_category', 'sex', 'user_stage']

# カテゴリカル特徴量をcategory型に変換
for col in categorical_cols:
    df_train[col] = df_train[col].astype('category')
    df_test[col] = df_test[col].astype('category')

# 欠損値の確認
print(f"\n  欠損値のあるカラム:")
for col in feature_cols:
    null_count = df_train[col].isnull().sum()
    if null_count > 0:
        print(f"    {col}: {null_count:,} ({null_count/len(df_train):.1%})")

# =============================================================================
# 3. モデル学習（LightGBM + クロスバリデーション）
# =============================================================================
print("\n🚀 モデル学習...")

X_train = df_train[feature_cols]
y_train = df_train['target']
X_test = df_test[feature_cols]

# LightGBMパラメータ
params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': -1,
    'min_child_samples': 20,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'seed': 42,
}

# クロスバリデーション
n_splits = 5
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X_train))
test_preds = np.zeros(len(X_test))
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train), 1):
    print(f"\n  Fold {fold}/{n_splits}")

    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    train_data = lgb.Dataset(X_tr, label=y_tr, categorical_feature=categorical_cols)
    val_data = lgb.Dataset(X_val, label=y_val, categorical_feature=categorical_cols)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=0)
        ]
    )

    # バリデーション予測
    val_pred = model.predict(X_val)
    oof_preds[val_idx] = val_pred

    # テスト予測（平均を取る）
    test_preds += model.predict(X_test) / n_splits

    # スコア計算
    fold_auc = roc_auc_score(y_val, val_pred)
    cv_scores.append(fold_auc)
    print(f"    AUC: {fold_auc:.5f}")

# 全体のCV スコア
overall_auc = roc_auc_score(y_train, oof_preds)
print(f"\n{'=' * 60}")
print(f"📈 CV結果")
print(f"{'=' * 60}")
print(f"  各Foldの AUC: {[f'{s:.5f}' for s in cv_scores]}")
print(f"  平均 AUC: {np.mean(cv_scores):.5f} (±{np.std(cv_scores):.5f})")
print(f"  OOF AUC: {overall_auc:.5f}")

# =============================================================================
# 4. 特徴量重要度
# =============================================================================
print(f"\n{'=' * 60}")
print("📊 特徴量重要度 (Top 15)")
print(f"{'=' * 60}")

importance = model.feature_importance(importance_type='gain')
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': importance
}).sort_values('importance', ascending=False)

for i, row in feature_importance.head(15).iterrows():
    pct = row['importance'] / importance.sum() * 100
    print(f"  {row['feature']}: {pct:.2f}%")

# =============================================================================
# 5. 提出ファイルの作成
# =============================================================================
print(f"\n{'=' * 60}")
print("📝 提出ファイルの作成")
print(f"{'=' * 60}")

submission = pd.DataFrame({
    'user_id': df_test['user_id'],
    'pred': test_preds
})

output_path = f"{OUTPUT_DIR}/submission.csv"
submission.to_csv(output_path, index=False)

print(f"  保存先: {output_path}")
print(f"  行数: {len(submission):,}")
print(f"  予測値の範囲: {submission['pred'].min():.4f} 〜 {submission['pred'].max():.4f}")
print(f"  予測値の平均: {submission['pred'].mean():.4f}")

print("\n✅ ベースラインモデル完了")
