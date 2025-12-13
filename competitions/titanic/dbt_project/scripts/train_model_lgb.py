"""
dbt生成データでLightGBMモデルを訓練し、予測を生成するスクリプト
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import cross_val_score, StratifiedKFold
import lightgbm as lgb

# パス設定
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / '../data/processed'
OUTPUT_DIR = PROJECT_ROOT / '../predictions'

# 出力ディレクトリを作成
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("dbt生成データでLightGBMモデルを訓練")
print("=" * 60)

# データの読み込み
print("\nデータを読み込み中...")
train_df = pd.read_csv(DATA_DIR / 'train_processed.csv')
test_df = pd.read_csv(DATA_DIR / 'test_processed.csv')

# 特徴量とターゲットに分割
feature_cols = ['pclass', 'sex_encoded', 'embarked_encoded', 'title_encoded',
                'family_size', 'is_alone',
                'age_band', 'fare_band', 'has_cabin',
                'age_is_child', 'ticket_group_size']

X_train = train_df[feature_cols]
y_train = train_df['survived']
X_test = test_df[feature_cols]
passenger_ids = test_df['passenger_id']

print(f"訓練データ: {X_train.shape}")
print(f"テストデータ: {X_test.shape}")
print(f"特徴量: {feature_cols}")

# 交差検証の設定
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# LightGBMモデル
model = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)

# モデルを評価
print("\n" + "=" * 60)
print("モデル評価（5-fold Cross Validation）")
print("=" * 60)

scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
print(f"LightGBM: {scores.mean():.4f} (+/- {scores.std():.4f})")

# 全訓練データでモデルを訓練
print("\n全訓練データでモデルを訓練中...")
model.fit(X_train, y_train)

# 特徴量の重要度を表示
print("\n特徴量の重要度:")
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print(feature_importance.to_string(index=False))

# テストデータで予測
print("\nテストデータで予測を生成中...")
predictions = model.predict(X_test)

# 提出用ファイルを作成
submission_df = pd.DataFrame({
    'PassengerId': passenger_ids,
    'Survived': predictions
})

# CSVとして保存
output_path = OUTPUT_DIR / 'submission_lgb.csv'
submission_df.to_csv(output_path, index=False)

print(f"\n予測結果を保存しました: {output_path}")
print(f"生存予測数: {predictions.sum()} / {len(predictions)} ({predictions.sum()/len(predictions):.2%})")

print("\n" + "=" * 60)
print("完了！")
print("=" * 60)
