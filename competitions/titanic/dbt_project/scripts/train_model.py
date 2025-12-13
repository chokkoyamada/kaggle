"""
dbt生成データでモデルを訓練し、予測を生成するスクリプト
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb

# パス設定
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / '../data/processed'
OUTPUT_DIR = PROJECT_ROOT / '../predictions'

# 出力ディレクトリを作成
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("dbt生成データでモデルを訓練")
print("=" * 60)

# データの読み込み
print("\nデータを読み込み中...")
train_df = pd.read_csv(DATA_DIR / 'train_processed.csv')
test_df = pd.read_csv(DATA_DIR / 'test_processed.csv')

# 特徴量とターゲットに分割
feature_cols = ['pclass', 'sex_encoded', 'embarked_encoded', 'title_encoded',
                'family_size', 'is_alone',
                'age_band', 'fare_band', 'has_cabin',
                'age_is_child', 'ticket_group_size',
                'sex_pclass_interaction', 'child_pclass_interaction']

X_train = train_df[feature_cols]
y_train = train_df['survived']
X_test = test_df[feature_cols]
passenger_ids = test_df['passenger_id']

print(f"訓練データ: {X_train.shape}")
print(f"テストデータ: {X_test.shape}")
print(f"特徴量: {feature_cols}")

# 交差検証の設定
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# モデルの定義
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss'),
    'LightGBM': lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
}

# 各モデルを評価
print("\n" + "=" * 60)
print("モデル評価（5-fold Cross Validation）")
print("=" * 60)

results = {}
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    results[name] = {
        'mean': scores.mean(),
        'std': scores.std(),
        'model': model
    }
    print(f"{name:20s}: {scores.mean():.4f} (+/- {scores.std():.4f})")

# 最良のモデルを選択
best_model_name = max(results, key=lambda x: results[x]['mean'])
best_model = results[best_model_name]['model']

print("\n" + "=" * 60)
print(f"最良モデル: {best_model_name}")
print(f"CV精度: {results[best_model_name]['mean']:.4f} (+/- {results[best_model_name]['std']:.4f})")
print("=" * 60)

# 全訓練データで最良モデルを訓練
print("\n全訓練データで最良モデルを訓練中...")
best_model.fit(X_train, y_train)

# 特徴量の重要度を表示（サポートしているモデルの場合）
if hasattr(best_model, 'feature_importances_'):
    print("\n特徴量の重要度:")
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(feature_importance.to_string(index=False))

# テストデータで予測
print("\nテストデータで予測を生成中...")
predictions = best_model.predict(X_test)

# 提出用ファイルを作成
submission_df = pd.DataFrame({
    'PassengerId': passenger_ids,
    'Survived': predictions
})

# CSVとして保存
output_path = OUTPUT_DIR / 'submission_dbt.csv'
submission_df.to_csv(output_path, index=False)

print(f"\n予測結果を保存しました: {output_path}")
print(f"生存予測数: {predictions.sum()} / {len(predictions)} ({predictions.sum()/len(predictions):.2%})")

print("\n" + "=" * 60)
print("完了！")
print("=" * 60)
print(f"\nKaggleに提出するには:")
print(f"  cd {OUTPUT_DIR.parent}")
print(f"  uv run python kaggle_cli_wrapper.py competitions submit -c titanic -f {output_path} -m 'dbt generated features'")
