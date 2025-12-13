"""
Titanic 機械学習モデルの構築と訓練
複数のアルゴリズムを比較して最良のモデルを選択します
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# データの読み込み
train_df = pd.read_csv('../data/train_processed.csv')

# 特徴量とターゲットに分割
X = train_df.drop('Survived', axis=1)
y = train_df['Survived']

print("=" * 70)
print("機械学習モデルの訓練")
print("=" * 70)
print(f"訓練データサイズ: {X.shape}")
print(f"特徴量: {list(X.columns)}")
print(f"生存率: {y.mean():.2%}")
print()

# クロスバリデーションの設定
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("=" * 70)
print("複数のモデルを評価中...")
print("=" * 70)
print()

# モデルの定義
models = {
    'ロジスティック回帰': LogisticRegression(max_iter=1000, random_state=42),
    'ランダムフォレスト': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='rbf', random_state=42),
    'XGBoost': XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
}

# 各モデルの評価
results = {}
for name, model in models.items():
    print(f"【{name}】を評価中...")

    # クロスバリデーションで評価
    scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')

    results[name] = {
        '平均精度': scores.mean(),
        '標準偏差': scores.std(),
        'スコア': scores
    }

    print(f"  精度: {scores.mean():.4f} (+/- {scores.std():.4f})")
    print(f"  各Foldのスコア: {[f'{s:.4f}' for s in scores]}")
    print()

# 結果をDataFrameで表示
print("=" * 70)
print("モデル比較結果")
print("=" * 70)
results_df = pd.DataFrame({
    'モデル': list(results.keys()),
    '平均精度': [results[m]['平均精度'] for m in results.keys()],
    '標準偏差': [results[m]['標準偏差'] for m in results.keys()]
})
results_df = results_df.sort_values('平均精度', ascending=False)
print(results_df.to_string(index=False))
print()

# 最良のモデルを選択
best_model_name = results_df.iloc[0]['モデル']
print("=" * 70)
print(f"最良のモデル: {best_model_name}")
print(f"精度: {results_df.iloc[0]['平均精度']:.4f}")
print("=" * 70)
print()

# 最良のモデルを全データで訓練
print("=" * 70)
print("最良のモデルを全訓練データで訓練中...")
print("=" * 70)

best_model = models[best_model_name]
best_model.fit(X, y)

# 特徴量の重要度を表示（可能な場合）
if hasattr(best_model, 'feature_importances_'):
    print("\n特徴量の重要度:")
    feature_importance = pd.DataFrame({
        '特徴量': X.columns,
        '重要度': best_model.feature_importances_
    }).sort_values('重要度', ascending=False)
    print(feature_importance.to_string(index=False))
elif hasattr(best_model, 'coef_'):
    print("\n特徴量の係数:")
    feature_coef = pd.DataFrame({
        '特徴量': X.columns,
        '係数': best_model.coef_[0]
    }).sort_values('係数', ascending=False, key=abs)
    print(feature_coef.to_string(index=False))
print()

# モデルを保存（pickle）
import pickle
with open('../data/best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

print("=" * 70)
print("モデルを保存しました: ../data/best_model.pkl")
print("=" * 70)
