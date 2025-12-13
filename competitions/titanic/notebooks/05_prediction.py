"""
Titanic テストデータに対する予測と提出ファイルの作成
"""
import pandas as pd
import pickle

print("=" * 70)
print("テストデータに対する予測")
print("=" * 70)
print()

# テストデータの読み込み
test_df = pd.read_csv('../data/test_processed.csv')
passenger_ids = test_df['PassengerId']
X_test = test_df.drop('PassengerId', axis=1)

print(f"テストデータサイズ: {X_test.shape}")
print(f"特徴量: {list(X_test.columns)}")
print()

# モデルの読み込み
print("=" * 70)
print("訓練済みモデルを読み込み中...")
print("=" * 70)

with open('../data/best_model.pkl', 'rb') as f:
    model = pickle.load(f)

print(f"モデル: {type(model).__name__}")
print()

# 予測の実行
print("=" * 70)
print("予測を実行中...")
print("=" * 70)

predictions = model.predict(X_test)

print(f"予測完了: {len(predictions)}件")
print(f"生存予測: {predictions.sum()}人 ({predictions.sum()/len(predictions):.2%})")
print(f"死亡予測: {len(predictions) - predictions.sum()}人 ({(len(predictions) - predictions.sum())/len(predictions):.2%})")
print()

# 提出ファイルの作成
print("=" * 70)
print("提出ファイルを作成中...")
print("=" * 70)

submission = pd.DataFrame({
    'PassengerId': passenger_ids,
    'Survived': predictions
})

# ファイルを保存
submission_path = '../submissions/submission.csv'
submission.to_csv(submission_path, index=False)

print(f"提出ファイルを保存しました: {submission_path}")
print()

print("提出ファイルのプレビュー:")
print(submission.head(10))
print()

print("=" * 70)
print("予測完了！")
print("=" * 70)
print()
print("次のステップ:")
print("1. 提出ファイルの確認")
print("2. Kaggleにログイン")
print("3. Titanic コンペティションページで提出")
print()
print("提出コマンド:")
print(f"uv run kaggle-cli competitions submit -c titanic -f {submission_path} -m 'First submission with XGBoost'")
