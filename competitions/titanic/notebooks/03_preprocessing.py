"""
Titanic データ前処理と特徴量エンジニアリング
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# データの読み込み
train_df = pd.read_csv('../data/train.csv')
test_df = pd.read_csv('../data/test.csv')

# 後で結合するため、元のデータを保持
train_size = len(train_df)
# PassengerIdとSurvivedを保存
train_survived = train_df['Survived'].copy()
test_passenger_id = test_df['PassengerId'].copy()

# 訓練データとテストデータを結合して同じ前処理を適用
all_data = pd.concat([train_df.drop('Survived', axis=1), test_df], axis=0).reset_index(drop=True)

print("=" * 70)
print("データ前処理と特徴量エンジニアリング開始")
print("=" * 70)
print(f"結合後のデータサイズ: {all_data.shape}")
print()

# 1. 名前から称号(Title)を抽出
print("=" * 70)
print("1. 名前から称号(Title)を抽出")
print("=" * 70)
all_data['Title'] = all_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
print("抽出された称号:")
print(all_data['Title'].value_counts())
print()

# 称号を統合（稀な称号をまとめる）
title_mapping = {
    'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
    'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
    'Mlle': 'Miss', 'Countess': 'Rare', 'Ms': 'Miss', 'Lady': 'Rare',
    'Jonkheer': 'Rare', 'Don': 'Rare', 'Dona': 'Rare', 'Mme': 'Mrs',
    'Capt': 'Rare', 'Sir': 'Rare'
}
all_data['Title'] = all_data['Title'].map(title_mapping)
print("統合後の称号:")
print(all_data['Title'].value_counts())
print()

# 2. 家族サイズの特徴量作成
print("=" * 70)
print("2. 家族サイズの特徴量作成")
print("=" * 70)
all_data['FamilySize'] = all_data['SibSp'] + all_data['Parch'] + 1
all_data['IsAlone'] = (all_data['FamilySize'] == 1).astype(int)
print(f"家族サイズの範囲: {all_data['FamilySize'].min()} - {all_data['FamilySize'].max()}")
print(f"単独者の数: {all_data['IsAlone'].sum()} / {len(all_data)}")
print()

# 3. 年齢の欠損値を中央値で補完（称号ごと）
print("=" * 70)
print("3. 年齢の欠損値補完")
print("=" * 70)
print(f"補完前の欠損値: {all_data['Age'].isnull().sum()}")

# 称号ごとの年齢中央値で補完
for title in all_data['Title'].unique():
    if pd.notna(title):
        age_median = all_data[all_data['Title'] == title]['Age'].median()
        all_data.loc[(all_data['Age'].isnull()) & (all_data['Title'] == title), 'Age'] = age_median

print(f"補完後の欠損値: {all_data['Age'].isnull().sum()}")
print()

# 年齢を区間に分割
all_data['AgeBand'] = pd.cut(all_data['Age'], bins=[0, 12, 18, 35, 60, 100], labels=[0, 1, 2, 3, 4])

# 4. 運賃の欠損値を中央値で補完
print("=" * 70)
print("4. 運賃の欠損値補完")
print("=" * 70)
print(f"補完前の欠損値: {all_data['Fare'].isnull().sum()}")
all_data['Fare'].fillna(all_data['Fare'].median(), inplace=True)
print(f"補完後の欠損値: {all_data['Fare'].isnull().sum()}")
print()

# 運賃を区間に分割
all_data['FareBand'] = pd.qcut(all_data['Fare'], q=4, labels=[0, 1, 2, 3], duplicates='drop')

# 5. Embarkedの欠損値を最頻値で補完
print("=" * 70)
print("5. Embarkedの欠損値補完")
print("=" * 70)
print(f"補完前の欠損値: {all_data['Embarked'].isnull().sum()}")
all_data['Embarked'].fillna(all_data['Embarked'].mode()[0], inplace=True)
print(f"補完後の欠損値: {all_data['Embarked'].isnull().sum()}")
print()

# 6. Cabinの処理（Cabin番号の有無）
print("=" * 70)
print("6. Cabin情報の処理")
print("=" * 70)
all_data['HasCabin'] = all_data['Cabin'].notna().astype(int)
print(f"客室情報がある乗客: {all_data['HasCabin'].sum()} / {len(all_data)}")
print()

# 7. カテゴリカル変数のエンコーディング
print("=" * 70)
print("7. カテゴリカル変数のエンコーディング")
print("=" * 70)

# 性別をエンコード
all_data['Sex'] = all_data['Sex'].map({'female': 1, 'male': 0})
print("性別エンコーディング: female=1, male=0")

# Embarkedをエンコード
all_data['Embarked'] = all_data['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
print("乗船港エンコーディング: C=0, Q=1, S=2")

# Titleをエンコード
title_map = {'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Rare': 4}
all_data['Title'] = all_data['Title'].map(title_map)
print("称号エンコーディング: Mr=0, Miss=1, Mrs=2, Master=3, Rare=4")
print()

# 8. 不要な列を削除
print("=" * 70)
print("8. 不要な列を削除")
print("=" * 70)
columns_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'Age', 'Fare', 'SibSp', 'Parch']
print(f"削除する列: {columns_to_drop}")
all_data.drop(columns_to_drop, axis=1, inplace=True)
print()

# 9. 訓練データとテストデータに分割
print("=" * 70)
print("9. データの分割")
print("=" * 70)
train_processed = all_data[:train_size].copy()
test_processed = all_data[train_size:].copy()

# Survivedを追加
train_processed['Survived'] = train_survived.values

print(f"処理後の訓練データ: {train_processed.shape}")
print(f"処理後のテストデータ: {test_processed.shape}")
print()

print("最終的な特徴量:")
print(train_processed.columns.tolist())
print()

# データを保存
train_processed.to_csv('../data/train_processed.csv', index=False)
test_processed['PassengerId'] = test_passenger_id.values
test_processed.to_csv('../data/test_processed.csv', index=False)

print("=" * 70)
print("前処理完了: データを保存しました")
print("  - ../data/train_processed.csv")
print("  - ../data/test_processed.csv")
print("=" * 70)
print()

# 前処理後のデータをプレビュー
print("=" * 70)
print("前処理後の訓練データ（最初の5行）")
print("=" * 70)
print(train_processed.head())
print()

print("=" * 70)
print("欠損値の最終確認")
print("=" * 70)
print("訓練データ:")
print(train_processed.isnull().sum())
print("\nテストデータ:")
print(test_processed.isnull().sum())
