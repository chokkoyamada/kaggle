"""
Titanic データ探索
データの基本的な確認を行います
"""
import pandas as pd
import numpy as np

# データの読み込み
train_df = pd.read_csv('../data/train.csv')
test_df = pd.read_csv('../data/test.csv')

print("=" * 50)
print("訓練データの形状")
print("=" * 50)
print(f"行数: {train_df.shape[0]}, 列数: {train_df.shape[1]}")
print()

print("=" * 50)
print("テストデータの形状")
print("=" * 50)
print(f"行数: {test_df.shape[0]}, 列数: {test_df.shape[1]}")
print()

print("=" * 50)
print("訓練データの最初の5行")
print("=" * 50)
print(train_df.head())
print()

print("=" * 50)
print("列の情報")
print("=" * 50)
print(train_df.info())
print()

print("=" * 50)
print("基本統計量")
print("=" * 50)
print(train_df.describe())
print()

print("=" * 50)
print("欠損値の確認")
print("=" * 50)
missing = train_df.isnull().sum()
missing_percent = 100 * train_df.isnull().sum() / len(train_df)
missing_table = pd.DataFrame({
    '欠損数': missing,
    '欠損率(%)': missing_percent
})
print(missing_table[missing_table['欠損数'] > 0].sort_values('欠損数', ascending=False))
print()

print("=" * 50)
print("生存率")
print("=" * 50)
survival_rate = train_df['Survived'].value_counts()
survival_percent = 100 * train_df['Survived'].value_counts(normalize=True)
survival_table = pd.DataFrame({
    '人数': survival_rate,
    '割合(%)': survival_percent
})
print(survival_table)
print(f"\n全体の生存率: {train_df['Survived'].mean():.2%}")
