"""
Titanic データ可視化
各特徴量と生存率の関係を可視化します
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 日本語フォント設定（macOS）
plt.rcParams['font.family'] = ['Hiragino Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# データの読み込み
train_df = pd.read_csv('../data/train.csv')

# 図のスタイル設定
sns.set_palette("husl")
plt.style.use('seaborn-v0_8-darkgrid')

# 1. 性別別の生存率
print("=" * 50)
print("性別別の生存率")
print("=" * 50)
sex_survival = train_df.groupby('Sex')['Survived'].agg(['sum', 'count', 'mean'])
sex_survival.columns = ['生存者数', '総数', '生存率']
print(sex_survival)
print()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.countplot(data=train_df, x='Sex', hue='Survived', ax=axes[0])
axes[0].set_title('性別ごとの生存者数')
axes[0].set_xlabel('性別')
axes[0].set_ylabel('人数')
axes[0].legend(title='生存', labels=['死亡', '生存'])

sns.barplot(data=train_df, x='Sex', y='Survived', ax=axes[1])
axes[1].set_title('性別別の生存率')
axes[1].set_xlabel('性別')
axes[1].set_ylabel('生存率')
plt.tight_layout()
plt.savefig('../data/01_sex_survival.png', dpi=150, bbox_inches='tight')
print("図を保存: 01_sex_survival.png\n")

# 2. チケットクラス別の生存率
print("=" * 50)
print("チケットクラス別の生存率")
print("=" * 50)
pclass_survival = train_df.groupby('Pclass')['Survived'].agg(['sum', 'count', 'mean'])
pclass_survival.columns = ['生存者数', '総数', '生存率']
print(pclass_survival)
print()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.countplot(data=train_df, x='Pclass', hue='Survived', ax=axes[0])
axes[0].set_title('チケットクラスごとの生存者数')
axes[0].set_xlabel('チケットクラス (1=上級, 2=中級, 3=下級)')
axes[0].set_ylabel('人数')
axes[0].legend(title='生存', labels=['死亡', '生存'])

sns.barplot(data=train_df, x='Pclass', y='Survived', ax=axes[1])
axes[1].set_title('チケットクラス別の生存率')
axes[1].set_xlabel('チケットクラス')
axes[1].set_ylabel('生存率')
plt.tight_layout()
plt.savefig('../data/02_pclass_survival.png', dpi=150, bbox_inches='tight')
print("図を保存: 02_pclass_survival.png\n")

# 3. 性別とチケットクラスの組み合わせ
print("=" * 50)
print("性別×チケットクラス別の生存率")
print("=" * 50)
sex_pclass_survival = train_df.groupby(['Sex', 'Pclass'])['Survived'].agg(['sum', 'count', 'mean'])
sex_pclass_survival.columns = ['生存者数', '総数', '生存率']
print(sex_pclass_survival)
print()

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=train_df, x='Pclass', y='Survived', hue='Sex', ax=ax)
ax.set_title('性別×チケットクラス別の生存率')
ax.set_xlabel('チケットクラス')
ax.set_ylabel('生存率')
ax.legend(title='性別')
plt.tight_layout()
plt.savefig('../data/03_sex_pclass_survival.png', dpi=150, bbox_inches='tight')
print("図を保存: 03_sex_pclass_survival.png\n")

# 4. 年齢分布と生存率
print("=" * 50)
print("年齢と生存率")
print("=" * 50)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 年齢分布のヒストグラム
train_df[train_df['Survived']==0]['Age'].hist(bins=30, alpha=0.5, label='死亡', ax=axes[0])
train_df[train_df['Survived']==1]['Age'].hist(bins=30, alpha=0.5, label='生存', ax=axes[0])
axes[0].set_title('年齢分布（生存/死亡別）')
axes[0].set_xlabel('年齢')
axes[0].set_ylabel('人数')
axes[0].legend()

# 年齢層別の生存率
train_df['AgeGroup'] = pd.cut(train_df['Age'], bins=[0, 12, 18, 35, 60, 100],
                               labels=['子供(0-12)', '青少年(13-18)', '成人(19-35)', '中年(36-60)', '高齢(61+)'])
age_group_survival = train_df.groupby('AgeGroup')['Survived'].mean()
age_group_survival.plot(kind='bar', ax=axes[1])
axes[1].set_title('年齢層別の生存率')
axes[1].set_xlabel('年齢層')
axes[1].set_ylabel('生存率')
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45)

plt.tight_layout()
plt.savefig('../data/04_age_survival.png', dpi=150, bbox_inches='tight')
print("図を保存: 04_age_survival.png\n")

# 5. 家族サイズと生存率
print("=" * 50)
print("家族サイズと生存率")
print("=" * 50)
train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch'] + 1
family_survival = train_df.groupby('FamilySize')['Survived'].agg(['sum', 'count', 'mean'])
family_survival.columns = ['生存者数', '総数', '生存率']
print(family_survival)
print()

fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(data=train_df, x='FamilySize', y='Survived', ax=ax)
ax.set_title('家族サイズ別の生存率')
ax.set_xlabel('家族サイズ（本人含む）')
ax.set_ylabel('生存率')
plt.tight_layout()
plt.savefig('../data/05_family_survival.png', dpi=150, bbox_inches='tight')
print("図を保存: 05_family_survival.png\n")

# 6. 運賃と生存率
print("=" * 50)
print("運賃と生存率")
print("=" * 50)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 運賃分布（生存/死亡別）
train_df[train_df['Survived']==0]['Fare'].hist(bins=50, alpha=0.5, label='死亡', ax=axes[0])
train_df[train_df['Survived']==1]['Fare'].hist(bins=50, alpha=0.5, label='生存', ax=axes[0])
axes[0].set_title('運賃分布（生存/死亡別）')
axes[0].set_xlabel('運賃')
axes[0].set_ylabel('人数')
axes[0].set_xlim(0, 300)
axes[0].legend()

# 運賃帯別の生存率
train_df['FareGroup'] = pd.qcut(train_df['Fare'], q=4, labels=['低', '中低', '中高', '高'])
fare_group_survival = train_df.groupby('FareGroup')['Survived'].mean()
fare_group_survival.plot(kind='bar', ax=axes[1])
axes[1].set_title('運賃帯別の生存率')
axes[1].set_xlabel('運賃帯')
axes[1].set_ylabel('生存率')
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)

plt.tight_layout()
plt.savefig('../data/06_fare_survival.png', dpi=150, bbox_inches='tight')
print("図を保存: 06_fare_survival.png\n")

# 7. 乗船港別の生存率
print("=" * 50)
print("乗船港別の生存率")
print("=" * 50)
embarked_survival = train_df.groupby('Embarked')['Survived'].agg(['sum', 'count', 'mean'])
embarked_survival.columns = ['生存者数', '総数', '生存率']
print(embarked_survival)
print()

fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(data=train_df, x='Embarked', y='Survived', ax=ax)
ax.set_title('乗船港別の生存率')
ax.set_xlabel('乗船港 (C=Cherbourg, Q=Queenstown, S=Southampton)')
ax.set_ylabel('生存率')
plt.tight_layout()
plt.savefig('../data/07_embarked_survival.png', dpi=150, bbox_inches='tight')
print("図を保存: 07_embarked_survival.png\n")

print("=" * 50)
print("すべての可視化が完了しました")
print("=" * 50)
