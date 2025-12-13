# プロジェクト概要

このプロジェクトはデータサイエンス・データコンペティション（主にKaggle）の学習用です。

## ユーザープロファイル

- データサイエンス・データコンペティションの初学者
- Claude Codeと対話しながら効果的に知識と経験を積んでいきたい

## コミュニケーション方針

### 解説を含めた進め方

各タスクを進める際は、以下の要素を含めてください：

1. **これからやること**: 次のステップの概要を簡潔に説明
2. **なぜそれをするのか**: その作業の目的や意義
3. **データサイエンス的な背景知識**: 使用する手法やツールの基本的な説明
4. **実装**: 実際のコードや操作
5. **結果の解釈**: 出力結果の見方や、次のアクションにどう繋がるか

### 学習重視の進め方

- 単に作業を完了するだけでなく、各ステップで学びがあるように意識する
- データサイエンスのベストプラクティスを紹介する
- よくある落とし穴や注意点も共有する
- 初学者向けにわかりやすい言葉で説明する

## プロジェクト環境

- Python 3.12
- UV パッケージマネージャー
- Kaggle API (v1.8.2)
- DuckDB + dbt-core (特徴量エンジニアリング管理)

## DuckDB + dbt による特徴量エンジニアリング

### 採用理由

**再現性**: `dbt run`一つで全特徴量を再生成可能
**バージョン管理**: SQLファイルをGitで管理、変更履歴を追跡
**型安全性**: stagingで型を固定、downstreamで型を信頼可能
**テスト**: schema.ymlで自動的にデータ品質をチェック
**ドキュメント**: 各変換ステップが明示的で理解しやすい

### 型管理のベストプラクティス

**重要**: dbtでDuckDBを使う際、型管理を適切に行わないとビルドエラーが発生します。以下の原則を厳守してください：

#### ① 明示的CAST（最も重要）

stagingレイヤーで全カラムの型を明示的に固定します：

```sql
-- stg_train.sql
with source as (
    select * from {{ ref('train') }}
),
renamed_and_typed as (
    select
        cast("PassengerId" as integer) as passenger_id,
        cast("Survived" as integer) as survived,
        cast("Age" as double) as age,
        cast("Name" as varchar) as name,
        -- 全カラムに明示的なCASTを適用
        ...
    from source
)
select * from renamed_and_typed
```

#### ② schema.ymlによる型テスト

stagingレイヤーに型チェックを集中させます：

```yaml
# models/staging/schema.yml
models:
  - name: stg_train
    columns:
      - name: passenger_id
        data_type: integer
        tests:
          - not_null
          - unique
      - name: survived
        data_type: integer
        tests:
          - accepted_values:
              values: [0, 1]
```

#### ③ SELECT *の回避

intermediateレイヤー以降では、SELECT *を避けて全カラムを明示的に列挙します：

```sql
-- ❌ 悪い例
with base as (
    select * from {{ ref('previous_model') }}
)

-- ✅ 良い例
with base as (
    select
        passenger_id,
        survived,
        pclass,
        -- 全カラムを明示的に列挙
        ...
    from {{ ref('previous_model') }}
)
```

**理由**: DuckDBはview chainを通じてSELECT *を使うと、カラムの順序が混乱し、型推論がリセットされることがあります。

### dbtプロジェクト構造

```
competitions/[competition_name]/dbt_project/
├── dbt_project.yml          # dbtプロジェクト設定
├── profiles.yml             # DuckDB接続設定
├── models/
│   ├── staging/             # 生データ取り込み + 型固定
│   │   ├── stg_train.sql
│   │   ├── stg_test.sql
│   │   └── schema.yml       # 型テスト（ここにのみ集中）
│   ├── intermediate/        # 特徴量エンジニアリング
│   │   ├── int_all_passengers.sql
│   │   ├── int_features_*.sql
│   │   └── ...
│   └── marts/              # ML用最終データ
│       ├── train_processed.sql
│       └── test_processed.sql
├── seeds/                  # 元データ（CSV）
│   ├── train.csv
│   └── test.csv
└── scripts/               # Python補助スクリプト
    ├── export_data.py     # DuckDB → CSV
    └── train_model.py     # モデル訓練
```

### よく使うコマンド

```bash
# dbtプロジェクトディレクトリで実行

# データを読み込む
uv run dbt seed

# モデルを全て再構築
uv run dbt run --full-refresh

# 型テストを実行
uv run dbt test

# データをエクスポート
uv run python scripts/export_data.py

# モデルを訓練
uv run python scripts/train_model.py
```

### 注意点

1. **DuckDBファイルロック**: VS Codeなどでduckdbファイルを開いているとロックがかかります。`dbt run`の前に閉じてください。

2. **型テストの配置**: 型テストはstagingレイヤーにのみ配置します。intermediateやmartsには不要です（stagingで型が保証されているため）。

3. **欠損値の扱い**: schema.ymlで`not_null`テストを設定する際、欠損値が許容されるカラム（age, fare, cabinなど）には設定しないでください。

4. **dbt-expectations（将来的）**: より高度な型契約が必要な場合は、dbt-expectationsパッケージの導入を検討してください。

### Titanic Competition での実績

- **CV精度**: 83.05% (Gradient Boosting)
- **最重要特徴量**: title_encoded (60.88%)
- **型テスト**: 23個全てPASS
- **ビルド時間**: 約0.3秒（全10モデル）
