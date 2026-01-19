"""
Japan AI Cup - 探索的データ分析（EDA）
"""
import duckdb

# データパス
DATA_DIR = "competitions/japan-ai-cup/data"

# DuckDB接続
con = duckdb.connect()

print("=" * 60)
print("1. データファイルの基本情報")
print("=" * 60)

# data.csvの行数とカラム
print("\n【data.csv】")
result = con.execute(f"""
    SELECT COUNT(*) as row_count
    FROM read_csv_auto('{DATA_DIR}/data.csv')
""").fetchone()
print(f"行数: {result[0]:,}")

# カラム情報
print("\nカラム情報:")
result = con.execute(f"""
    DESCRIBE SELECT * FROM read_csv_auto('{DATA_DIR}/data.csv')
""").fetchall()
for col in result:
    print(f"  {col[0]}: {col[1]}")

# train_flag.csvの情報
print("\n【train_flag.csv】")
result = con.execute(f"""
    SELECT
        COUNT(*) as total,
        SUM(churn) as positive,
        AVG(churn) as positive_rate
    FROM read_csv_auto('{DATA_DIR}/train_flag.csv')
""").fetchone()
print(f"訓練ユーザー数: {result[0]:,}")
print(f"正例（再訪する）: {result[1]:,} ({result[2]:.2%})")
print(f"負例（再訪しない）: {result[0] - result[1]:,} ({1 - result[2]:.2%})")

# sample_submission.csvの情報
print("\n【sample_submission.csv】")
result = con.execute(f"""
    SELECT COUNT(*) as row_count
    FROM read_csv_auto('{DATA_DIR}/sample_submission.csv')
""").fetchone()
print(f"テストユーザー数: {result[0]:,}")

print("\n" + "=" * 60)
print("2. ユーザー数の確認")
print("=" * 60)

result = con.execute(f"""
    SELECT COUNT(DISTINCT user_id) as unique_users
    FROM read_csv_auto('{DATA_DIR}/data.csv')
""").fetchone()
print(f"data.csvのユニークユーザー数: {result[0]:,}")

print("\n" + "=" * 60)
print("3. 期間の確認")
print("=" * 60)

result = con.execute(f"""
    SELECT
        MIN(date) as min_date,
        MAX(date) as max_date,
        COUNT(DISTINCT date) as unique_dates
    FROM read_csv_auto('{DATA_DIR}/data.csv')
""").fetchone()
print(f"期間: {result[0]} 〜 {result[1]}")
print(f"ユニーク日数: {result[2]}")

print("\n" + "=" * 60)
print("4. 商品カテゴリの確認")
print("=" * 60)

result = con.execute(f"""
    SELECT
        COUNT(DISTINCT item_category_cd_1) as cat1,
        COUNT(DISTINCT item_category_cd_2) as cat2,
        COUNT(DISTINCT item_category_cd_3) as cat3,
        COUNT(DISTINCT item_category_name) as cat_name
    FROM read_csv_auto('{DATA_DIR}/data.csv')
""").fetchone()
print(f"カテゴリ1のユニーク数: {result[0]}")
print(f"カテゴリ2のユニーク数: {result[1]}")
print(f"カテゴリ3のユニーク数: {result[2]}")
print(f"カテゴリ名のユニーク数: {result[3]}")

print("\n" + "=" * 60)
print("5. 顧客属性の確認")
print("=" * 60)

# 年齢カテゴリ
print("\n【年齢カテゴリ】")
result = con.execute(f"""
    SELECT
        age_category,
        COUNT(DISTINCT user_id) as user_count
    FROM read_csv_auto('{DATA_DIR}/data.csv')
    GROUP BY age_category
    ORDER BY age_category
""").fetchall()
for row in result:
    print(f"  {row[0]}: {row[1]:,} users")

# 性別
print("\n【性別】")
result = con.execute(f"""
    SELECT
        sex,
        COUNT(DISTINCT user_id) as user_count
    FROM read_csv_auto('{DATA_DIR}/data.csv')
    GROUP BY sex
    ORDER BY user_count DESC
""").fetchall()
for row in result:
    print(f"  {row[0]}: {row[1]:,} users")

# 会員ステータス
print("\n【会員ステータス】")
result = con.execute(f"""
    SELECT
        user_stage,
        COUNT(DISTINCT user_id) as user_count
    FROM read_csv_auto('{DATA_DIR}/data.csv')
    GROUP BY user_stage
    ORDER BY user_count DESC
""").fetchall()
for row in result:
    print(f"  {row[0]}: {row[1]:,} users")

print("\n" + "=" * 60)
print("6. 購買金額の統計")
print("=" * 60)

result = con.execute(f"""
    SELECT
        MIN(total_price) as min_price,
        AVG(total_price) as avg_price,
        MEDIAN(total_price) as median_price,
        MAX(total_price) as max_price,
        SUM(total_price) as total_sales
    FROM read_csv_auto('{DATA_DIR}/data.csv')
""").fetchone()
print(f"最小購買金額: ¥{result[0]:,.0f}")
print(f"平均購買金額: ¥{result[1]:,.0f}")
print(f"中央値: ¥{result[2]:,.0f}")
print(f"最大購買金額: ¥{result[3]:,.0f}")
print(f"総売上: ¥{result[4]:,.0f}")

print("\n" + "=" * 60)
print("7. ユーザーごとの購買統計")
print("=" * 60)

result = con.execute(f"""
    WITH user_stats AS (
        SELECT
            user_id,
            COUNT(*) as purchase_count,
            SUM(total_price) as total_spent,
            COUNT(DISTINCT date) as visit_days
        FROM read_csv_auto('{DATA_DIR}/data.csv')
        GROUP BY user_id
    )
    SELECT
        AVG(purchase_count) as avg_purchases,
        MEDIAN(purchase_count) as median_purchases,
        AVG(total_spent) as avg_spent,
        MEDIAN(total_spent) as median_spent,
        AVG(visit_days) as avg_visit_days,
        MEDIAN(visit_days) as median_visit_days
    FROM user_stats
""").fetchone()
print(f"平均購入回数/ユーザー: {result[0]:.1f} (中央値: {result[1]:.0f})")
print(f"平均購入金額/ユーザー: ¥{result[2]:,.0f} (中央値: ¥{result[3]:,.0f})")
print(f"平均来店日数/ユーザー: {result[4]:.1f} (中央値: {result[5]:.0f})")

print("\n" + "=" * 60)
print("8. Train/Test ユーザーの重複確認")
print("=" * 60)

result = con.execute(f"""
    WITH train_users AS (
        SELECT DISTINCT user_id FROM read_csv_auto('{DATA_DIR}/train_flag.csv')
    ),
    test_users AS (
        SELECT DISTINCT user_id FROM read_csv_auto('{DATA_DIR}/sample_submission.csv')
    ),
    data_users AS (
        SELECT DISTINCT user_id FROM read_csv_auto('{DATA_DIR}/data.csv')
    )
    SELECT
        (SELECT COUNT(*) FROM train_users) as train_count,
        (SELECT COUNT(*) FROM test_users) as test_count,
        (SELECT COUNT(*) FROM data_users) as data_count,
        (SELECT COUNT(*) FROM train_users t JOIN data_users d ON t.user_id = d.user_id) as train_in_data,
        (SELECT COUNT(*) FROM test_users t JOIN data_users d ON t.user_id = d.user_id) as test_in_data,
        (SELECT COUNT(*) FROM train_users t JOIN test_users te ON t.user_id = te.user_id) as train_test_overlap
""").fetchone()
print(f"Train ユーザー数: {result[0]:,}")
print(f"Test ユーザー数: {result[1]:,}")
print(f"Data ユーザー数: {result[2]:,}")
print(f"Train ∩ Data: {result[3]:,}")
print(f"Test ∩ Data: {result[4]:,}")
print(f"Train ∩ Test: {result[5]:,}")

con.close()
print("\n✅ EDA完了")
