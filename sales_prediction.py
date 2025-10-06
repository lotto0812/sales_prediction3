# -*- coding: utf-8 -*-
"""
売上予測モデル
文字列カテゴリ変数をテキストエンベディングで処理
"""

import sys
import io
import os
# Windows環境でのUnicodeエラー回避
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
import warnings
import os
warnings.filterwarnings('ignore')

# 結果フォルダの作成
RESULTS_DIR = 'sales_prediction_results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# 日本語設定
plt.rcParams['font.sans-serif'] = ['MS Gothic', 'Yu Gothic', 'Meiryo']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. データ読み込み
# ==========================================
print("=" * 60)
print("ステップ1: データ読み込み中...")
print("=" * 60)

print("  Excelファイルを読み込んでいます...")
df = pd.read_excel('aggregated_df.xlsx')
print(f"  [完了] 読み込み完了！")
print(f"  データ形状: {df.shape[0]} 行 × {df.shape[1]} 列")
print(f"\nカラム一覧:\n{df.columns.tolist()}")

# ターゲット変数の確認
print(f"\ntarget_amount_tableau の統計:")
print(df['target_amount_tableau'].describe())

# ==========================================
# 2. テキストエンベディングの作成
# ==========================================
print("\n" + "=" * 60)
print("ステップ2: テキストエンベディング作成中...")
print("=" * 60)

# 日本語対応のSentence-BERTモデルをロード
# 軽量で高速なモデルを使用
print("  AIモデルをロード中...")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
print(f"  [完了] 使用モデル: paraphrase-multilingual-MiniLM-L12-v2")

# 文字列カラムのリスト
text_columns = ['CITY', 'CUISINE_CAT_origin', 'RST_TITLE']

# 各文字列カラムに対してエンベディングを作成
embeddings_dict = {}

for idx, col in enumerate(text_columns, 1):
    print(f"\n[{idx}/{len(text_columns)}] {col} の処理中...")
    print(f"  ステップ1: データの前処理...")
    
    # 欠損値を "UNKNOWN" で埋める
    texts = df[col].fillna('UNKNOWN').astype(str).tolist()
    
    # ユニークな値の数を表示
    unique_count = df[col].nunique()
    print(f"  - ユニーク値数: {unique_count}")
    print(f"  - 総データ数: {len(texts)}")
    
    print(f"  ステップ2: エンベディングを生成中（時間がかかります）...")
    # エンベディングを生成
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
    
    print(f"  ステップ3: データフレームに変換中...")
    # DataFrameに変換
    embedding_df = pd.DataFrame(
        embeddings, 
        columns=[f'{col}_emb_{i}' for i in range(embeddings.shape[1])]
    )
    
    embeddings_dict[col] = embedding_df
    print(f"  [完了] エンベディング次元: {embeddings.shape[1]}")
    print(f"  ({idx}/{len(text_columns)} カラム完了)")

# ==========================================
# 3. 特徴量の準備
# ==========================================
print("\n" + "=" * 60)
print("ステップ3: 特徴量の準備中...")
print("=" * 60)
print("  数値特徴量とエンベディングを結合しています...")

# 数値特徴量
numeric_features = [
    'AVG_MONTHLY_POPULATION',
    'NEAREST_STATION_INFO_count',
    'DISTANCE_VALUE',
    'RATING_CNT',
    'RATING_SCORE',
    'IS_FAMILY_FRIENDLY',
    'IS_FRIEND_FRIENDLY',
    'IS_ALONE_FRIENDLY',
    'DINNER_INFO',
    'LUNCH_INFO',
    'DINNER_PRICE',
    'LUNCH_PRICE',
    'HOLIDAY',
    'HOME_PAGE_URL',
    'PHONE_NUM',
    'NUM_SEATS',
    'AGE_RESTAURANT',
    'CITY_count',
]

# 数値特徴量のDataFrameを作成
X_numeric = df[numeric_features].copy()

# 欠損値を中央値で埋める
X_numeric = X_numeric.fillna(X_numeric.median())

print(f"数値特徴量の数: {len(numeric_features)}")
print(f"数値特徴量の形状: {X_numeric.shape}")

# エンベディングを結合
X_combined = pd.concat([X_numeric] + list(embeddings_dict.values()), axis=1)

print(f"\n結合後の特徴量の形状: {X_combined.shape}")
print(f"総特徴量数: {X_combined.shape[1]}")

# ターゲット変数
y = df['target_amount_tableau'].copy()

# 欠損値を持つ行を削除
valid_idx = ~y.isna()
X_combined = X_combined[valid_idx]
y = y[valid_idx]

print(f"\n欠損値除去後のデータ形状: {X_combined.shape}")
print(f"有効なサンプル数: {len(y)}")

# ==========================================
# 4. データ分割
# ==========================================
print("\n" + "=" * 60)
print("ステップ4: データ分割中...")
print("=" * 60)
print("  訓練データとテストデータに分割しています（80:20）...")

X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y, test_size=0.2, random_state=42
)

print(f"訓練データ: {X_train.shape}")
print(f"テストデータ: {X_test.shape}")

# ==========================================
# 5. 特徴量のスケーリング
# ==========================================
print("\n" + "=" * 60)
print("ステップ5: 特徴量のスケーリング中...")
print("=" * 60)
print("  特徴量を標準化しています（平均0、標準偏差1）...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("  [完了] スケーリング完了！")

# ==========================================
# 6. モデル訓練と評価
# ==========================================
print("\n" + "=" * 60)
print("ステップ6: モデル訓練と評価中...")
print("=" * 60)

# モデル1: Random Forest
print("\n[モデル 1/2] Random Forest Regressor")
print("  設定: n_estimators=200, max_depth=20")
print("  訓練開始...")
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
    verbose=0
)

rf_model.fit(X_train_scaled, y_train)
print("  訓練完了！予測中...")
rf_pred_train = rf_model.predict(X_train_scaled)
rf_pred_test = rf_model.predict(X_test_scaled)

print("\n  【結果】")
print(f"  訓練 R²: {r2_score(y_train, rf_pred_train):.4f}")
print(f"  訓練 RMSE: {np.sqrt(mean_squared_error(y_train, rf_pred_train)):.2f}")
print(f"  訓練 MAE: {mean_absolute_error(y_train, rf_pred_train):.2f}")
print(f"\n  テスト R²: {r2_score(y_test, rf_pred_test):.4f}")
print(f"  テスト RMSE: {np.sqrt(mean_squared_error(y_test, rf_pred_test)):.2f}")
print(f"  テスト MAE: {mean_absolute_error(y_test, rf_pred_test):.2f}")
print("  [完了] Random Forest 完了！")

# モデル2: Gradient Boosting
print("\n[モデル 2/2] Gradient Boosting Regressor")
print("  設定: n_estimators=200, max_depth=5, learning_rate=0.1")
print("  訓練開始（少し時間がかかります）...")
gb_model = GradientBoostingRegressor(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    verbose=0
)

gb_model.fit(X_train_scaled, y_train)
print("  訓練完了！予測中...")
gb_pred_train = gb_model.predict(X_train_scaled)
gb_pred_test = gb_model.predict(X_test_scaled)

print("\n  【結果】")
print(f"  訓練 R²: {r2_score(y_train, gb_pred_train):.4f}")
print(f"  訓練 RMSE: {np.sqrt(mean_squared_error(y_train, gb_pred_train)):.2f}")
print(f"  訓練 MAE: {mean_absolute_error(y_train, gb_pred_train):.2f}")
print(f"\n  テスト R²: {r2_score(y_test, gb_pred_test):.4f}")
print(f"  テスト RMSE: {np.sqrt(mean_squared_error(y_test, gb_pred_test)):.2f}")
print(f"  テスト MAE: {mean_absolute_error(y_test, gb_pred_test):.2f}")
print("  [完了] Gradient Boosting 完了！")

# ==========================================
# 7. 特徴量重要度の分析
# ==========================================
print("\n" + "=" * 60)
print("ステップ7: 特徴量重要度の分析中...")
print("=" * 60)
print("  どの特徴量が予測に重要かを分析しています...")

# Random Forestの特徴量重要度
feature_importance = pd.DataFrame({
    'feature': X_combined.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n上位20の重要な特徴量:")
print(feature_importance.head(20).to_string(index=False))

# 特徴量タイプ別の重要度集計
feature_importance['type'] = feature_importance['feature'].apply(
    lambda x: 'CITY_emb' if 'CITY_emb' in x else
              'CUISINE_emb' if 'CUISINE_CAT_origin_emb' in x else
              'RST_TITLE_emb' if 'RST_TITLE_emb' in x else
              'numeric'
)

type_importance = feature_importance.groupby('type')['importance'].sum().sort_values(ascending=False)
print("\n特徴量タイプ別の重要度:")
print(type_importance)

# ==========================================
# 8. 可視化
# ==========================================
print("\n" + "=" * 60)
print("ステップ8: 可視化を作成中...")
print("=" * 60)
print("  4つのグラフを生成しています...")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. 予測値 vs 実測値 (Random Forest)
axes[0, 0].scatter(y_test, rf_pred_test, alpha=0.5, s=30)
axes[0, 0].plot([y_test.min(), y_test.max()], 
                [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 0].set_xlabel('実測値', fontsize=12)
axes[0, 0].set_ylabel('予測値', fontsize=12)
axes[0, 0].set_title(f'Random Forest - 予測 vs 実測\nR² = {r2_score(y_test, rf_pred_test):.4f}', 
                     fontsize=13, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# 2. 予測値 vs 実測値 (Gradient Boosting)
axes[0, 1].scatter(y_test, gb_pred_test, alpha=0.5, s=30, color='green')
axes[0, 1].plot([y_test.min(), y_test.max()], 
                [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 1].set_xlabel('実測値', fontsize=12)
axes[0, 1].set_ylabel('予測値', fontsize=12)
axes[0, 1].set_title(f'Gradient Boosting - 予測 vs 実測\nR² = {r2_score(y_test, gb_pred_test):.4f}', 
                     fontsize=13, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# 3. 特徴量重要度 (上位15)
top_features = feature_importance.head(15)
axes[1, 0].barh(range(len(top_features)), top_features['importance'].values)
axes[1, 0].set_yticks(range(len(top_features)))
axes[1, 0].set_yticklabels(top_features['feature'].values, fontsize=9)
axes[1, 0].invert_yaxis()
axes[1, 0].set_xlabel('重要度', fontsize=12)
axes[1, 0].set_title('特徴量重要度 (上位15)', fontsize=13, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3, axis='x')

# 4. 特徴量タイプ別の重要度
axes[1, 1].bar(range(len(type_importance)), type_importance.values)
axes[1, 1].set_xticks(range(len(type_importance)))
axes[1, 1].set_xticklabels(type_importance.index, rotation=45, ha='right', fontsize=10)
axes[1, 1].set_ylabel('合計重要度', fontsize=12)
axes[1, 1].set_title('特徴量タイプ別の重要度', fontsize=13, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
print("  グラフを保存中...")
output_plot = os.path.join(RESULTS_DIR, 'sales_prediction_results.png')
plt.savefig(output_plot, dpi=300, bbox_inches='tight')
print(f"  [完了] 結果を '{output_plot}' に保存しました")

# ==========================================
# 9. 結果の保存
# ==========================================
print("\n" + "=" * 60)
print("ステップ9: 結果を保存中...")
print("=" * 60)

# 予測結果をDataFrameに保存
print("  予測結果をCSVに保存中...")
results_df = pd.DataFrame({
    'actual': y_test,
    'rf_predicted': rf_pred_test,
    'gb_predicted': gb_pred_test,
    'rf_error': y_test - rf_pred_test,
    'gb_error': y_test - gb_pred_test
})
output_results = os.path.join(RESULTS_DIR, 'prediction_results.csv')
results_df.to_csv(output_results, index=False, encoding='utf-8-sig')
print(f"  [完了] 予測結果を '{output_results}' に保存しました")

# 特徴量重要度を保存
print("  特徴量重要度をCSVに保存中...")
output_importance = os.path.join(RESULTS_DIR, 'feature_importance.csv')
feature_importance.to_csv(output_importance, index=False, encoding='utf-8-sig')
print(f"  [完了] 特徴量重要度を '{output_importance}' に保存しました")

# ==========================================
# 10. まとめ
# ==========================================
print("\n" + "=" * 60)
print("【完了】すべての処理が完了しました！")
print("=" * 60)

print("\n【テキストエンベディングの効果】")
print("- 文字列変数をベクトル空間に埋め込むことで、意味的な類似性を捉えられます")
print("- 例: 「カレー」と「カレーライス」は近いベクトルになります")
print("- 例: チェーン店名は互いに近いベクトルになります")
print("\n【使用したアプローチ】")
print("1. Sentence Transformersを使用した多言語対応のテキストエンベディング")
print("2. 数値特徴量とエンベディングを結合")
print("3. Random ForestとGradient Boostingで予測")
print("\n【次のステップ】")
print("- より高度な日本語モデル（例: cl-tohoku/bert-base-japanese）を試す")
print("- ハイパーパラメータチューニング")
print("- クロスバリデーションで安定性を確認")
print("- アンサンブルモデルの構築")

