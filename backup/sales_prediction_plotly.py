# -*- coding: utf-8 -*-
"""
売上予測モデル（Plotly版）
テキストエンベディングを使用し、結果を保存
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sentence_transformers import SentenceTransformer
import warnings
import pickle
warnings.filterwarnings('ignore')

# 結果フォルダの作成
RESULTS_DIR = 'sales_prediction_results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# ==========================================
# 1. データ読み込み
# ==========================================
print("=" * 60)
print("ステップ1: データ読み込み中...")
print("=" * 60)

print("  Excelファイルを読み込んでいます...")
df = pd.read_excel('aggregated_df.xlsx')
print(f"  [完了] 読み込み完了！")
print(f"  元データ形状: {df.shape[0]} 行 × {df.shape[1]} 列")

# target_amount_tableauを20以上1000以下に絞る
print("\n  target_amount_tableau を 20以上1000以下に絞り込んでいます...")
df = df[(df['target_amount_tableau'] >= 20) & (df['target_amount_tableau'] <= 1000)].reset_index(drop=True)
print(f"  [完了] フィルタリング後のデータ形状: {df.shape[0]} 行 × {df.shape[1]} 列")

# ターゲット変数の確認
print(f"\ntarget_amount_tableau の統計:")
print(df['target_amount_tableau'].describe())

# ==========================================
# 2. テキストエンベディングの作成または読み込み
# ==========================================
print("\n" + "=" * 60)
print("ステップ2: テキストエンベディング処理...")
print("=" * 60)

embedding_file = os.path.join(RESULTS_DIR, 'embeddings_data.pkl')

if os.path.exists(embedding_file):
    print(f"  既存のエンベディングファイルを読み込んでいます...")
    with open(embedding_file, 'rb') as f:
        embeddings_dict = pickle.load(f)
    print(f"  [完了] エンベディングを読み込みました")
else:
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
    
    # エンベディングを保存
    print(f"\n  エンベディング結果を保存中...")
    with open(embedding_file, 'wb') as f:
        pickle.dump(embeddings_dict, f)
    print(f"  [完了] '{embedding_file}' に保存しました")

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

# ターゲット変数（対数変換）
print("\n  ターゲット変数を対数変換しています...")
y_original = df['target_amount_tableau'].copy()
y = np.log(y_original)  # 対数変換

# 欠損値を持つ行を削除
valid_idx = ~y.isna()
X_combined = X_combined[valid_idx]
y = y[valid_idx]
y_original = y_original[valid_idx]
df_valid = df[valid_idx].reset_index(drop=True)

print(f"\n欠損値除去後のデータ形状: {X_combined.shape}")
print(f"有効なサンプル数: {len(y)}")
print(f"対数変換後のターゲット変数の統計:")
print(f"  平均: {y.mean():.4f}, 標準偏差: {y.std():.4f}")

# ==========================================
# 4. データ分割
# ==========================================
print("\n" + "=" * 60)
print("ステップ4: データ分割中...")
print("=" * 60)
print("  訓練データとテストデータに分割しています（80:20）...")

# インデックスとy_originalも保持して分割
X_train, X_test, y_train, y_test, y_train_original, y_test_original, idx_train, idx_test = train_test_split(
    X_combined, y, y_original, df_valid.index, test_size=0.2, random_state=42
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
rf_pred_train_log = rf_model.predict(X_train_scaled)
rf_pred_test_log = rf_model.predict(X_test_scaled)

# 対数変換を元に戻す（指数変換）
print("  予測値を指数変換して元のスケールに戻しています...")
rf_pred_train = np.exp(rf_pred_train_log)
rf_pred_test = np.exp(rf_pred_test_log)

print("\n  【結果（元のスケール）】")
print(f"  訓練 R²: {r2_score(y_train_original, rf_pred_train):.4f}")
print(f"  訓練 RMSE: {np.sqrt(mean_squared_error(y_train_original, rf_pred_train)):.2f}")
print(f"  訓練 MAE: {mean_absolute_error(y_train_original, rf_pred_train):.2f}")
print(f"\n  テスト R²: {r2_score(y_test_original, rf_pred_test):.4f}")
print(f"  テスト RMSE: {np.sqrt(mean_squared_error(y_test_original, rf_pred_test)):.2f}")
print(f"  テスト MAE: {mean_absolute_error(y_test_original, rf_pred_test):.2f}")
print("  [完了] Random Forest 完了！")

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

# ==========================================
# 8. 詳細な評価指標の計算
# ==========================================
print("\n" + "=" * 60)
print("ステップ8: 詳細な評価指標を計算中...")
print("=" * 60)

# テストデータの評価指標（元のスケールで計算）
test_mape = np.mean(np.abs((y_test_original - rf_pred_test) / y_test_original)) * 100
test_rmse = np.sqrt(mean_squared_error(y_test_original, rf_pred_test))
test_r2 = r2_score(y_test_original, rf_pred_test)

print("\n【テストデータの評価指標（元のスケール）】")
print(f"  R²: {test_r2:.4f}")
print(f"  RMSE: {test_rmse:.2f}")
print(f"  MAE: {mean_absolute_error(y_test_original, rf_pred_test):.2f}")
print(f"  MAPE: {test_mape:.2f}%")

# 訓練データの評価指標（元のスケールで計算）
train_mape = np.mean(np.abs((y_train_original - rf_pred_train) / y_train_original)) * 100
train_rmse = np.sqrt(mean_squared_error(y_train_original, rf_pred_train))
train_r2 = r2_score(y_train_original, rf_pred_train)

print("\n【訓練データの評価指標（元のスケール）】")
print(f"  R²: {train_r2:.4f}")
print(f"  RMSE: {train_rmse:.2f}")
print(f"  MAE: {mean_absolute_error(y_train_original, rf_pred_train):.2f}")
print(f"  MAPE: {train_mape:.2f}%")

# 全データの予測（対数スケールで予測して指数変換）
X_all_scaled = scaler.transform(X_combined)
rf_pred_all_log = rf_model.predict(X_all_scaled)
rf_pred_all = np.exp(rf_pred_all_log)

# 全データの評価指標（元のスケールで計算）
all_mape = np.mean(np.abs((y_original - rf_pred_all) / y_original)) * 100
all_rmse = np.sqrt(mean_squared_error(y_original, rf_pred_all))
all_r2 = r2_score(y_original, rf_pred_all)

print("\n【全データの評価指標（元のスケール）】")
print(f"  R²: {all_r2:.4f}")
print(f"  RMSE: {all_rmse:.2f}")
print(f"  MAE: {mean_absolute_error(y_original, rf_pred_all):.2f}")
print(f"  MAPE: {all_mape:.2f}%")

# ==========================================
# 9. Plotlyで可視化
# ==========================================
print("\n" + "=" * 60)
print("ステップ9: Plotlyで可視化を作成中...")
print("=" * 60)

# テストデータの情報を含むDataFrameを作成（元のスケールで）
test_df = df_valid.iloc[idx_test].copy()
test_df['actual'] = y_test_original.values
test_df['predicted'] = rf_pred_test
test_df['error'] = test_df['actual'] - test_df['predicted']
test_df['abs_error'] = np.abs(test_df['error'])
test_df['percentage_error'] = np.abs(test_df['error'] / test_df['actual']) * 100

# カスタム情報の作成
test_df['hover_text'] = (
    'CITY: ' + test_df['CITY'].astype(str) + '<br>' +
    'CUISINE: ' + test_df['CUISINE_CAT_origin'].astype(str) + '<br>' +
    'RESTAURANT: ' + test_df['RST_TITLE'].astype(str) + '<br>' +
    'NUM_SEATS: ' + test_df['NUM_SEATS'].astype(str) + '<br>' +
    'RATING: ' + test_df['RATING_SCORE'].astype(str)
)

# サブプロットの作成
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        f'実測値 vs 予測値 (Test RMSE: {test_rmse:.2f}, MAPE: {test_mape:.2f}%)',
        '実測値と予測値のヒストグラム',
        'Absolute Percentage Error vs 実測値',
        '特徴量重要度 Top 20'
    ),
    specs=[
        [{"type": "scatter"}, {"type": "bar"}],
        [{"type": "scatter"}, {"type": "bar"}]
    ],
    vertical_spacing=0.12,
    horizontal_spacing=0.12
)

# 1. 実測値 vs 予測値の散布図
scatter_trace = go.Scatter(
    x=test_df['actual'],
    y=test_df['predicted'],
    mode='markers',
    marker=dict(
        size=5,
        color=test_df['abs_error'],
        colorscale='Viridis',
        showscale=True,
        colorbar=dict(
            title="誤差",
            x=0.46,
            len=0.4,
            y=0.77
        )
    ),
    text=test_df['hover_text'],
    hovertemplate='<b>実測値:</b> %{x:.2f}<br>' +
                  '<b>予測値:</b> %{y:.2f}<br>' +
                  '<br>%{text}<extra></extra>',
    name='予測'
)

# 理想線
min_val = min(test_df['actual'].min(), test_df['predicted'].min())
max_val = max(test_df['actual'].max(), test_df['predicted'].max())
ideal_line = go.Scatter(
    x=[min_val, max_val],
    y=[min_val, max_val],
    mode='lines',
    line=dict(color='red', dash='dash', width=2),
    name='理想線',
    hoverinfo='skip'
)

fig.add_trace(scatter_trace, row=1, col=1)
fig.add_trace(ideal_line, row=1, col=1)

# 2. ヒストグラム（重ねて表示）
fig.add_trace(
    go.Histogram(
        x=test_df['actual'],
        name='実測値',
        marker_color='rgba(135, 206, 250, 0.6)',
        nbinsx=50
    ),
    row=1, col=2
)

fig.add_trace(
    go.Histogram(
        x=test_df['predicted'],
        name='予測値',
        marker_color='rgba(255, 127, 127, 0.6)',
        nbinsx=50
    ),
    row=1, col=2
)

# 3. Absolute Percentage Error vs 実測値
ape_trace = go.Scatter(
    x=test_df['actual'],
    y=test_df['percentage_error'],
    mode='markers',
    marker=dict(
        size=5,
        color='rgba(135, 206, 250, 0.6)'
    ),
    text=test_df['hover_text'],
    hovertemplate='<b>実測値:</b> %{x:.2f}<br>' +
                  '<b>誤差率:</b> %{y:.2f}%<br>' +
                  '<br>%{text}<extra></extra>',
    name='誤差率'
)

fig.add_trace(ape_trace, row=2, col=1)

# 4. 特徴量重要度 Top 20
top20 = feature_importance.head(20)
fig.add_trace(
    go.Bar(
        y=top20['feature'][::-1],
        x=top20['importance'][::-1],
        orientation='h',
        marker_color='lightgreen',
        hovertemplate='<b>%{y}</b><br>重要度: %{x:.4f}<extra></extra>'
    ),
    row=2, col=2
)

# レイアウトの更新
fig.update_xaxes(title_text="実測値", row=1, col=1)
fig.update_yaxes(title_text="予測値", row=1, col=1)

fig.update_xaxes(title_text="売上額", row=1, col=2)
fig.update_yaxes(title_text="頻度", row=1, col=2)

fig.update_xaxes(title_text="実測値", row=2, col=1)
fig.update_yaxes(title_text="Absolute Percentage Error (%)", row=2, col=1)

fig.update_xaxes(title_text="重要度", row=2, col=2)
fig.update_yaxes(title_text="特徴量", row=2, col=2)

# 全体のレイアウト
fig.update_layout(
    title_text=f"売上予測分析結果<br><sub>Test - R²: {test_r2:.4f}, RMSE: {test_rmse:.2f}, MAPE: {test_mape:.2f}% | Train - R²: {train_r2:.4f}, RMSE: {train_rmse:.2f}, MAPE: {train_mape:.2f}%</sub>",
    title_font_size=18,
    height=1000,
    showlegend=True,
    hovermode='closest',
    font=dict(size=10),
    barmode='overlay'  # ヒストグラムを重ねて表示
)

# HTMLファイルとして保存
output_html = os.path.join(RESULTS_DIR, 'interactive_results.html')
fig.write_html(output_html)
print(f"  [完了] インタラクティブなグラフを '{output_html}' に保存しました")

# ==========================================
# 10. 結果の保存
# ==========================================
print("\n" + "=" * 60)
print("ステップ10: 結果を保存中...")
print("=" * 60)

# テストデータの予測結果をDataFrameに保存
print("  テストデータの予測結果をCSVに保存中...")
output_results = os.path.join(RESULTS_DIR, 'prediction_results_test.csv')
test_df.to_csv(output_results, index=False, encoding='utf-8-sig')
print(f"  [完了] テストデータの予測結果を '{output_results}' に保存しました")

# 全データの予測結果を保存（元のスケールで）
print("  全データの予測結果をCSVに保存中...")
all_results_df = df_valid.copy()
all_results_df['actual'] = y_original.values
all_results_df['predicted'] = rf_pred_all
all_results_df['error'] = all_results_df['actual'] - all_results_df['predicted']
all_results_df['abs_error'] = np.abs(all_results_df['error'])
all_results_df['percentage_error'] = np.abs(all_results_df['error'] / all_results_df['actual']) * 100
output_all_results = os.path.join(RESULTS_DIR, 'prediction_results_all.csv')
all_results_df.to_csv(output_all_results, index=False, encoding='utf-8-sig')
print(f"  [完了] 全データの予測結果を '{output_all_results}' に保存しました")

# 特徴量重要度を保存
print("  特徴量重要度をCSVに保存中...")
output_importance = os.path.join(RESULTS_DIR, 'feature_importance.csv')
feature_importance.to_csv(output_importance, index=False, encoding='utf-8-sig')
print(f"  [完了] 特徴量重要度を '{output_importance}' に保存しました")

# エンベディング済みの全データを保存
print("  エンベディング済みデータを保存中...")
combined_data = pd.concat([df_valid, X_combined], axis=1)
output_combined = os.path.join(RESULTS_DIR, 'data_with_embeddings.csv')
combined_data.to_csv(output_combined, index=False, encoding='utf-8-sig')
print(f"  [完了] エンベディング済みデータを '{output_combined}' に保存しました")

# ==========================================
# 10. まとめ
# ==========================================
print("\n" + "=" * 60)
print("【完了】すべての処理が完了しました！")
print("=" * 60)

print("\n【生成されたファイル】")
print(f"1. {output_html} - インタラクティブなグラフ（ブラウザで開いてください）")
print(f"2. {output_results} - テストデータの予測結果")
print(f"3. {output_all_results} - 全データの予測結果")
print(f"4. {output_importance} - 特徴量重要度")
print(f"5. {embedding_file} - エンベディングデータ（再利用可能）")
print(f"6. {output_combined} - エンベディング済み全データ")

print("\n【評価指標サマリー】")
print(f"テストデータ   : R²={test_r2:.4f}, RMSE={test_rmse:.2f}, MAPE={test_mape:.2f}%")
print(f"訓練データ     : R²={train_r2:.4f}, RMSE={train_rmse:.2f}, MAPE={train_mape:.2f}%")
print(f"全データ       : R²={all_r2:.4f}, RMSE={all_rmse:.2f}, MAPE={all_mape:.2f}%")

print("\n【テキストエンベディングの効果】")
print("- 文字列変数をベクトル空間に埋め込むことで、意味的な類似性を捉えられます")
print("- エンベディング結果は保存されているので、次回は即座に読み込めます")
print("- インタラクティブなグラフで各データポイントの詳細を確認できます")

print("\n【次のステップ】")
print("- より高度な日本語モデル（例: cl-tohoku/bert-base-japanese）を試す")
print("- ハイパーパラメータチューニング")
print("- クロスバリデーションで安定性を確認")
print("- アンサンブルモデルの構築")

