# -*- coding: utf-8 -*-
"""
売上予測モデル（料理カテゴリ別）
CUISINE_CAT_originごとにモデルを作成
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sentence_transformers import SentenceTransformer
import warnings
import pickle
warnings.filterwarnings('ignore')

# 結果フォルダの作成
RESULTS_DIR = 'sales_prediction_results'
CUISINE_RESULTS_DIR = os.path.join(RESULTS_DIR, 'by_cuisine')
os.makedirs(CUISINE_RESULTS_DIR, exist_ok=True)

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

# ==========================================
# 2. CUISINE_CAT_originのカテゴリ処理
# ==========================================
print("\n" + "=" * 60)
print("ステップ2: CUISINE_CAT_originのカテゴリ処理...")
print("=" * 60)

# カテゴリごとのデータ数を確認
cuisine_counts = df['CUISINE_CAT_origin'].value_counts()
print(f"\n全カテゴリ数: {len(cuisine_counts)}")
print(f"サンプル数が10以上のカテゴリ数: {sum(cuisine_counts >= 10)}")
print(f"サンプル数が10未満のカテゴリ数: {sum(cuisine_counts < 10)}")

# n<10のカテゴリをotherにまとめる
df['CUISINE_CAT_processed'] = df['CUISINE_CAT_origin'].apply(
    lambda x: x if cuisine_counts.get(x, 0) >= 10 else 'other'
)

# 処理後のカテゴリ数
processed_counts = df['CUISINE_CAT_processed'].value_counts()
print(f"\n処理後のカテゴリ数: {len(processed_counts)}")
print(f"\n各カテゴリのサンプル数:")
for cat, count in processed_counts.items():
    print(f"  {cat}: {count}")

# ==========================================
# 3. エンベディングの読み込み
# ==========================================
print("\n" + "=" * 60)
print("ステップ3: テキストエンベディング読み込み...")
print("=" * 60)

embedding_file = os.path.join(RESULTS_DIR, 'embeddings_data.pkl')

if os.path.exists(embedding_file):
    print(f"  既存のエンベディングファイルを読み込んでいます...")
    with open(embedding_file, 'rb') as f:
        embeddings_dict = pickle.load(f)
    print(f"  [完了] エンベディングを読み込みました")
else:
    print("  エラー: エンベディングファイルが見つかりません。")
    print("  先に sales_prediction_plotly.py を実行してください。")
    sys.exit(1)

# ==========================================
# 4. 特徴量の準備
# ==========================================
print("\n" + "=" * 60)
print("ステップ4: 特徴量の準備中...")
print("=" * 60)

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
X_numeric = X_numeric.fillna(X_numeric.median())

# エンベディングを結合
X_combined = pd.concat([X_numeric] + list(embeddings_dict.values()), axis=1)

# ターゲット変数
y = df['target_amount_tableau'].copy()

# 欠損値を持つ行を削除
valid_idx = ~y.isna()
X_combined = X_combined[valid_idx]
y = y[valid_idx]
df_valid = df[valid_idx].reset_index(drop=True)
X_combined = X_combined.reset_index(drop=True)
y = y.reset_index(drop=True)

print(f"特徴量の形状: {X_combined.shape}")
print(f"有効なサンプル数: {len(y)}")

# ==========================================
# 5. カテゴリごとにモデル訓練と評価
# ==========================================
print("\n" + "=" * 60)
print("ステップ5: カテゴリごとにモデル訓練...")
print("=" * 60)

# 結果を保存する辞書
results_summary = []
all_predictions = []

# 各カテゴリでループ
cuisines = df_valid['CUISINE_CAT_processed'].unique()
print(f"\n処理対象のカテゴリ数: {len(cuisines)}")

for i, cuisine in enumerate(sorted(cuisines), 1):
    print(f"\n{'=' * 60}")
    print(f"[{i}/{len(cuisines)}] カテゴリ: {cuisine}")
    print(f"{'=' * 60}")
    
    # カテゴリごとにデータを抽出
    cuisine_idx = df_valid['CUISINE_CAT_processed'] == cuisine
    X_cuisine = X_combined[cuisine_idx].reset_index(drop=True)
    y_cuisine = y[cuisine_idx].reset_index(drop=True)
    df_cuisine = df_valid[cuisine_idx].reset_index(drop=True)
    
    n_samples = len(y_cuisine)
    print(f"  サンプル数: {n_samples}")
    
    # サンプル数が少なすぎる場合はスキップ
    if n_samples < 20:
        print(f"  [スキップ] サンプル数が少なすぎます（最低20必要）")
        continue
    
    # データ分割
    try:
        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X_cuisine, y_cuisine, df_cuisine.index, 
            test_size=0.2, random_state=42
        )
    except Exception as e:
        print(f"  [エラー] データ分割に失敗: {e}")
        continue
    
    print(f"  訓練データ: {len(X_train)}, テストデータ: {len(X_test)}")
    
    # スケーリング
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # モデル訓練
    print(f"  モデル訓練中...")
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
    
    # 予測
    rf_pred_train = rf_model.predict(X_train_scaled)
    rf_pred_test = rf_model.predict(X_test_scaled)
    
    # 全データの予測
    X_all_scaled = scaler.transform(X_cuisine)
    rf_pred_all = rf_model.predict(X_all_scaled)
    
    # 評価指標の計算
    # テストデータ
    test_r2 = r2_score(y_test, rf_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, rf_pred_test))
    test_mae = mean_absolute_error(y_test, rf_pred_test)
    test_mape = np.mean(np.abs((y_test - rf_pred_test) / y_test)) * 100
    
    # 訓練データ
    train_r2 = r2_score(y_train, rf_pred_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, rf_pred_train))
    train_mae = mean_absolute_error(y_train, rf_pred_train)
    train_mape = np.mean(np.abs((y_train - rf_pred_train) / y_train)) * 100
    
    # 全データ
    all_r2 = r2_score(y_cuisine, rf_pred_all)
    all_rmse = np.sqrt(mean_squared_error(y_cuisine, rf_pred_all))
    all_mae = mean_absolute_error(y_cuisine, rf_pred_all)
    all_mape = np.mean(np.abs((y_cuisine - rf_pred_all) / y_cuisine)) * 100
    
    print(f"\n  【評価指標】")
    print(f"  テスト - R²: {test_r2:.4f}, RMSE: {test_rmse:.2f}, MAPE: {test_mape:.2f}%")
    print(f"  訓練   - R²: {train_r2:.4f}, RMSE: {train_rmse:.2f}, MAPE: {train_mape:.2f}%")
    print(f"  全体   - R²: {all_r2:.4f}, RMSE: {all_rmse:.2f}, MAPE: {all_mape:.2f}%")
    
    # 結果をサマリーに追加
    results_summary.append({
        'cuisine': cuisine,
        'n_samples': n_samples,
        'n_train': len(X_train),
        'n_test': len(X_test),
        'test_r2': test_r2,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'test_mape': test_mape,
        'train_r2': train_r2,
        'train_rmse': train_rmse,
        'train_mae': train_mae,
        'train_mape': train_mape,
        'all_r2': all_r2,
        'all_rmse': all_rmse,
        'all_mae': all_mae,
        'all_mape': all_mape
    })
    
    # 特徴量重要度
    feature_importance = pd.DataFrame({
        'feature': X_combined.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # テストデータの詳細
    test_df = df_cuisine.iloc[idx_test].copy()
    test_df = test_df.reset_index(drop=True)
    test_df['actual'] = y_test.values
    test_df['predicted'] = rf_pred_test
    test_df['error'] = test_df['actual'] - test_df['predicted']
    test_df['abs_error'] = np.abs(test_df['error'])
    test_df['percentage_error'] = np.abs(test_df['error'] / test_df['actual']) * 100
    
    test_df['hover_text'] = (
        'CITY: ' + test_df['CITY'].astype(str) + '<br>' +
        'CUISINE: ' + test_df['CUISINE_CAT_origin'].astype(str) + '<br>' +
        'RESTAURANT: ' + test_df['RST_TITLE'].astype(str) + '<br>' +
        'NUM_SEATS: ' + test_df['NUM_SEATS'].astype(str) + '<br>' +
        'RATING: ' + test_df['RATING_SCORE'].astype(str)
    )
    
    # ==========================================
    # グラフの作成
    # ==========================================
    print(f"  グラフ作成中...")
    
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
            nbinsx=30
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Histogram(
            x=test_df['predicted'],
            name='予測値',
            marker_color='rgba(255, 127, 127, 0.6)',
            nbinsx=30
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
        title_text=f"売上予測分析結果: {cuisine}<br><sub>Test - R²: {test_r2:.4f}, RMSE: {test_rmse:.2f}, MAPE: {test_mape:.2f}% | Train - R²: {train_r2:.4f}, RMSE: {train_rmse:.2f}, MAPE: {train_mape:.2f}%</sub>",
        title_font_size=18,
        height=1000,
        showlegend=True,
        hovermode='closest',
        font=dict(size=10),
        barmode='overlay'
    )
    
    # HTMLファイルとして保存
    safe_cuisine_name = cuisine.replace('/', '_').replace('\\', '_').replace(':', '_')
    output_html = os.path.join(CUISINE_RESULTS_DIR, f'{safe_cuisine_name}_results.html')
    fig.write_html(output_html)
    print(f"  [完了] グラフを保存: {output_html}")
    
    # 予測結果を保存
    output_csv = os.path.join(CUISINE_RESULTS_DIR, f'{safe_cuisine_name}_predictions.csv')
    test_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    
    # 全データの予測結果を保存
    all_results_df = df_cuisine.copy()
    all_results_df['actual'] = y_cuisine
    all_results_df['predicted'] = rf_pred_all
    all_results_df['error'] = all_results_df['actual'] - all_results_df['predicted']
    all_results_df['abs_error'] = np.abs(all_results_df['error'])
    all_results_df['percentage_error'] = np.abs(all_results_df['error'] / all_results_df['actual']) * 100
    
    output_all_csv = os.path.join(CUISINE_RESULTS_DIR, f'{safe_cuisine_name}_predictions_all.csv')
    all_results_df.to_csv(output_all_csv, index=False, encoding='utf-8-sig')

# ==========================================
# 6. サマリーの作成と保存
# ==========================================
print("\n" + "=" * 60)
print("ステップ6: サマリーの作成...")
print("=" * 60)

# サマリーDataFrame
summary_df = pd.DataFrame(results_summary)
summary_df = summary_df.sort_values('n_samples', ascending=False)

# サマリーを保存
summary_output = os.path.join(CUISINE_RESULTS_DIR, 'summary.csv')
summary_df.to_csv(summary_output, index=False, encoding='utf-8-sig')
print(f"\n[完了] サマリーを保存: {summary_output}")

# サマリーを表示
print("\n【カテゴリ別モデル性能サマリー】")
print(summary_df.to_string(index=False))

# ==========================================
# 7. 総合グラフの作成
# ==========================================
print("\n" + "=" * 60)
print("ステップ7: 総合グラフの作成...")
print("=" * 60)

# カテゴリ別の性能比較グラフ
fig_summary = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        'テストR² by カテゴリ',
        'テストRMSE by カテゴリ',
        'テストMAPE by カテゴリ',
        'サンプル数 by カテゴリ'
    ),
    specs=[
        [{"type": "bar"}, {"type": "bar"}],
        [{"type": "bar"}, {"type": "bar"}]
    ],
    vertical_spacing=0.15,
    horizontal_spacing=0.12
)

# R²
fig_summary.add_trace(
    go.Bar(
        x=summary_df['cuisine'],
        y=summary_df['test_r2'],
        marker_color='lightblue',
        name='Test R²',
        hovertemplate='<b>%{x}</b><br>R²: %{y:.4f}<extra></extra>'
    ),
    row=1, col=1
)

# RMSE
fig_summary.add_trace(
    go.Bar(
        x=summary_df['cuisine'],
        y=summary_df['test_rmse'],
        marker_color='lightcoral',
        name='Test RMSE',
        hovertemplate='<b>%{x}</b><br>RMSE: %{y:.2f}<extra></extra>'
    ),
    row=1, col=2
)

# MAPE
fig_summary.add_trace(
    go.Bar(
        x=summary_df['cuisine'],
        y=summary_df['test_mape'],
        marker_color='lightgreen',
        name='Test MAPE',
        hovertemplate='<b>%{x}</b><br>MAPE: %{y:.2f}%<extra></extra>'
    ),
    row=2, col=1
)

# サンプル数
fig_summary.add_trace(
    go.Bar(
        x=summary_df['cuisine'],
        y=summary_df['n_samples'],
        marker_color='lightyellow',
        name='サンプル数',
        hovertemplate='<b>%{x}</b><br>サンプル数: %{y}<extra></extra>'
    ),
    row=2, col=2
)

# レイアウト更新
fig_summary.update_xaxes(title_text="カテゴリ", tickangle=-45)
fig_summary.update_yaxes(title_text="R²", row=1, col=1)
fig_summary.update_yaxes(title_text="RMSE", row=1, col=2)
fig_summary.update_yaxes(title_text="MAPE (%)", row=2, col=1)
fig_summary.update_yaxes(title_text="サンプル数", row=2, col=2)

fig_summary.update_layout(
    title_text="カテゴリ別モデル性能サマリー",
    title_font_size=20,
    height=1000,
    showlegend=False,
    font=dict(size=10)
)

# 保存
summary_html = os.path.join(CUISINE_RESULTS_DIR, 'summary_comparison.html')
fig_summary.write_html(summary_html)
print(f"[完了] 総合グラフを保存: {summary_html}")

# ==========================================
# 完了
# ==========================================
print("\n" + "=" * 60)
print("【完了】すべての処理が完了しました！")
print("=" * 60)

print(f"\n【生成されたファイル】")
print(f"1. {CUISINE_RESULTS_DIR}/ - カテゴリごとの結果フォルダ")
print(f"   - 各カテゴリのHTMLグラフ")
print(f"   - 各カテゴリの予測結果CSV")
print(f"2. {summary_output} - 性能サマリーCSV")
print(f"3. {summary_html} - 総合比較グラフ")

print(f"\n処理したカテゴリ数: {len(results_summary)}")
print(f"総合比較グラフをブラウザで開いて確認してください！")

