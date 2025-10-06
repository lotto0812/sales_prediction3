# -*- coding: utf-8 -*-
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
import os
warnings.filterwarnings('ignore')

print("=" * 100)
print("最終店舗クラスタリング分析")
print("=" * 100)

# 結果フォルダを作成
results_folder = 'final_store_clustering_results'
os.makedirs(results_folder, exist_ok=True)
print(f"\n結果フォルダ: {results_folder}")

# データを読み込む
print("\n[1/7] データを読み込んでいます...")
print("  ファイル: aggregated_df_filtered.csv")
df = pd.read_csv('aggregated_df_filtered.csv')
print(f"  データ形状: {df.shape}")

# 特徴量を選択
feature_cols = ['AVG_MONTHLY_POPULATION', 'NUM_SEATS', 'DINNER_PRICE', 'IS_FAMILY_FRIENDLY', 'cuisine_cluster_id']
print(f"\n[2/7] 特徴量を選択しています...")
print(f"  特徴量: {feature_cols}")

# 欠損値の確認と処理
print("\n[3/7] 欠損値を確認・処理しています...")
missing_counts = df[feature_cols].isnull().sum()
print("  欠損値の数:")
for col in feature_cols:
    print(f"    {col}: {missing_counts[col]} ({missing_counts[col]/len(df)*100:.1f}%)")

# 欠損値を含む行を削除
df_analysis = df.dropna(subset=feature_cols).copy()
print(f"  欠損値を含む行を削除: {len(df_analysis)}店舗")

# 特徴量を抽出
X = df_analysis[feature_cols].values
print(f"\n  クラスタリング対象: {X.shape[0]}店舗, {X.shape[1]}特徴量")

# 特徴量の基本統計
print("\n  【特徴量の基本統計】")
stats_df = df_analysis[feature_cols].describe()
print(stats_df.to_string(float_format="%.2f"))

# 標準化
print("\n[4/7] 特徴量を標準化しています...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 最適なクラスタ数を探索
print("\n[5/7] 最適なクラスタ数を探索しています（エルボー法・シルエット分析）...")
inertias = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
    print(f"  k={k}: イナーシャ={kmeans.inertia_:.2f}, シルエット係数={silhouette_scores[-1]:.3f}")

# Plotlyでエルボー法とシルエット分析のグラフを作成
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=('エルボー法（最適クラスタ数の探索）', 'シルエット係数分析'),
)

# エルボー法
fig.add_trace(
    go.Scatter(
        x=list(K_range), y=inertias,
        mode='lines+markers',
        marker=dict(size=10, color='blue'),
        line=dict(color='blue', width=2),
        name='イナーシャ',
        hovertemplate='クラスタ数: %{x}<br>イナーシャ: %{y:.2f}<extra></extra>'
    ),
    row=1, col=1
)

# シルエット係数
fig.add_trace(
    go.Scatter(
        x=list(K_range), y=silhouette_scores,
        mode='lines+markers',
        marker=dict(size=10, color='red'),
        line=dict(color='red', width=2),
        name='シルエット係数',
        hovertemplate='クラスタ数: %{x}<br>シルエット係数: %{y:.3f}<extra></extra>'
    ),
    row=1, col=2
)

fig.update_xaxes(title_text="クラスタ数 (k)", row=1, col=1)
fig.update_yaxes(title_text="イナーシャ", row=1, col=1)
fig.update_xaxes(title_text="クラスタ数 (k)", row=1, col=2)
fig.update_yaxes(title_text="シルエット係数", row=1, col=2)

fig.update_layout(height=500, width=1200, showlegend=True)
fig.write_html(os.path.join(results_folder, 'optimal_k_analysis.html'))
print(f"\n  グラフを保存: optimal_k_analysis.html")

# 最適なクラスタ数を決定
optimal_k = K_range[np.argmax(silhouette_scores)]
print(f"\n  最適なクラスタ数: {optimal_k} (シルエット係数: {max(silhouette_scores):.3f})")

# 最終的なクラスタリング
print(f"\n[6/7] k={optimal_k}でクラスタリングを実行しています...")
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = kmeans_final.fit_predict(X_scaled)

df_analysis['store_cluster_old'] = cluster_labels

# 中央値でクラスタを並び替え
print("\n  中央値でクラスタを並び替えています...")
cluster_medians = df_analysis.groupby('store_cluster_old')['target_amount_tableau'].median().sort_values()
print("\n  【元のクラスタと中央値】")
for old_id, median in cluster_medians.items():
    count = len(df_analysis[df_analysis['store_cluster_old'] == old_id])
    print(f"    旧クラスタ {old_id}: 中央値 {median:.2f}円 ({count}店舗)")

# 中央値昇順でクラスタIDを再割り当て
old_to_new_mapping = {old_id: new_id for new_id, old_id in enumerate(cluster_medians.index)}
df_analysis['store_cluster'] = df_analysis['store_cluster_old'].map(old_to_new_mapping)

print("\n  【新しいクラスタID（中央値昇順）】")
for new_id in range(optimal_k):
    cluster_data = df_analysis[df_analysis['store_cluster'] == new_id]
    median = cluster_data['target_amount_tableau'].median()
    print(f"    新ID {new_id}: 中央値 {median:.2f}円, 店舗数: {len(cluster_data)}")

# クラスタ中心を元のスケールに戻す（新しいIDで並び替え）
centers_original = scaler.inverse_transform(kmeans_final.cluster_centers_)
centers_df = pd.DataFrame(centers_original, columns=feature_cols)
centers_df.insert(0, 'old_cluster', range(optimal_k))
centers_df['new_cluster'] = centers_df['old_cluster'].map(old_to_new_mapping)
centers_df = centers_df.sort_values('new_cluster')

print("\n  【クラスタ中心（元のスケール、新ID順）】")
display_cols = ['new_cluster'] + feature_cols
print(centers_df[display_cols].to_string(index=False, float_format="%.2f"))

# クラスタ別統計（新しいIDで）
print("\n[7/7] クラスタ別統計を計算しています...")

# target_amount_tableauの詳細統計を計算
cluster_stats = df_analysis.groupby('store_cluster')['target_amount_tableau'].agg([
    ('データ数', 'count'),
    ('平均', 'mean'),
    ('標準偏差', 'std'),
    ('Min', 'min'),
    ('5%', lambda x: x.quantile(0.05)),
    ('10%', lambda x: x.quantile(0.10)),
    ('25%', lambda x: x.quantile(0.25)),
    ('50%(中央値)', lambda x: x.quantile(0.50)),
    ('75%', lambda x: x.quantile(0.75)),
    ('90%', lambda x: x.quantile(0.90)),
    ('95%', lambda x: x.quantile(0.95)),
    ('Max', 'max')
]).reset_index()

# クラスタ名を整形
cluster_stats['store_cluster'] = cluster_stats['store_cluster'].apply(lambda x: f'クラスタ {int(x)}')

print("\n【クラスタ別統計表（target_amount_tableau）】")
print("=" * 120)
print(cluster_stats.to_string(index=False, float_format="%.2f"))

# CSVに保存
output_csv = os.path.join(results_folder, 'cluster_statistics.csv')
cluster_stats.to_csv(output_csv, encoding='utf-8-sig', index=False)
print(f"\n  統計表を保存: cluster_statistics.csv")

# Excelにも保存
output_excel = os.path.join(results_folder, 'cluster_statistics.xlsx')
with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
    cluster_stats.to_excel(writer, sheet_name='クラスタ統計', index=False)
    
    # 列幅を調整
    worksheet = writer.sheets['クラスタ統計']
    worksheet.column_dimensions['A'].width = 20
    for col in ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M']:
        worksheet.column_dimensions[col].width = 15

print(f"  統計表を保存: cluster_statistics.xlsx")

# クラスタ別サマリー（特徴量の平均値）
cluster_summary = []
for cluster_id in range(optimal_k):
    cluster_data = df_analysis[df_analysis['store_cluster'] == cluster_id]
    
    summary = {
        'cluster_id': cluster_id,
        'count': len(cluster_data),
        'percentage': len(cluster_data) / len(df_analysis) * 100,
    }
    
    # 各特徴量の平均
    for col in feature_cols:
        summary[f'{col}_mean'] = cluster_data[col].mean()
    
    # target_amount_tableauの統計
    summary['target_mean'] = cluster_data['target_amount_tableau'].mean()
    summary['target_median'] = cluster_data['target_amount_tableau'].median()
    summary['target_std'] = cluster_data['target_amount_tableau'].std()
    
    cluster_summary.append(summary)

summary_df = pd.DataFrame(cluster_summary)
summary_df.to_csv(os.path.join(results_folder, 'cluster_summary.csv'), encoding='utf-8-sig', index=False)

print("\n  【クラスタ別サマリー（中央値昇順）】")
for _, row in summary_df.iterrows():
    print(f"\n  ■ クラスタ {int(row['cluster_id'])} ({int(row['count'])}店舗, {row['percentage']:.1f}%)")
    print(f"      月間人口: {row['AVG_MONTHLY_POPULATION_mean']:.0f}")
    print(f"      座席数: {row['NUM_SEATS_mean']:.1f}")
    print(f"      ディナー価格: {row['DINNER_PRICE_mean']:.0f}円")
    print(f"      ファミリー向け: {row['IS_FAMILY_FRIENDLY_mean']:.2f}")
    print(f"      料理クラスタ: {row['cuisine_cluster_id_mean']:.2f}")
    print(f"      平均売上: {row['target_mean']:.2f}円")
    print(f"      中央値: {row['target_median']:.2f}円")

# 可視化
print("\n  クラスタリング結果を可視化しています...")

colors = px.colors.qualitative.Set1[:optimal_k]

# ホバー情報を作成
hover_data_list = []
for _, row in df_analysis.iterrows():
    hover_text = (
        f"<b>店舗: {row.get('RST_TITLE', 'N/A')}</b><br>"
        f"料理カテゴリ: {row.get('CUISINE_CAT_origin', 'N/A')}<br>"
        f"クラスタ: {int(row['store_cluster'])}<br>"
        f"---<br>"
        f"月間人口: {row['AVG_MONTHLY_POPULATION']:.0f}<br>"
        f"座席数: {row['NUM_SEATS']:.0f}<br>"
        f"ディナー価格: {row['DINNER_PRICE']:.0f}円<br>"
        f"ファミリー向け: {'はい' if row['IS_FAMILY_FRIENDLY'] == 1 else 'いいえ'}<br>"
        f"料理クラスタ: {int(row['cuisine_cluster_id'])}<br>"
        f"売上: {row['target_amount_tableau']:.2f}円"
    )
    hover_data_list.append(hover_text)

df_analysis['hover_text'] = hover_data_list

# 散布図1: 座席数 vs ディナー価格
fig1 = go.Figure()

for cluster_id in range(optimal_k):
    cluster_data = df_analysis[df_analysis['store_cluster'] == cluster_id]
    
    fig1.add_trace(go.Scatter(
        x=cluster_data['NUM_SEATS'],
        y=cluster_data['DINNER_PRICE'],
        mode='markers',
        name=f'クラスタ {cluster_id}',
        marker=dict(
            size=5,
            color=colors[cluster_id],
            opacity=0.6,
            line=dict(width=0.5, color='white')
        ),
        hovertext=cluster_data['hover_text'],
        hoverinfo='text'
    ))

# クラスタ中心を追加
centers_sorted = centers_df.sort_values('new_cluster')
fig1.add_trace(go.Scatter(
    x=centers_sorted['NUM_SEATS'],
    y=centers_sorted['DINNER_PRICE'],
    mode='markers',
    name='クラスタ中心',
    marker=dict(
        size=20,
        color='black',
        symbol='star',
        line=dict(width=2, color='white')
    ),
    hovertemplate='クラスタ中心<br>座席数: %{x:.1f}<br>ディナー価格: %{y:.0f}円<extra></extra>'
))

fig1.update_layout(
    title=f'店舗クラスタリング結果: 座席数 vs ディナー価格 (k={optimal_k})',
    xaxis_title='座席数',
    yaxis_title='ディナー価格（円）',
    height=700,
    width=1200,
    hovermode='closest',
    showlegend=True
)

fig1.write_html(os.path.join(results_folder, 'cluster_scatter_seats_price.html'))
print(f"  散布図1を保存: cluster_scatter_seats_price.html")

# 散布図2: 月間人口 vs 売上
fig2 = go.Figure()

for cluster_id in range(optimal_k):
    cluster_data = df_analysis[df_analysis['store_cluster'] == cluster_id]
    
    fig2.add_trace(go.Scatter(
        x=cluster_data['AVG_MONTHLY_POPULATION'],
        y=cluster_data['target_amount_tableau'],
        mode='markers',
        name=f'クラスタ {cluster_id}',
        marker=dict(
            size=5,
            color=colors[cluster_id],
            opacity=0.6,
            line=dict(width=0.5, color='white')
        ),
        hovertext=cluster_data['hover_text'],
        hoverinfo='text'
    ))

fig2.update_layout(
    title=f'店舗クラスタリング結果: 月間人口 vs 売上 (k={optimal_k})',
    xaxis_title='月間人口',
    yaxis_title='売上（円）',
    height=700,
    width=1200,
    hovermode='closest',
    showlegend=True
)

fig2.write_html(os.path.join(results_folder, 'cluster_scatter_population_sales.html'))
print(f"  散布図2を保存: cluster_scatter_population_sales.html")

# 散布図3: 料理クラスタ vs 売上
fig3 = go.Figure()

for cluster_id in range(optimal_k):
    cluster_data = df_analysis[df_analysis['store_cluster'] == cluster_id]
    
    fig3.add_trace(go.Scatter(
        x=cluster_data['cuisine_cluster_id'],
        y=cluster_data['target_amount_tableau'],
        mode='markers',
        name=f'クラスタ {cluster_id}',
        marker=dict(
            size=5,
            color=colors[cluster_id],
            opacity=0.6,
            line=dict(width=0.5, color='white')
        ),
        hovertext=cluster_data['hover_text'],
        hoverinfo='text'
    ))

fig3.update_layout(
    title=f'店舗クラスタリング結果: 料理クラスタ vs 売上 (k={optimal_k})',
    xaxis_title='料理クラスタID',
    yaxis_title='売上（円）',
    height=700,
    width=1200,
    hovermode='closest',
    showlegend=True
)

fig3.write_html(os.path.join(results_folder, 'cluster_scatter_cuisine_sales.html'))
print(f"  散布図3を保存: cluster_scatter_cuisine_sales.html")

# 結果を保存
output_file = os.path.join(results_folder, 'stores_with_cluster.csv')
# 保存用にstore_cluster_old列を削除
df_save = df_analysis.drop(columns=['store_cluster_old', 'hover_text'])
df_save.to_csv(output_file, encoding='utf-8-sig', index=False)
print(f"\n  クラスタリング結果を保存: stores_with_cluster.csv")

# クラスタ中心も保存
centers_file = os.path.join(results_folder, 'cluster_centers.csv')
centers_df.to_csv(centers_file, encoding='utf-8-sig', index=False)
print(f"  クラスタ中心を保存: cluster_centers.csv")

# クラスタマッピングを保存
mapping_df = pd.DataFrame([
    {
        '旧クラスタID': old_id,
        '新クラスタID': new_id,
        '中央値': cluster_medians[old_id]
    }
    for old_id, new_id in old_to_new_mapping.items()
]).sort_values('新クラスタID')

mapping_file = os.path.join(results_folder, 'cluster_name_mapping.csv')
mapping_df.to_csv(mapping_file, encoding='utf-8-sig', index=False)
print(f"  クラスタマッピングを保存: cluster_name_mapping.csv")

print("\n" + "=" * 100)
print("【完了】")
print("=" * 100)
print(f"\n生成されたファイル（{results_folder}フォルダ）:")
print("  1. optimal_k_analysis.html - エルボー法とシルエット分析")
print("  2. cluster_scatter_seats_price.html - 散布図（座席数 vs ディナー価格）")
print("  3. cluster_scatter_population_sales.html - 散布図（月間人口 vs 売上）")
print("  4. cluster_scatter_cuisine_sales.html - 散布図（料理クラスタ vs 売上）")
print("  5. cluster_statistics.csv - クラスタ別統計表（CSV形式）")
print("  6. cluster_statistics.xlsx - クラスタ別統計表（Excel形式）")
print("  7. cluster_summary.csv - クラスタ別サマリー")
print("  8. stores_with_cluster.csv - 全店舗のクラスタ割当結果")
print("  9. cluster_centers.csv - クラスタ中心の座標")
print(" 10. cluster_name_mapping.csv - クラスタID変換マッピング")

print("\n※ クラスタは売上中央値の昇順で並び替えられています")
print("※ インプットデータ: aggregated_df_filtered.csv (target_amount_tableau: 20~1000)")
print("\n" + "=" * 100)

