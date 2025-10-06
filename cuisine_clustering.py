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

# 結果フォルダを作成
results_folder = 'cuisine_clustering_results'
os.makedirs(results_folder, exist_ok=True)

print("=" * 100)
print("料理カテゴリのクラスタリング分析")
print("=" * 100)

print("\n[1/5] データを読み込んでいます...")
print("  ファイル: aggregated_df_filtered.csv")
df_analysis = pd.read_csv('aggregated_df_filtered.csv')
print(f"  データ形状: {df_analysis.shape}")
print(f"  ※ target_amount_tableau: 20~1000 でフィルタリング済み")

print("\n[2/5] 10店舗以下のカテゴリを'other'にまとめています...")
# 10店舗以下のカテゴリを"other"にまとめる
cuisine_counts = df_analysis['CUISINE_CAT_origin'].value_counts()
small_categories = cuisine_counts[cuisine_counts <= 10].index
df_analysis['CUISINE_CAT_processed'] = df_analysis['CUISINE_CAT_origin'].copy()
df_analysis.loc[df_analysis['CUISINE_CAT_origin'].isin(small_categories), 'CUISINE_CAT_processed'] = 'other'

print(f"  10店舗以下のカテゴリ数: {len(small_categories)}")
print(f"  'other'にまとめられた店舗数: {len(df_analysis[df_analysis['CUISINE_CAT_processed'] == 'other'])}")

cuisine_stats = df_analysis.groupby('CUISINE_CAT_processed')['target_amount_tableau'].agg([
    'count', 'mean', 'std', 'min', 'max', 'median'
]).reset_index()
cuisine_stats.rename(columns={'CUISINE_CAT_processed': 'CUISINE_CAT_origin'}, inplace=True)

cuisine_stats = cuisine_stats[cuisine_stats['count'] >= 10]
print(f"  処理後の料理カテゴリ数: {len(cuisine_stats)}")

print("\n[3/5] 特徴量を標準化しています...")
X = cuisine_stats[['mean', 'std']].values
print(f"  特徴量: {X.shape}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n[4/5] 最適なクラスタ数を探索しています（エルボー法・シルエット分析）...")
inertias = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

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
fig.write_html(os.path.join(results_folder, 'cuisine_clustering_analysis.html'))
print(f"  グラフ保存: cuisine_clustering_analysis.html")

optimal_k = K_range[np.argmax(silhouette_scores)]
print(f"\n  最適なクラスタ数: {optimal_k} (シルエット係数: {max(silhouette_scores):.3f})")

print(f"\n[5/5] k={optimal_k}でクラスタリングを実行しています...")
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = kmeans_final.fit_predict(X_scaled)

cuisine_stats['cluster_old'] = cluster_labels

# 中央値でクラスタを並び替え
cluster_medians = cuisine_stats.groupby('cluster_old')['median'].median().sort_values()
old_to_new_mapping = {old_id: new_id for new_id, old_id in enumerate(cluster_medians.index)}
cuisine_stats['cluster'] = cuisine_stats['cluster_old'].map(old_to_new_mapping)

print(f"  中央値昇順でクラスタIDを再割り当てしました")

# クラスタの中心を計算
centers = kmeans_final.cluster_centers_
centers_original = scaler.inverse_transform(centers)

# Plotlyで散布図を作成
fig = go.Figure()

# 各クラスタごとに散布図を作成
colors = px.colors.qualitative.Set1[:optimal_k]
for cluster_id in range(optimal_k):
    cluster_data = cuisine_stats[cuisine_stats['cluster'] == cluster_id]
    
    # ホバー情報を作成
    hover_text = []
    for _, row in cluster_data.iterrows():
        text = (
            f"<b>{row['CUISINE_CAT_origin']}</b><br>"
            f"クラスタ: {cluster_id}<br>"
            f"店舗数: {int(row['count'])}<br>"
            f"平均: {row['mean']:.2f}円<br>"
            f"標準偏差: {row['std']:.2f}円<br>"
            f"中央値: {row['median']:.2f}円<br>"
            f"最小: {row['min']:.2f}円<br>"
            f"最大: {row['max']:.2f}円"
        )
        hover_text.append(text)
    
    fig.add_trace(go.Scatter(
        x=cluster_data['mean'],
        y=cluster_data['std'],
        mode='markers',
        name=f'クラスタ {cluster_id}',
        marker=dict(
            size=cluster_data['count'] / 30,  # 店舗数に比例したサイズ
            color=colors[cluster_id],
            opacity=0.7,
            line=dict(width=1, color='white')
        ),
        text=cluster_data['CUISINE_CAT_origin'],
        hovertext=hover_text,
        hoverinfo='text'
    ))

# クラスタ中心を追加
fig.add_trace(go.Scatter(
    x=centers_original[:, 0],
    y=centers_original[:, 1],
    mode='markers',
    name='クラスタ中心',
    marker=dict(
        size=20,
        color='black',
        symbol='star',
        line=dict(width=2, color='white')
    ),
    hovertemplate='クラスタ中心<br>平均: %{x:.2f}円<br>標準偏差: %{y:.2f}円<extra></extra>'
))

fig.update_layout(
    title=f'料理カテゴリのクラスタリング結果 (k={optimal_k})',
    xaxis_title='target_amount_tableau 平均 (円)',
    yaxis_title='target_amount_tableau 標準偏差 (円)',
    height=800,
    width=1200,
    hovermode='closest',
    showlegend=True
)

fig.write_html(os.path.join(results_folder, 'cuisine_clustering_result.html'))
print(f"  グラフ保存: cuisine_clustering_result.html")

# クラスタ情報をdf_analysisにマージ
print("\n" + "=" * 100)
print("【クラスタごとの詳細統計を計算中】")
print("=" * 100)

# cuisine_stats の cluster 情報を df_analysis にマージ
cuisine_to_cluster = dict(zip(cuisine_stats['CUISINE_CAT_origin'], cuisine_stats['cluster']))
df_analysis['cuisine_cluster_id'] = df_analysis['CUISINE_CAT_processed'].map(cuisine_to_cluster)

# クラスタ別の統計を計算
cluster_detailed_stats = df_analysis.groupby('cuisine_cluster_id')['target_amount_tableau'].agg([
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
cluster_detailed_stats['cuisine_cluster_id'] = cluster_detailed_stats['cuisine_cluster_id'].apply(
    lambda x: f'クラスタ {int(x)}'
)

print("\n【クラスタ別統計表（target_amount_tableau: 20~1000）】")
print("=" * 120)
print(cluster_detailed_stats.to_string(index=False, float_format="%.2f"))

# 統計表を保存
cluster_stats_csv = os.path.join(results_folder, 'cuisine_cluster_statistics.csv')
cluster_detailed_stats.to_csv(cluster_stats_csv, encoding='utf-8-sig', index=False)

cluster_stats_excel = os.path.join(results_folder, 'cuisine_cluster_statistics.xlsx')
with pd.ExcelWriter(cluster_stats_excel, engine='openpyxl') as writer:
    cluster_detailed_stats.to_excel(writer, sheet_name='クラスタ統計', index=False)
    
    # 列幅を調整
    worksheet = writer.sheets['クラスタ統計']
    worksheet.column_dimensions['A'].width = 20
    for col in ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M']:
        worksheet.column_dimensions[col].width = 15

print(f"\n統計表を保存しました:")
print(f"  - cuisine_cluster_statistics.csv")
print(f"  - cuisine_cluster_statistics.xlsx")

print("\n" + "=" * 100)
print("【各クラスタの料理カテゴリ】")
print("=" * 100)
for cluster_id in sorted(cuisine_stats['cluster'].unique()):
    cluster_cuisines = cuisine_stats[cuisine_stats['cluster'] == cluster_id]
    cluster_stores = df_analysis[df_analysis['cuisine_cluster_id'] == cluster_id]
    
    print(f"\n【クラスタ {cluster_id}】")
    print(f"カテゴリ数: {len(cluster_cuisines)}")
    print(f"店舗数: {len(cluster_stores)}")
    print(f"売上平均: {cluster_stores['target_amount_tableau'].mean():.2f}円")
    print(f"売上中央値: {cluster_stores['target_amount_tableau'].median():.2f}円")
    print(f"主要カテゴリ（上位5位）:")
    top_cuisines = cluster_cuisines.nlargest(5, 'count')
    for idx, row in top_cuisines.iterrows():
        print(f"  {row['CUISINE_CAT_origin']}: {int(row['count'])}店舗 (平均{row['mean']:.1f})")

# 保存用にcluster_old列を削除
cuisine_stats_save = cuisine_stats.drop(columns=['cluster_old'])
cuisine_stats_save.to_csv(os.path.join(results_folder, 'cuisine_clustering_results.csv'), encoding='utf-8-sig', index=False)

# クラスタマッピングを保存
mapping_df = pd.DataFrame([
    {
        '旧クラスタID': old_id,
        '新クラスタID': new_id,
        '中央値': cluster_medians[old_id]
    }
    for old_id, new_id in old_to_new_mapping.items()
]).sort_values('新クラスタID')
mapping_df.to_csv(os.path.join(results_folder, 'cluster_name_mapping.csv'), encoding='utf-8-sig', index=False)

print("\n" + "=" * 100)
print("【完了】")
print("=" * 100)
print(f"\n生成されたファイル（{results_folder}フォルダ）:")
print("  1. cuisine_clustering_analysis.html - エルボー法とシルエット分析")
print("  2. cuisine_clustering_result.html - クラスタリング散布図（ホバー情報付き）")
print("  3. cuisine_clustering_results.csv - 全カテゴリの詳細データ")
print("  4. cuisine_cluster_statistics.csv - クラスタ別統計表（CSV形式）")
print("  5. cuisine_cluster_statistics.xlsx - クラスタ別統計表（Excel形式）")
print("  6. cluster_name_mapping.csv - クラスタID変換マッピング")
print("\n※ インプットデータ: aggregated_df_filtered.csv (target_amount_tableau: 20~1000)")
print("※ クラスタは中央値の昇順で並び替えられています")
print("\n" + "=" * 100)
