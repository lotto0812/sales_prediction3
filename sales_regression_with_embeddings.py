# -*- coding: utf-8 -*-
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import lightgbm as lgb
import optuna
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import os
warnings.filterwarnings('ignore')

print("=" * 100)
print("売上回帰問題（対数変換版 + CUISINE_CAT_origin & CITY Embedding）")
print("=" * 100)

# 結果フォルダを作成
results_folder = 'sales_regression_results_with_embeddings'
os.makedirs(results_folder, exist_ok=True)
print(f"\n結果フォルダ: {results_folder}")

# データを読み込む
print("\n[1/9] データを読み込んでいます...")
stores_df = pd.read_csv('final_store_clustering_results/stores_with_cluster.csv')
agg_df = pd.read_csv('aggregated_df_filtered.csv')

# CUISINE_CAT_originとCITYを追加
df = stores_df.copy()
df['CUISINE_CAT_origin'] = agg_df['CUISINE_CAT_origin'].values
df['CITY'] = agg_df['CITY'].values

print(f"  データ形状: {df.shape}")
print(f"  CUISINE_CAT_originのユニーク数: {df['CUISINE_CAT_origin'].nunique()}")
print(f"  CITYのユニーク数: {df['CITY'].nunique()}")

# 特徴量を選択
feature_cols = [
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
    'CUISINE_CAT_1_bread and desert',
    'CUISINE_CAT_1_cafe and coffee shop',
    'CUISINE_CAT_1_cafeteria',
    'CUISINE_CAT_1_chinese cuisine',
    'CUISINE_CAT_1_convenience store',
    'CUISINE_CAT_1_family restaurant',
    'CUISINE_CAT_1_fastfood light meal',
    'CUISINE_CAT_1_foreign ethnic cuisine',
    'CUISINE_CAT_1_hotel and ryokan',
    'CUISINE_CAT_1_italian cuisine',
    'CUISINE_CAT_1_izakaya',
    'CUISINE_CAT_1_japanese cuisine',
    'CUISINE_CAT_1_noodles',
    'CUISINE_CAT_1_other',
    'CUISINE_CAT_1_restaurant',
    'rate_count',
    'seats_rate_count',
    'cuisine_cluster_id',
    'store_cluster'
]

print("\n[2/9] 特徴量を選択しています...")
print(f"  基本特徴量数: {len(feature_cols)}")

# 欠損値の確認
missing_cols = []
for col in feature_cols:
    if col not in df.columns:
        missing_cols.append(col)

if missing_cols:
    print(f"\n  警告: 以下の列がデータに存在しません: {missing_cols}")
    feature_cols = [col for col in feature_cols if col in df.columns]
    print(f"  使用可能な特徴量数: {len(feature_cols)}")

# 欠損値を含む行を削除
df_clean = df.dropna(subset=feature_cols + ['target_amount_tableau', 'CUISINE_CAT_origin', 'CITY']).copy()
print(f"  欠損値を含む行を削除: {len(df_clean)}店舗")

# 目的変数の対数変換
print("\n[3/9] 目的変数を対数変換しています...")
df_clean['target_log'] = np.log(df_clean['target_amount_tableau'])
print(f"  元のtarget_amount_tableau範囲: {df_clean['target_amount_tableau'].min():.2f} ~ {df_clean['target_amount_tableau'].max():.2f}")
print(f"  対数変換後の範囲: {df_clean['target_log'].min():.2f} ~ {df_clean['target_log'].max():.2f}")

# 特徴量とラベルを分離
X = df_clean[feature_cols].values
y_log = df_clean['target_log'].values
y_original = df_clean['target_amount_tableau'].values

print(f"\n  X shape: {X.shape}")
print(f"  y_log shape: {y_log.shape}")

# データ分割
print("\n[4/9] データを分割しています...")
X_train, X_test, y_train_log, y_test_log, y_train_original, y_test_original, train_idx, test_idx = train_test_split(
    X, y_log, y_original, np.arange(len(X)), test_size=0.2, random_state=42
)
print(f"  Train: {X_train.shape[0]}店舗")
print(f"  Test: {X_test.shape[0]}店舗")

# 訓練データとテストデータのDataFrameを取得
train_df = df_clean.iloc[train_idx].copy()
test_df = df_clean.iloc[test_idx].copy()

# Target Encodingを追加（Data leakage防止のため訓練データのみで計算）
print("\n[5/9] カテゴリ変数のTarget Encodingを計算しています...")

# cuisine_cluster_id ごとの中央値（対数空間）
cuisine_median_map = train_df.groupby('cuisine_cluster_id')['target_log'].median().to_dict()
print(f"\n  料理クラスタごとの中央値（対数空間）:")
for cid, median in sorted(cuisine_median_map.items()):
    print(f"    クラスタ {int(cid)}: {median:.3f} (元: {np.exp(median):.2f}円)")

# store_cluster ごとの中央値（対数空間）
store_median_map = train_df.groupby('store_cluster')['target_log'].median().to_dict()
print(f"\n  店舗クラスタごとの中央値（対数空間）:")
for sid, median in sorted(store_median_map.items()):
    print(f"    クラスタ {int(sid)}: {median:.3f} (元: {np.exp(median):.2f}円)")

# CUISINE_CAT_origin ごとの中央値（対数空間）- 新規追加
cuisine_origin_median_map = train_df.groupby('CUISINE_CAT_origin')['target_log'].median().to_dict()
cuisine_origin_count_map = train_df['CUISINE_CAT_origin'].value_counts().to_dict()
print(f"\n  料理カテゴリ（origin）ごとの中央値（上位10件）:")
top_cuisines = sorted(cuisine_origin_median_map.items(), key=lambda x: cuisine_origin_count_map.get(x[0], 0), reverse=True)[:10]
for cuisine, median in top_cuisines:
    count = cuisine_origin_count_map.get(cuisine, 0)
    print(f"    {cuisine} (n={count}): {median:.3f} (元: {np.exp(median):.2f}円)")

# CITY ごとの中央値（対数空間）- 新規追加
city_median_map = train_df.groupby('CITY')['target_log'].median().to_dict()
city_count_map = train_df['CITY'].value_counts().to_dict()
print(f"\n  都市ごとの中央値（上位10件）:")
top_cities = sorted(city_median_map.items(), key=lambda x: city_count_map.get(x[0], 0), reverse=True)[:10]
for city, median in top_cities:
    count = city_count_map.get(city, 0)
    print(f"    {city} (n={count}): {median:.3f} (元: {np.exp(median):.2f}円)")

# 訓練データに適用
cuisine_cluster_median_train = train_df['cuisine_cluster_id'].map(cuisine_median_map).fillna(train_df['target_log'].median()).values
store_cluster_median_train = train_df['store_cluster'].map(store_median_map).fillna(train_df['target_log'].median()).values
cuisine_origin_median_train = train_df['CUISINE_CAT_origin'].map(cuisine_origin_median_map).fillna(train_df['target_log'].median()).values
city_median_train = train_df['CITY'].map(city_median_map).fillna(train_df['target_log'].median()).values

# Frequency Encodingも追加
cuisine_origin_freq_train = train_df['CUISINE_CAT_origin'].map(cuisine_origin_count_map).fillna(1).values
city_freq_train = train_df['CITY'].map(city_count_map).fillna(1).values

# テストデータに適用
cuisine_cluster_median_test = test_df['cuisine_cluster_id'].map(cuisine_median_map).fillna(train_df['target_log'].median()).values
store_cluster_median_test = test_df['store_cluster'].map(store_median_map).fillna(train_df['target_log'].median()).values
cuisine_origin_median_test = test_df['CUISINE_CAT_origin'].map(cuisine_origin_median_map).fillna(train_df['target_log'].median()).values
city_median_test = test_df['CITY'].map(city_median_map).fillna(train_df['target_log'].median()).values
cuisine_origin_freq_test = test_df['CUISINE_CAT_origin'].map(cuisine_origin_count_map).fillna(1).values
city_freq_test = test_df['CITY'].map(city_count_map).fillna(1).values

# 新しい特徴量を追加
X_train_with_encodings = np.column_stack([
    X_train, 
    cuisine_cluster_median_train, 
    store_cluster_median_train,
    cuisine_origin_median_train,
    city_median_train,
    cuisine_origin_freq_train,
    city_freq_train
])
X_test_with_encodings = np.column_stack([
    X_test, 
    cuisine_cluster_median_test, 
    store_cluster_median_test,
    cuisine_origin_median_test,
    city_median_test,
    cuisine_origin_freq_test,
    city_freq_test
])

print(f"\n[6/9] Target Encoding & Frequency Encodingを追加しました")
print(f"  追加した特徴量:")
print(f"    - cuisine_cluster_median_log (Target Encoding)")
print(f"    - store_cluster_median_log (Target Encoding)")
print(f"    - cuisine_origin_median_log (Target Encoding) ← 新規")
print(f"    - city_median_log (Target Encoding) ← 新規")
print(f"    - cuisine_origin_frequency (Frequency Encoding) ← 新規")
print(f"    - city_frequency (Frequency Encoding) ← 新規")
print(f"\n  特徴量追加後のshape:")
print(f"    Train: {X_train_with_encodings.shape}")
print(f"    Test: {X_test_with_encodings.shape}")

# 特徴量名リストを更新
feature_names = feature_cols + [
    'cuisine_cluster_median_log', 
    'store_cluster_median_log',
    'cuisine_origin_median_log',
    'city_median_log',
    'cuisine_origin_frequency',
    'city_frequency'
]

# 標準化
print("\n[7/9] 特徴量を標準化しています...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_with_encodings)
X_test_scaled = scaler.transform(X_test_with_encodings)

# MAPE計算関数
def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Optunaでハイパーパラメータ最適化（LightGBM）
print("\n[8/9] LightGBMのハイパーパラメータを最適化しています...")

def objective_lgb(trial):
    param = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    }
    
    train_data = lgb.Dataset(X_train_scaled, label=y_train_log)
    
    cv_results = lgb.cv(
        param,
        train_data,
        num_boost_round=1000,
        nfold=5,
        stratified=False,
        callbacks=[lgb.early_stopping(stopping_rounds=50)],
        seed=42
    )
    
    return cv_results['valid rmse-mean'][-1]

study_lgb = optuna.create_study(direction='minimize', study_name='lgbm_regression_with_embeddings')
study_lgb.optimize(objective_lgb, n_trials=30, show_progress_bar=True)

print(f"\n  LightGBM最適パラメータ:")
for key, value in study_lgb.best_params.items():
    print(f"    {key}: {value}")
print(f"  Best CV RMSE: {study_lgb.best_value:.4f}")

# LightGBMモデルを訓練
print("\n  LightGBMモデルを訓練しています...")
best_params_lgb = study_lgb.best_params
best_params_lgb['objective'] = 'regression'
best_params_lgb['metric'] = 'rmse'
best_params_lgb['verbosity'] = -1

train_data = lgb.Dataset(X_train_scaled, label=y_train_log)
test_data = lgb.Dataset(X_test_scaled, label=y_test_log, reference=train_data)

model_lgb = lgb.train(
    best_params_lgb,
    train_data,
    num_boost_round=1000,
    valid_sets=[train_data, test_data],
    valid_names=['train', 'test'],
    callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=100)]
)

# Random Forestのハイパーパラメータ最適化
print("\n  Random Forestのハイパーパラメータを最適化しています...")

def objective_rf(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 5, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'random_state': 42,
        'n_jobs': -1
    }
    
    model = RandomForestRegressor(**param)
    model.fit(X_train_scaled, y_train_log)
    y_pred = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test_log, y_pred))
    
    return rmse

study_rf = optuna.create_study(direction='minimize', study_name='rf_regression_with_embeddings')
study_rf.optimize(objective_rf, n_trials=20, show_progress_bar=True)

print(f"\n  Random Forest最適パラメータ:")
for key, value in study_rf.best_params.items():
    print(f"    {key}: {value}")
print(f"  Best Test RMSE: {study_rf.best_value:.4f}")

# Random Forestモデルを訓練
print("\n  Random Forestモデルを訓練しています...")
best_params_rf = study_rf.best_params
best_params_rf['random_state'] = 42
best_params_rf['n_jobs'] = -1

model_rf = RandomForestRegressor(**best_params_rf)
model_rf.fit(X_train_scaled, y_train_log)

# 予測（対数空間）
print("\n[9/9] 予測と評価を実行しています...")

# LightGBM予測
lgb_pred_train_log = model_lgb.predict(X_train_scaled, num_iteration=model_lgb.best_iteration)
lgb_pred_test_log = model_lgb.predict(X_test_scaled, num_iteration=model_lgb.best_iteration)

# Random Forest予測
rf_pred_train_log = model_rf.predict(X_train_scaled)
rf_pred_test_log = model_rf.predict(X_test_scaled)

# 指数変換して元のスケールに戻す
lgb_pred_train = np.exp(lgb_pred_train_log)
lgb_pred_test = np.exp(lgb_pred_test_log)
rf_pred_train = np.exp(rf_pred_train_log)
rf_pred_test = np.exp(rf_pred_test_log)

# テストデータの店舗情報を取得（グラフのホバー表示用）
test_info = df_clean.iloc[test_idx].copy()
store_names = test_info['RST_TITLE'].values if 'RST_TITLE' in test_info.columns else ['N/A'] * len(test_idx)
cities = test_info['CITY'].values if 'CITY' in test_info.columns else ['N/A'] * len(test_idx)
cuisine_types = test_info['CUISINE_CAT_origin'].values if 'CUISINE_CAT_origin' in test_info.columns else ['N/A'] * len(test_idx)
store_clusters = test_info['store_cluster'].values
cuisine_clusters = test_info['cuisine_cluster_id'].values
num_seats = test_info['NUM_SEATS'].values if 'NUM_SEATS' in test_info.columns else [0] * len(test_idx)
avg_population = test_info['AVG_MONTHLY_POPULATION'].values if 'AVG_MONTHLY_POPULATION' in test_info.columns else [0] * len(test_idx)
dinner_price = test_info['DINNER_PRICE'].values if 'DINNER_PRICE' in test_info.columns else [0] * len(test_idx)
lunch_price = test_info['LUNCH_PRICE'].values if 'LUNCH_PRICE' in test_info.columns else [0] * len(test_idx)
is_family = test_info['IS_FAMILY_FRIENDLY'].values if 'IS_FAMILY_FRIENDLY' in test_info.columns else [0] * len(test_idx)
is_friend = test_info['IS_FRIEND_FRIENDLY'].values if 'IS_FRIEND_FRIENDLY' in test_info.columns else [0] * len(test_idx)
is_alone = test_info['IS_ALONE_FRIENDLY'].values if 'IS_ALONE_FRIENDLY' in test_info.columns else [0] * len(test_idx)

# 評価指標を計算
def evaluate_model(y_true, y_pred, model_name, dataset_name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = calculate_mape(y_true, y_pred)
    
    print(f"\n  {model_name} - {dataset_name}:")
    print(f"    RMSE: {rmse:.2f}")
    print(f"    R²: {r2:.4f}")
    print(f"    MAPE: {mape:.2f}%")
    
    return {'RMSE': rmse, 'R2': r2, 'MAPE': mape}

# 評価
print("\n" + "=" * 100)
print("【LightGBM 評価結果（Embedding追加版）】")
print("=" * 100)

lgb_train_metrics = evaluate_model(y_train_original, lgb_pred_train, "LightGBM", "Train")
lgb_test_metrics = evaluate_model(y_test_original, lgb_pred_test, "LightGBM", "Test")

# 全データで評価
X_all_with_encodings = np.column_stack([
    X,
    df_clean['cuisine_cluster_id'].map(cuisine_median_map).fillna(df_clean['target_log'].median()).values,
    df_clean['store_cluster'].map(store_median_map).fillna(df_clean['target_log'].median()).values,
    df_clean['CUISINE_CAT_origin'].map(cuisine_origin_median_map).fillna(df_clean['target_log'].median()).values,
    df_clean['CITY'].map(city_median_map).fillna(df_clean['target_log'].median()).values,
    df_clean['CUISINE_CAT_origin'].map(cuisine_origin_count_map).fillna(1).values,
    df_clean['CITY'].map(city_count_map).fillna(1).values
])
X_all_scaled = scaler.transform(X_all_with_encodings)
lgb_pred_all_log = model_lgb.predict(X_all_scaled, num_iteration=model_lgb.best_iteration)
lgb_pred_all = np.exp(lgb_pred_all_log)
lgb_all_metrics = evaluate_model(y_original, lgb_pred_all, "LightGBM", "All")

print("\n" + "=" * 100)
print("【Random Forest 評価結果（Embedding追加版）】")
print("=" * 100)

rf_train_metrics = evaluate_model(y_train_original, rf_pred_train, "Random Forest", "Train")
rf_test_metrics = evaluate_model(y_test_original, rf_pred_test, "Random Forest", "Test")

# 全データでRF評価
rf_pred_all_log = model_rf.predict(X_all_scaled)
rf_pred_all = np.exp(rf_pred_all_log)
rf_all_metrics = evaluate_model(y_original, rf_pred_all, "Random Forest", "All")

# 結果をCSVに保存
results_summary = pd.DataFrame({
    'Model': ['LightGBM', 'LightGBM', 'LightGBM', 'RandomForest', 'RandomForest', 'RandomForest'],
    'Dataset': ['Train', 'Test', 'All', 'Train', 'Test', 'All'],
    'RMSE': [
        lgb_train_metrics['RMSE'], lgb_test_metrics['RMSE'], lgb_all_metrics['RMSE'],
        rf_train_metrics['RMSE'], rf_test_metrics['RMSE'], rf_all_metrics['RMSE']
    ],
    'R2': [
        lgb_train_metrics['R2'], lgb_test_metrics['R2'], lgb_all_metrics['R2'],
        rf_train_metrics['R2'], rf_test_metrics['R2'], rf_all_metrics['R2']
    ],
    'MAPE': [
        lgb_train_metrics['MAPE'], lgb_test_metrics['MAPE'], lgb_all_metrics['MAPE'],
        rf_train_metrics['MAPE'], rf_test_metrics['MAPE'], rf_all_metrics['MAPE']
    ]
})

results_summary.to_csv(os.path.join(results_folder, 'model_comparison.csv'), index=False, encoding='utf-8-sig')
print("\n  モデル比較結果を保存しました: model_comparison.csv")

# 予測結果を保存
test_results = pd.DataFrame({
    'actual': y_test_original,
    'lgb_predicted': lgb_pred_test,
    'rf_predicted': rf_pred_test,
    'lgb_error': y_test_original - lgb_pred_test,
    'rf_error': y_test_original - rf_pred_test,
    'lgb_ape': np.abs((y_test_original - lgb_pred_test) / y_test_original) * 100,
    'rf_ape': np.abs((y_test_original - rf_pred_test) / y_test_original) * 100
})
test_results.to_csv(os.path.join(results_folder, 'test_predictions.csv'), index=False, encoding='utf-8-sig')

# 可視化（2x2サブプロット）
print("\n  グラフを作成しています...")

# LightGBMのグラフ
fig_lgb = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        f'実測値 vs 予測値 (LightGBM)<br>RMSE={lgb_test_metrics["RMSE"]:.2f}, R²={lgb_test_metrics["R2"]:.3f}, MAPE={lgb_test_metrics["MAPE"]:.2f}%',
        '実測値と予測値のヒストグラム (LightGBM)',
        '実測値 vs 絶対パーセント誤差 (LightGBM)',
        '特徴量重要度 Top 25 (LightGBM)'
    ),
    specs=[[{'type': 'scatter'}, {'type': 'bar'}],
           [{'type': 'scatter'}, {'type': 'bar'}]]
)

# 1. 実測値 vs 予測値
fig_lgb.add_trace(
    go.Scatter(
        x=y_test_original,
        y=lgb_pred_test,
        mode='markers',
        marker=dict(size=5, color='blue', opacity=0.6),
        name='Test Data',
        customdata=np.column_stack([store_names, cities, cuisine_types, store_clusters, cuisine_clusters, 
                                     num_seats, avg_population, dinner_price, lunch_price, 
                                     is_family, is_friend, is_alone]),
        hovertemplate='<b>RST_TITLE:</b> %{customdata[0]}<br>' +
                      '<b>CITY:</b> %{customdata[1]}<br>' +
                      '<b>CUISINE_CAT_origin:</b> %{customdata[2]}<br>' +
                      '<b>store_cluster:</b> %{customdata[3]}<br>' +
                      '<b>cuisine_cluster_id:</b> %{customdata[4]}<br>' +
                      '<b>NUM_SEATS:</b> %{customdata[5]:.0f}<br>' +
                      '<b>AVG_MONTHLY_POPULATION:</b> %{customdata[6]:,.0f}<br>' +
                      '<b>DINNER_PRICE:</b> %{customdata[7]:,.0f}円<br>' +
                      '<b>LUNCH_PRICE:</b> %{customdata[8]:,.0f}円<br>' +
                      '<b>IS_FAMILY_FRIENDLY:</b> %{customdata[9]:.0f}<br>' +
                      '<b>IS_FRIEND_FRIENDLY:</b> %{customdata[10]:.0f}<br>' +
                      '<b>IS_ALONE_FRIENDLY:</b> %{customdata[11]:.0f}<br>' +
                      '<b>actual:</b> %{x:,.1f}本/月<br>' +
                      '<b>predicted:</b> %{y:,.1f}本/月<br>' +
                      '<b>error:</b> %{text:,.1f}本/月<extra></extra>',
        text=y_test_original - lgb_pred_test
    ),
    row=1, col=1
)
# 45度線
max_val = max(y_test_original.max(), lgb_pred_test.max())
fig_lgb.add_trace(
    go.Scatter(
        x=[0, max_val],
        y=[0, max_val],
        mode='lines',
        line=dict(color='red', dash='dash'),
        name='Perfect Prediction',
        showlegend=False
    ),
    row=1, col=1
)

# 2. ヒストグラム
fig_lgb.add_trace(
    go.Histogram(
        x=y_test_original,
        name='実測値',
        marker=dict(color='blue', opacity=0.5),
        nbinsx=50
    ),
    row=1, col=2
)
fig_lgb.add_trace(
    go.Histogram(
        x=lgb_pred_test,
        name='予測値',
        marker=dict(color='red', opacity=0.5),
        nbinsx=50
    ),
    row=1, col=2
)

# 3. 実測値 vs APE
ape = np.abs((y_test_original - lgb_pred_test) / y_test_original) * 100
fig_lgb.add_trace(
    go.Scatter(
        x=y_test_original,
        y=ape,
        mode='markers',
        marker=dict(size=5, color='green', opacity=0.6),
        name='APE',
        customdata=np.column_stack([store_names, cities, cuisine_types, store_clusters, cuisine_clusters, 
                                     num_seats, avg_population, dinner_price, lunch_price, 
                                     is_family, is_friend, is_alone]),
        hovertemplate='<b>RST_TITLE:</b> %{customdata[0]}<br>' +
                      '<b>CITY:</b> %{customdata[1]}<br>' +
                      '<b>CUISINE_CAT_origin:</b> %{customdata[2]}<br>' +
                      '<b>store_cluster:</b> %{customdata[3]}<br>' +
                      '<b>cuisine_cluster_id:</b> %{customdata[4]}<br>' +
                      '<b>NUM_SEATS:</b> %{customdata[5]:.0f}<br>' +
                      '<b>AVG_MONTHLY_POPULATION:</b> %{customdata[6]:,.0f}<br>' +
                      '<b>DINNER_PRICE:</b> %{customdata[7]:,.0f}円<br>' +
                      '<b>LUNCH_PRICE:</b> %{customdata[8]:,.0f}円<br>' +
                      '<b>IS_FAMILY_FRIENDLY:</b> %{customdata[9]:.0f}<br>' +
                      '<b>IS_FRIEND_FRIENDLY:</b> %{customdata[10]:.0f}<br>' +
                      '<b>IS_ALONE_FRIENDLY:</b> %{customdata[11]:.0f}<br>' +
                      '<b>actual:</b> %{x:,.1f}本/月<br>' +
                      '<b>APE:</b> %{y:.2f}%<extra></extra>'
    ),
    row=2, col=1
)

# 4. 特徴量重要度
feature_importance_lgb = pd.DataFrame({
    'feature': feature_names,
    'importance': model_lgb.feature_importance(importance_type='gain')
}).sort_values('importance', ascending=False).head(25)

# 新規追加した特徴量を色分け
colors = ['red' if 'cuisine_origin' in f or 'city' in f else 'purple' for f in feature_importance_lgb['feature']]

fig_lgb.add_trace(
    go.Bar(
        y=feature_importance_lgb['feature'][::-1],
        x=feature_importance_lgb['importance'][::-1],
        orientation='h',
        marker=dict(color=colors[::-1]),
        name='Importance',
        showlegend=False
    ),
    row=2, col=2
)

# レイアウト更新
fig_lgb.update_xaxes(title_text="実測値 (本/月)", row=1, col=1)
fig_lgb.update_yaxes(title_text="予測値 (本/月)", row=1, col=1)
fig_lgb.update_xaxes(title_text="売上 (本/月)", row=1, col=2)
fig_lgb.update_yaxes(title_text="頻度", row=1, col=2)
fig_lgb.update_xaxes(title_text="実測値 (本/月)", row=2, col=1)
fig_lgb.update_yaxes(title_text="絶対パーセント誤差 (%)", row=2, col=1)
fig_lgb.update_xaxes(title_text="重要度 (Gain)", row=2, col=2)
fig_lgb.update_yaxes(title_text="特徴量", row=2, col=2)

fig_lgb.update_layout(
    height=1000, 
    width=1400, 
    showlegend=True, 
    title_text="LightGBM 回帰結果（CUISINE_CAT_origin & CITY Embedding追加）"
)
fig_lgb.write_html(os.path.join(results_folder, 'lightgbm_analysis.html'))

# Random Forestのグラフ
fig_rf = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        f'実測値 vs 予測値 (Random Forest)<br>RMSE={rf_test_metrics["RMSE"]:.2f}, R²={rf_test_metrics["R2"]:.3f}, MAPE={rf_test_metrics["MAPE"]:.2f}%',
        '実測値と予測値のヒストグラム (Random Forest)',
        '実測値 vs 絶対パーセント誤差 (Random Forest)',
        '特徴量重要度 Top 25 (Random Forest)'
    ),
    specs=[[{'type': 'scatter'}, {'type': 'bar'}],
           [{'type': 'scatter'}, {'type': 'bar'}]]
)

# 1. 実測値 vs 予測値
fig_rf.add_trace(
    go.Scatter(
        x=y_test_original,
        y=rf_pred_test,
        mode='markers',
        marker=dict(size=5, color='blue', opacity=0.6),
        name='Test Data',
        customdata=np.column_stack([store_names, cities, cuisine_types, store_clusters, cuisine_clusters, 
                                     num_seats, avg_population, dinner_price, lunch_price, 
                                     is_family, is_friend, is_alone]),
        hovertemplate='<b>RST_TITLE:</b> %{customdata[0]}<br>' +
                      '<b>CITY:</b> %{customdata[1]}<br>' +
                      '<b>CUISINE_CAT_origin:</b> %{customdata[2]}<br>' +
                      '<b>store_cluster:</b> %{customdata[3]}<br>' +
                      '<b>cuisine_cluster_id:</b> %{customdata[4]}<br>' +
                      '<b>NUM_SEATS:</b> %{customdata[5]:.0f}<br>' +
                      '<b>AVG_MONTHLY_POPULATION:</b> %{customdata[6]:,.0f}<br>' +
                      '<b>DINNER_PRICE:</b> %{customdata[7]:,.0f}円<br>' +
                      '<b>LUNCH_PRICE:</b> %{customdata[8]:,.0f}円<br>' +
                      '<b>IS_FAMILY_FRIENDLY:</b> %{customdata[9]:.0f}<br>' +
                      '<b>IS_FRIEND_FRIENDLY:</b> %{customdata[10]:.0f}<br>' +
                      '<b>IS_ALONE_FRIENDLY:</b> %{customdata[11]:.0f}<br>' +
                      '<b>actual:</b> %{x:,.1f}本/月<br>' +
                      '<b>predicted:</b> %{y:,.1f}本/月<br>' +
                      '<b>error:</b> %{text:,.1f}本/月<extra></extra>',
        text=y_test_original - rf_pred_test
    ),
    row=1, col=1
)
fig_rf.add_trace(
    go.Scatter(
        x=[0, max_val],
        y=[0, max_val],
        mode='lines',
        line=dict(color='red', dash='dash'),
        name='Perfect Prediction',
        showlegend=False
    ),
    row=1, col=1
)

# 2. ヒストグラム
fig_rf.add_trace(
    go.Histogram(
        x=y_test_original,
        name='実測値',
        marker=dict(color='blue', opacity=0.5),
        nbinsx=50
    ),
    row=1, col=2
)
fig_rf.add_trace(
    go.Histogram(
        x=rf_pred_test,
        name='予測値',
        marker=dict(color='red', opacity=0.5),
        nbinsx=50
    ),
    row=1, col=2
)

# 3. 実測値 vs APE
ape_rf = np.abs((y_test_original - rf_pred_test) / y_test_original) * 100
fig_rf.add_trace(
    go.Scatter(
        x=y_test_original,
        y=ape_rf,
        mode='markers',
        marker=dict(size=5, color='green', opacity=0.6),
        name='APE',
        customdata=np.column_stack([store_names, cities, cuisine_types, store_clusters, cuisine_clusters, 
                                     num_seats, avg_population, dinner_price, lunch_price, 
                                     is_family, is_friend, is_alone]),
        hovertemplate='<b>RST_TITLE:</b> %{customdata[0]}<br>' +
                      '<b>CITY:</b> %{customdata[1]}<br>' +
                      '<b>CUISINE_CAT_origin:</b> %{customdata[2]}<br>' +
                      '<b>store_cluster:</b> %{customdata[3]}<br>' +
                      '<b>cuisine_cluster_id:</b> %{customdata[4]}<br>' +
                      '<b>NUM_SEATS:</b> %{customdata[5]:.0f}<br>' +
                      '<b>AVG_MONTHLY_POPULATION:</b> %{customdata[6]:,.0f}<br>' +
                      '<b>DINNER_PRICE:</b> %{customdata[7]:,.0f}円<br>' +
                      '<b>LUNCH_PRICE:</b> %{customdata[8]:,.0f}円<br>' +
                      '<b>IS_FAMILY_FRIENDLY:</b> %{customdata[9]:.0f}<br>' +
                      '<b>IS_FRIEND_FRIENDLY:</b> %{customdata[10]:.0f}<br>' +
                      '<b>IS_ALONE_FRIENDLY:</b> %{customdata[11]:.0f}<br>' +
                      '<b>actual:</b> %{x:,.1f}本/月<br>' +
                      '<b>APE:</b> %{y:.2f}%<extra></extra>'
    ),
    row=2, col=1
)

# 4. 特徴量重要度
feature_importance_rf = pd.DataFrame({
    'feature': feature_names,
    'importance': model_rf.feature_importances_
}).sort_values('importance', ascending=False).head(25)

colors_rf = ['red' if 'cuisine_origin' in f or 'city' in f else 'orange' for f in feature_importance_rf['feature']]

fig_rf.add_trace(
    go.Bar(
        y=feature_importance_rf['feature'][::-1],
        x=feature_importance_rf['importance'][::-1],
        orientation='h',
        marker=dict(color=colors_rf[::-1]),
        name='Importance',
        showlegend=False
    ),
    row=2, col=2
)

# レイアウト更新
fig_rf.update_xaxes(title_text="実測値 (本/月)", row=1, col=1)
fig_rf.update_yaxes(title_text="予測値 (本/月)", row=1, col=1)
fig_rf.update_xaxes(title_text="売上 (本/月)", row=1, col=2)
fig_rf.update_yaxes(title_text="頻度", row=1, col=2)
fig_rf.update_xaxes(title_text="実測値 (本/月)", row=2, col=1)
fig_rf.update_yaxes(title_text="絶対パーセント誤差 (%)", row=2, col=1)
fig_rf.update_xaxes(title_text="重要度", row=2, col=2)
fig_rf.update_yaxes(title_text="特徴量", row=2, col=2)

fig_rf.update_layout(
    height=1000, 
    width=1400, 
    showlegend=True, 
    title_text="Random Forest 回帰結果（CUISINE_CAT_origin & CITY Embedding追加）"
)
fig_rf.write_html(os.path.join(results_folder, 'random_forest_analysis.html'))

# 特徴量重要度を保存
feature_importance_lgb_full = pd.DataFrame({
    'feature': feature_names,
    'importance': model_lgb.feature_importance(importance_type='gain')
}).sort_values('importance', ascending=False)
feature_importance_lgb_full.to_csv(os.path.join(results_folder, 'feature_importance_lightgbm.csv'), index=False, encoding='utf-8-sig')

feature_importance_rf_full = pd.DataFrame({
    'feature': feature_names,
    'importance': model_rf.feature_importances_
}).sort_values('importance', ascending=False)
feature_importance_rf_full.to_csv(os.path.join(results_folder, 'feature_importance_random_forest.csv'), index=False, encoding='utf-8-sig')

# Encodingマップを保存（再利用のため）
encoding_maps = {
    'cuisine_median_map': cuisine_median_map,
    'store_median_map': store_median_map,
    'cuisine_origin_median_map': cuisine_origin_median_map,
    'city_median_map': city_median_map,
    'cuisine_origin_count_map': cuisine_origin_count_map,
    'city_count_map': city_count_map
}
pd.DataFrame([encoding_maps]).to_json(os.path.join(results_folder, 'encoding_maps.json'), orient='records', force_ascii=False)

print("\n" + "=" * 100)
print("【完了】")
print("=" * 100)
print(f"\n生成されたファイル（{results_folder}フォルダ）:")
print("  1. model_comparison.csv - モデル比較結果（RMSE, R², MAPE）")
print("  2. test_predictions.csv - テストデータの予測結果")
print("  3. lightgbm_analysis.html - LightGBM分析グラフ（2x2サブプロット）")
print("  4. random_forest_analysis.html - Random Forest分析グラフ（2x2サブプロット）")
print("  5. feature_importance_lightgbm.csv - LightGBM特徴量重要度（全特徴量）")
print("  6. feature_importance_random_forest.csv - Random Forest特徴量重要度（全特徴量）")
print("  7. encoding_maps.json - カテゴリ変数のEncodingマップ（予測時に再利用可能）")
print("\n【追加された新規特徴量】")
print("  - cuisine_origin_median_log: CUISINE_CAT_originごとの売上中央値（Target Encoding）")
print("  - city_median_log: CITYごとの売上中央値（Target Encoding）")
print("  - cuisine_origin_frequency: CUISINE_CAT_originの出現頻度（Frequency Encoding）")
print("  - city_frequency: CITYの出現頻度（Frequency Encoding）")
print("\n" + "=" * 100)


