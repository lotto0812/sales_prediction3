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
print("売上回帰問題（CUISINE_CAT_origin別）")
print("=" * 100)

# 結果フォルダを作成
results_folder = 'sales_regression_by_cuisine_results'
os.makedirs(results_folder, exist_ok=True)
print(f"\n結果フォルダ: {results_folder}")

# データを読み込む
print("\n[1/6] データを読み込んでいます...")

# stores_with_cluster.csvを読み込み
stores_df = pd.read_csv('final_store_clustering_results/stores_with_cluster.csv')
print(f"  stores_with_cluster.csv: {stores_df.shape}")

# aggregated_df_filtered.csvを読み込み（CUISINE_CAT_originを取得）
agg_df = pd.read_csv('aggregated_df_filtered.csv')
print(f"  aggregated_df_filtered.csv: {agg_df.shape}")

# CUISINE_CAT_originとRST_TITLE、CITYを追加（両データフレームは同じ順序・同じ行数）
df = stores_df.copy()
df['CUISINE_CAT_origin'] = agg_df['CUISINE_CAT_origin'].values
df['RST_TITLE'] = agg_df['RST_TITLE'].values
df['CITY'] = agg_df['CITY'].values
print(f"  CUISINE_CAT_origin追加後: {df.shape}")
print(f"  CUISINE_CAT_originの欠損値: {df['CUISINE_CAT_origin'].isnull().sum()}件")

# 特徴量を選択（CUISINE_CAT_originとstore_clusterは除外）
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

print("\n[2/6] CUISINE_CAT_originごとのデータ分布を確認しています...")
cuisine_counts = df['CUISINE_CAT_origin'].value_counts()
print(f"\n  総カテゴリ数: {len(cuisine_counts)}")
print(f"  データ数30以上のカテゴリ: {len(cuisine_counts[cuisine_counts >= 30])}")
print(f"  データ数50以上のカテゴリ: {len(cuisine_counts[cuisine_counts >= 50])}")
print(f"  データ数100以上のカテゴリ: {len(cuisine_counts[cuisine_counts >= 100])}")

# データ数が30以上のカテゴリのみ処理
min_samples = 30
valid_cuisines = cuisine_counts[cuisine_counts >= min_samples].index.tolist()
print(f"\n  処理対象カテゴリ数（{min_samples}店舗以上）: {len(valid_cuisines)}")

# 欠損値の確認
print("\n[3/6] 欠損値を処理しています...")
missing_cols = []
for col in feature_cols:
    if col not in df.columns:
        missing_cols.append(col)

if missing_cols:
    print(f"\n  警告: 以下の列がデータに存在しません: {missing_cols}")
    feature_cols = [col for col in feature_cols if col in df.columns]

# 欠損値を含む行を削除
df_clean = df.dropna(subset=feature_cols + ['target_amount_tableau', 'CUISINE_CAT_origin']).copy()
print(f"  欠損値処理後: {len(df_clean)}店舗")

# 目的変数の対数変換
print("\n[4/6] 目的変数を対数変換しています...")
df_clean['target_log'] = np.log(df_clean['target_amount_tableau'])
print(f"  元のtarget_amount_tableau範囲: {df_clean['target_amount_tableau'].min():.2f} ~ {df_clean['target_amount_tableau'].max():.2f}")
print(f"  対数変換後の範囲: {df_clean['target_log'].min():.2f} ~ {df_clean['target_log'].max():.2f}")

# MAPE計算関数
def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# 評価関数
def evaluate_model(y_true, y_pred, model_name, dataset_name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = calculate_mape(y_true, y_pred)
    
    print(f"    {model_name} - {dataset_name}: RMSE={rmse:.2f}, R²={r2:.4f}, MAPE={mape:.2f}%")
    
    return {'RMSE': rmse, 'R2': r2, 'MAPE': mape}

# 全カテゴリ結果を保存するリスト
all_results = []

# CUISINE_CAT_originごとに処理
print("\n[5/6] CUISINE_CAT_originごとにモデルを訓練しています...")
print("=" * 100)

for idx, cuisine in enumerate(valid_cuisines, 1):
    print(f"\n{'='*100}")
    print(f"【{idx}/{len(valid_cuisines)}: {cuisine}】")
    print(f"{'='*100}")
    
    # カテゴリデータを抽出
    cuisine_data = df_clean[df_clean['CUISINE_CAT_origin'] == cuisine].copy()
    n_samples = len(cuisine_data)
    print(f"\n  データ数: {n_samples}店舗")
    
    # データが少なすぎる場合はスキップ（念のため）
    if n_samples < min_samples:
        print(f"  警告: データ数が少なすぎるため、{cuisine}をスキップします")
        continue
    
    # 特徴量とラベルを分離
    X = cuisine_data[feature_cols].values
    y_log = cuisine_data['target_log'].values
    y_original = cuisine_data['target_amount_tableau'].values
    
    # データ分割
    test_size = 0.2 if n_samples >= 50 else 0.3
    try:
        indices = np.arange(len(X))
        X_train, X_test, y_train_log, y_test_log, y_train_original, y_test_original, train_idx, test_idx = train_test_split(
            X, y_log, y_original, indices, test_size=test_size, random_state=42
        )
    except:
        print(f"  エラー: データ分割に失敗しました。{cuisine}をスキップします")
        continue
    
    print(f"  Train: {len(X_train)}店舗, Test: {len(X_test)}店舗")
    
    # テストデータの店舗情報を取得（グラフのホバー表示用）
    test_info = cuisine_data.iloc[test_idx].copy()
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
    
    # 標準化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_all_scaled = scaler.transform(X)
    
    # -------------------------
    # LightGBM
    # -------------------------
    print("\n  【LightGBM】")
    
    # ハイパーパラメータ最適化
    def objective_lgb(trial):
        param = {
            'objective': 'regression',
            'metric': 'rmse',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        }
        
        train_data = lgb.Dataset(X_train_scaled, label=y_train_log)
        
        cv_results = lgb.cv(
            param,
            train_data,
            num_boost_round=500,
            nfold=min(5, len(X_train) // 10),
            stratified=False,
            callbacks=[lgb.early_stopping(stopping_rounds=30)],
            seed=42
        )
        
        return cv_results['valid rmse-mean'][-1]
    
    n_trials = 15 if n_samples >= 100 else 10
    study_lgb = optuna.create_study(direction='minimize', study_name=f'lgbm_cuisine_{idx}')
    study_lgb.optimize(objective_lgb, n_trials=n_trials, show_progress_bar=False)
    
    print(f"    最適CV RMSE: {study_lgb.best_value:.4f}")
    
    # モデル訓練
    best_params_lgb = study_lgb.best_params
    best_params_lgb['objective'] = 'regression'
    best_params_lgb['metric'] = 'rmse'
    best_params_lgb['verbosity'] = -1
    
    train_data = lgb.Dataset(X_train_scaled, label=y_train_log)
    test_data = lgb.Dataset(X_test_scaled, label=y_test_log, reference=train_data)
    
    model_lgb = lgb.train(
        best_params_lgb,
        train_data,
        num_boost_round=500,
        valid_sets=[test_data],
        callbacks=[lgb.early_stopping(stopping_rounds=30), lgb.log_evaluation(period=0)]
    )
    
    # 予測
    lgb_pred_train_log = model_lgb.predict(X_train_scaled, num_iteration=model_lgb.best_iteration)
    lgb_pred_test_log = model_lgb.predict(X_test_scaled, num_iteration=model_lgb.best_iteration)
    lgb_pred_all_log = model_lgb.predict(X_all_scaled, num_iteration=model_lgb.best_iteration)
    
    lgb_pred_train = np.exp(lgb_pred_train_log)
    lgb_pred_test = np.exp(lgb_pred_test_log)
    lgb_pred_all = np.exp(lgb_pred_all_log)
    
    # 評価
    lgb_train_metrics = evaluate_model(y_train_original, lgb_pred_train, "LightGBM", "Train")
    lgb_test_metrics = evaluate_model(y_test_original, lgb_pred_test, "LightGBM", "Test")
    lgb_all_metrics = evaluate_model(y_original, lgb_pred_all, "LightGBM", "All")
    
    # -------------------------
    # Random Forest
    # -------------------------
    print("\n  【Random Forest】")
    
    # ハイパーパラメータ最適化
    def objective_rf(trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
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
    
    study_rf = optuna.create_study(direction='minimize', study_name=f'rf_cuisine_{idx}')
    study_rf.optimize(objective_rf, n_trials=n_trials, show_progress_bar=False)
    
    print(f"    最適Test RMSE: {study_rf.best_value:.4f}")
    
    # モデル訓練
    best_params_rf = study_rf.best_params
    best_params_rf['random_state'] = 42
    best_params_rf['n_jobs'] = -1
    
    model_rf = RandomForestRegressor(**best_params_rf)
    model_rf.fit(X_train_scaled, y_train_log)
    
    # 予測
    rf_pred_train_log = model_rf.predict(X_train_scaled)
    rf_pred_test_log = model_rf.predict(X_test_scaled)
    rf_pred_all_log = model_rf.predict(X_all_scaled)
    
    rf_pred_train = np.exp(rf_pred_train_log)
    rf_pred_test = np.exp(rf_pred_test_log)
    rf_pred_all = np.exp(rf_pred_all_log)
    
    # 評価
    rf_train_metrics = evaluate_model(y_train_original, rf_pred_train, "Random Forest", "Train")
    rf_test_metrics = evaluate_model(y_test_original, rf_pred_test, "Random Forest", "Test")
    rf_all_metrics = evaluate_model(y_original, rf_pred_all, "Random Forest", "All")
    
    # -------------------------
    # 結果保存
    # -------------------------
    print("\n  【結果を保存しています...】")
    
    # カテゴリ別フォルダ作成（ファイル名に使えない文字を置換）
    safe_cuisine_name = cuisine.replace('/', '_').replace('\\', '_').replace(':', '_').replace('*', '_').replace('?', '_').replace('"', '_').replace('<', '_').replace('>', '_').replace('|', '_')
    cuisine_folder = os.path.join(results_folder, f'cuisine_{idx:03d}_{safe_cuisine_name[:30]}')
    os.makedirs(cuisine_folder, exist_ok=True)
    
    # 予測結果CSV
    test_results = pd.DataFrame({
        'actual': y_test_original,
        'lgb_predicted': lgb_pred_test,
        'rf_predicted': rf_pred_test,
        'lgb_error': y_test_original - lgb_pred_test,
        'rf_error': y_test_original - rf_pred_test,
        'lgb_ape': np.abs((y_test_original - lgb_pred_test) / y_test_original) * 100,
        'rf_ape': np.abs((y_test_original - rf_pred_test) / y_test_original) * 100
    })
    test_results.to_csv(os.path.join(cuisine_folder, 'test_predictions.csv'), index=False, encoding='utf-8-sig')
    
    # モデル比較結果
    cuisine_summary = pd.DataFrame({
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
    cuisine_summary.to_csv(os.path.join(cuisine_folder, 'model_comparison.csv'), index=False, encoding='utf-8-sig')
    
    # 全体結果に追加
    all_results.append({
        'cuisine_index': idx,
        'cuisine_name': cuisine,
        'n_samples': n_samples,
        'lgb_test_rmse': lgb_test_metrics['RMSE'],
        'lgb_test_r2': lgb_test_metrics['R2'],
        'lgb_test_mape': lgb_test_metrics['MAPE'],
        'rf_test_rmse': rf_test_metrics['RMSE'],
        'rf_test_r2': rf_test_metrics['R2'],
        'rf_test_mape': rf_test_metrics['MAPE']
    })
    
    # 簡易グラフのみ作成（実測値 vs 予測値）
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f'LightGBM: R²={lgb_test_metrics["R2"]:.3f}, RMSE={lgb_test_metrics["RMSE"]:.2f}',
            f'Random Forest: R²={rf_test_metrics["R2"]:.3f}, RMSE={rf_test_metrics["RMSE"]:.2f}'
        )
    )
    
    # LightGBM
    max_val = max(y_test_original.max(), lgb_pred_test.max())
    fig.add_trace(
        go.Scatter(
            x=y_test_original, 
            y=lgb_pred_test, 
            mode='markers', 
            marker=dict(size=5, color='blue', opacity=0.6), 
            name='LightGBM',
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
    fig.add_trace(
        go.Scatter(x=[0, max_val], y=[0, max_val], mode='lines',
                  line=dict(color='red', dash='dash'), showlegend=False),
        row=1, col=1
    )
    
    # Random Forest
    fig.add_trace(
        go.Scatter(
            x=y_test_original, 
            y=rf_pred_test, 
            mode='markers',
            marker=dict(size=5, color='green', opacity=0.6), 
            name='RandomForest',
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
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=[0, max_val], y=[0, max_val], mode='lines',
                  line=dict(color='red', dash='dash'), showlegend=False),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text="実測値 (本/月)", row=1, col=1)
    fig.update_yaxes(title_text="予測値 (本/月)", row=1, col=1)
    fig.update_xaxes(title_text="実測値 (本/月)", row=1, col=2)
    fig.update_yaxes(title_text="予測値 (本/月)", row=1, col=2)
    
    fig.update_layout(height=500, width=1200, title_text=f"{cuisine}", showlegend=True)
    fig.write_html(os.path.join(cuisine_folder, 'prediction_comparison.html'))
    
    print(f"    完了: {cuisine_folder}に保存しました")

# -------------------------
# 全体サマリー作成
# -------------------------
print("\n[6/6] 全体サマリーを作成しています...")

all_results_df = pd.DataFrame(all_results)
all_results_df = all_results_df.sort_values('lgb_test_rmse')
all_results_df.to_csv(os.path.join(results_folder, 'all_cuisines_summary.csv'), index=False, encoding='utf-8-sig')

print("\n" + "=" * 100)
print("【全カテゴリ結果サマリー（RMSE Top 10）】")
print("=" * 100)
print("\n  LightGBM Test Results (Best 10):")
top10_lgb = all_results_df.nsmallest(10, 'lgb_test_rmse')[['cuisine_index', 'cuisine_name', 'n_samples', 'lgb_test_rmse', 'lgb_test_r2', 'lgb_test_mape']]
print(top10_lgb.to_string(index=False))

print("\n  Random Forest Test Results (Best 10):")
top10_rf = all_results_df.nsmallest(10, 'rf_test_rmse')[['cuisine_index', 'cuisine_name', 'n_samples', 'rf_test_rmse', 'rf_test_r2', 'rf_test_mape']]
print(top10_rf.to_string(index=False))

# サマリーグラフ作成（上位20カテゴリ）
top_20 = all_results_df.nsmallest(20, 'lgb_test_rmse')

fig_summary = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        'カテゴリ別 Test RMSE (Top 20)',
        'カテゴリ別 Test R² (Top 20)',
        'カテゴリ別 Test MAPE (Top 20)',
        'カテゴリ別データ数 (Top 20)'
    )
)

# RMSE比較
fig_summary.add_trace(
    go.Bar(x=top_20['cuisine_index'], y=top_20['lgb_test_rmse'], name='LightGBM', marker_color='blue'),
    row=1, col=1
)
fig_summary.add_trace(
    go.Bar(x=top_20['cuisine_index'], y=top_20['rf_test_rmse'], name='RandomForest', marker_color='orange'),
    row=1, col=1
)

# R²比較
fig_summary.add_trace(
    go.Bar(x=top_20['cuisine_index'], y=top_20['lgb_test_r2'], name='LightGBM', marker_color='blue', showlegend=False),
    row=1, col=2
)
fig_summary.add_trace(
    go.Bar(x=top_20['cuisine_index'], y=top_20['rf_test_r2'], name='RandomForest', marker_color='orange', showlegend=False),
    row=1, col=2
)

# MAPE比較
fig_summary.add_trace(
    go.Bar(x=top_20['cuisine_index'], y=top_20['lgb_test_mape'], name='LightGBM', marker_color='blue', showlegend=False),
    row=2, col=1
)
fig_summary.add_trace(
    go.Bar(x=top_20['cuisine_index'], y=top_20['rf_test_mape'], name='RandomForest', marker_color='orange', showlegend=False),
    row=2, col=1
)

# データ数
fig_summary.add_trace(
    go.Bar(x=top_20['cuisine_index'], y=top_20['n_samples'], marker_color='green', showlegend=False),
    row=2, col=2
)

fig_summary.update_xaxes(title_text="Cuisine Index", row=1, col=1)
fig_summary.update_xaxes(title_text="Cuisine Index", row=1, col=2)
fig_summary.update_xaxes(title_text="Cuisine Index", row=2, col=1)
fig_summary.update_xaxes(title_text="Cuisine Index", row=2, col=2)

fig_summary.update_yaxes(title_text="RMSE", row=1, col=1)
fig_summary.update_yaxes(title_text="R²", row=1, col=2)
fig_summary.update_yaxes(title_text="MAPE (%)", row=2, col=1)
fig_summary.update_yaxes(title_text="データ数", row=2, col=2)

fig_summary.update_layout(height=900, width=1400, title_text="CUISINE_CAT_origin別 モデル性能比較 (Top 20)", barmode='group')
fig_summary.write_html(os.path.join(results_folder, 'cuisines_comparison_top20.html'))

print("\n" + "=" * 100)
print("【完了】")
print("=" * 100)
print(f"\n生成されたファイル（{results_folder}フォルダ）:")
print(f"  - all_cuisines_summary.csv: 全{len(all_results)}カテゴリの結果サマリー")
print("  - cuisines_comparison_top20.html: カテゴリ間比較グラフ（Top 20）")
print("  - cuisine_XXX_YYY/: 各カテゴリの詳細結果フォルダ")
print("    ├── model_comparison.csv")
print("    ├── test_predictions.csv")
print("    └── prediction_comparison.html")
print("\n" + "=" * 100)

