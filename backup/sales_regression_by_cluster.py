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
print("売上回帰問題（store_cluster別）")
print("=" * 100)

# 結果フォルダを作成
results_folder = 'sales_regression_by_cluster_results'
os.makedirs(results_folder, exist_ok=True)
print(f"\n結果フォルダ: {results_folder}")

# データを読み込む
print("\n[1/6] データを読み込んでいます...")
input_file = 'final_store_clustering_results/stores_with_cluster.csv'
print(f"  ファイル: {input_file}")
df = pd.read_csv(input_file)
print(f"  データ形状: {df.shape}")

# 特徴量を選択（store_clusterは除外）
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
    'cuisine_cluster_id'
]

print("\n[2/6] store_clusterごとのデータ分布を確認しています...")
cluster_counts = df['store_cluster'].value_counts().sort_index()
print("\n  【store_cluster別データ数】")
for cluster_id, count in cluster_counts.items():
    print(f"    Cluster {int(cluster_id)}: {count}店舗")

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
df_clean = df.dropna(subset=feature_cols + ['target_amount_tableau', 'store_cluster']).copy()
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

# 全クラスター結果を保存するリスト
all_results = []

# store_clusterごとに処理
print("\n[5/6] store_clusterごとにモデルを訓練しています...")
print("=" * 100)

for cluster_id in sorted(df_clean['store_cluster'].unique()):
    print(f"\n{'='*100}")
    print(f"【Store Cluster {int(cluster_id)}】")
    print(f"{'='*100}")
    
    # クラスターデータを抽出
    cluster_data = df_clean[df_clean['store_cluster'] == cluster_id].copy()
    n_samples = len(cluster_data)
    print(f"\n  データ数: {n_samples}店舗")
    
    # データが少なすぎる場合はスキップ
    if n_samples < 30:
        print(f"  警告: データ数が少なすぎるため、Cluster {int(cluster_id)}をスキップします")
        continue
    
    # 特徴量とラベルを分離
    X = cluster_data[feature_cols].values
    y_log = cluster_data['target_log'].values
    y_original = cluster_data['target_amount_tableau'].values
    
    # cuisine_cluster_idごとの中央値を特徴量として追加
    cuisine_median_map = cluster_data.groupby('cuisine_cluster_id')['target_log'].median().to_dict()
    cuisine_cluster_median = cluster_data['cuisine_cluster_id'].map(cuisine_median_map).values
    X_with_median = np.column_stack([X, cuisine_cluster_median])
    
    feature_names = feature_cols + ['cuisine_cluster_median_log']
    
    # データ分割
    test_size = 0.2 if n_samples >= 50 else 0.3
    X_train, X_test, y_train_log, y_test_log, y_train_original, y_test_original = train_test_split(
        X_with_median, y_log, y_original, test_size=test_size, random_state=42
    )
    
    print(f"  Train: {len(X_train)}店舗, Test: {len(X_test)}店舗")
    
    # 標準化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_all_scaled = scaler.transform(X_with_median)
    
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
    
    n_trials = 20 if n_samples >= 100 else 10
    study_lgb = optuna.create_study(direction='minimize', study_name=f'lgbm_cluster_{int(cluster_id)}')
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
    
    study_rf = optuna.create_study(direction='minimize', study_name=f'rf_cluster_{int(cluster_id)}')
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
    
    # クラスター別フォルダ作成
    cluster_folder = os.path.join(results_folder, f'cluster_{int(cluster_id)}')
    os.makedirs(cluster_folder, exist_ok=True)
    
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
    test_results.to_csv(os.path.join(cluster_folder, 'test_predictions.csv'), index=False, encoding='utf-8-sig')
    
    # モデル比較結果
    cluster_summary = pd.DataFrame({
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
    cluster_summary.to_csv(os.path.join(cluster_folder, 'model_comparison.csv'), index=False, encoding='utf-8-sig')
    
    # 全体結果に追加
    all_results.append({
        'store_cluster': int(cluster_id),
        'n_samples': n_samples,
        'lgb_test_rmse': lgb_test_metrics['RMSE'],
        'lgb_test_r2': lgb_test_metrics['R2'],
        'lgb_test_mape': lgb_test_metrics['MAPE'],
        'rf_test_rmse': rf_test_metrics['RMSE'],
        'rf_test_r2': rf_test_metrics['R2'],
        'rf_test_mape': rf_test_metrics['MAPE']
    })
    
    # -------------------------
    # グラフ作成（LightGBM）
    # -------------------------
    fig_lgb = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            f'実測値 vs 予測値 (LightGBM)<br>RMSE={lgb_test_metrics["RMSE"]:.2f}, R²={lgb_test_metrics["R2"]:.3f}, MAPE={lgb_test_metrics["MAPE"]:.2f}%',
            '実測値と予測値のヒストグラム (LightGBM)',
            '実測値 vs 絶対パーセント誤差 (LightGBM)',
            '特徴量重要度 Top 20 (LightGBM)'
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
            hovertemplate='実測値: %{x:.2f}<br>予測値: %{y:.2f}<extra></extra>'
        ),
        row=1, col=1
    )
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
        go.Histogram(x=y_test_original, name='実測値', marker=dict(color='blue', opacity=0.5), nbinsx=30),
        row=1, col=2
    )
    fig_lgb.add_trace(
        go.Histogram(x=lgb_pred_test, name='予測値', marker=dict(color='red', opacity=0.5), nbinsx=30),
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
            hovertemplate='実測値: %{x:.2f}<br>APE: %{y:.2f}%<extra></extra>'
        ),
        row=2, col=1
    )
    
    # 4. 特徴量重要度
    feature_importance_lgb = pd.DataFrame({
        'feature': feature_names,
        'importance': model_lgb.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False).head(20)
    
    fig_lgb.add_trace(
        go.Bar(
            y=feature_importance_lgb['feature'][::-1],
            x=feature_importance_lgb['importance'][::-1],
            orientation='h',
            marker=dict(color='purple'),
            name='Importance'
        ),
        row=2, col=2
    )
    
    fig_lgb.update_xaxes(title_text="実測値 (円)", row=1, col=1)
    fig_lgb.update_yaxes(title_text="予測値 (円)", row=1, col=1)
    fig_lgb.update_xaxes(title_text="売上 (円)", row=1, col=2)
    fig_lgb.update_yaxes(title_text="頻度", row=1, col=2)
    fig_lgb.update_xaxes(title_text="実測値 (円)", row=2, col=1)
    fig_lgb.update_yaxes(title_text="絶対パーセント誤差 (%)", row=2, col=1)
    fig_lgb.update_xaxes(title_text="重要度 (Gain)", row=2, col=2)
    fig_lgb.update_yaxes(title_text="特徴量", row=2, col=2)
    
    fig_lgb.update_layout(height=1000, width=1400, showlegend=True, title_text=f"Store Cluster {int(cluster_id)} - LightGBM")
    fig_lgb.write_html(os.path.join(cluster_folder, 'lightgbm_analysis.html'))
    
    # -------------------------
    # グラフ作成（Random Forest）
    # -------------------------
    fig_rf = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            f'実測値 vs 予測値 (Random Forest)<br>RMSE={rf_test_metrics["RMSE"]:.2f}, R²={rf_test_metrics["R2"]:.3f}, MAPE={rf_test_metrics["MAPE"]:.2f}%',
            '実測値と予測値のヒストグラム (Random Forest)',
            '実測値 vs 絶対パーセント誤差 (Random Forest)',
            '特徴量重要度 Top 20 (Random Forest)'
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
            hovertemplate='実測値: %{x:.2f}<br>予測値: %{y:.2f}<extra></extra>'
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
        go.Histogram(x=y_test_original, name='実測値', marker=dict(color='blue', opacity=0.5), nbinsx=30),
        row=1, col=2
    )
    fig_rf.add_trace(
        go.Histogram(x=rf_pred_test, name='予測値', marker=dict(color='red', opacity=0.5), nbinsx=30),
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
            hovertemplate='実測値: %{x:.2f}<br>APE: %{y:.2f}%<extra></extra>'
        ),
        row=2, col=1
    )
    
    # 4. 特徴量重要度
    feature_importance_rf = pd.DataFrame({
        'feature': feature_names,
        'importance': model_rf.feature_importances_
    }).sort_values('importance', ascending=False).head(20)
    
    fig_rf.add_trace(
        go.Bar(
            y=feature_importance_rf['feature'][::-1],
            x=feature_importance_rf['importance'][::-1],
            orientation='h',
            marker=dict(color='orange'),
            name='Importance'
        ),
        row=2, col=2
    )
    
    fig_rf.update_xaxes(title_text="実測値 (円)", row=1, col=1)
    fig_rf.update_yaxes(title_text="予測値 (円)", row=1, col=1)
    fig_rf.update_xaxes(title_text="売上 (円)", row=1, col=2)
    fig_rf.update_yaxes(title_text="頻度", row=1, col=2)
    fig_rf.update_xaxes(title_text="実測値 (円)", row=2, col=1)
    fig_rf.update_yaxes(title_text="絶対パーセント誤差 (%)", row=2, col=1)
    fig_rf.update_xaxes(title_text="重要度", row=2, col=2)
    fig_rf.update_yaxes(title_text="特徴量", row=2, col=2)
    
    fig_rf.update_layout(height=1000, width=1400, showlegend=True, title_text=f"Store Cluster {int(cluster_id)} - Random Forest")
    fig_rf.write_html(os.path.join(cluster_folder, 'random_forest_analysis.html'))
    
    # 特徴量重要度保存
    feature_importance_lgb.to_csv(os.path.join(cluster_folder, 'feature_importance_lightgbm.csv'), index=False, encoding='utf-8-sig')
    feature_importance_rf.to_csv(os.path.join(cluster_folder, 'feature_importance_random_forest.csv'), index=False, encoding='utf-8-sig')
    
    print(f"    完了: cluster_{int(cluster_id)}フォルダに保存しました")

# -------------------------
# 全体サマリー作成
# -------------------------
print("\n[6/6] 全体サマリーを作成しています...")

all_results_df = pd.DataFrame(all_results)
all_results_df = all_results_df.sort_values('store_cluster')
all_results_df.to_csv(os.path.join(results_folder, 'all_clusters_summary.csv'), index=False, encoding='utf-8-sig')

print("\n" + "=" * 100)
print("【全クラスター結果サマリー】")
print("=" * 100)
print("\n  LightGBM Test Results:")
print(all_results_df[['store_cluster', 'n_samples', 'lgb_test_rmse', 'lgb_test_r2', 'lgb_test_mape']].to_string(index=False))

print("\n  Random Forest Test Results:")
print(all_results_df[['store_cluster', 'n_samples', 'rf_test_rmse', 'rf_test_r2', 'rf_test_mape']].to_string(index=False))

# サマリーグラフ作成
fig_summary = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        'クラスター別 Test RMSE',
        'クラスター別 Test R²',
        'クラスター別 Test MAPE',
        'クラスター別データ数'
    )
)

# RMSE比較
fig_summary.add_trace(
    go.Bar(x=all_results_df['store_cluster'], y=all_results_df['lgb_test_rmse'], name='LightGBM', marker_color='blue'),
    row=1, col=1
)
fig_summary.add_trace(
    go.Bar(x=all_results_df['store_cluster'], y=all_results_df['rf_test_rmse'], name='RandomForest', marker_color='orange'),
    row=1, col=1
)

# R²比較
fig_summary.add_trace(
    go.Bar(x=all_results_df['store_cluster'], y=all_results_df['lgb_test_r2'], name='LightGBM', marker_color='blue', showlegend=False),
    row=1, col=2
)
fig_summary.add_trace(
    go.Bar(x=all_results_df['store_cluster'], y=all_results_df['rf_test_r2'], name='RandomForest', marker_color='orange', showlegend=False),
    row=1, col=2
)

# MAPE比較
fig_summary.add_trace(
    go.Bar(x=all_results_df['store_cluster'], y=all_results_df['lgb_test_mape'], name='LightGBM', marker_color='blue', showlegend=False),
    row=2, col=1
)
fig_summary.add_trace(
    go.Bar(x=all_results_df['store_cluster'], y=all_results_df['rf_test_mape'], name='RandomForest', marker_color='orange', showlegend=False),
    row=2, col=2
)

# データ数
fig_summary.add_trace(
    go.Bar(x=all_results_df['store_cluster'], y=all_results_df['n_samples'], marker_color='green', showlegend=False),
    row=2, col=2
)

fig_summary.update_xaxes(title_text="Store Cluster", row=1, col=1)
fig_summary.update_xaxes(title_text="Store Cluster", row=1, col=2)
fig_summary.update_xaxes(title_text="Store Cluster", row=2, col=1)
fig_summary.update_xaxes(title_text="Store Cluster", row=2, col=2)

fig_summary.update_yaxes(title_text="RMSE", row=1, col=1)
fig_summary.update_yaxes(title_text="R²", row=1, col=2)
fig_summary.update_yaxes(title_text="MAPE (%)", row=2, col=1)
fig_summary.update_yaxes(title_text="データ数", row=2, col=2)

fig_summary.update_layout(height=900, width=1400, title_text="Store Cluster別 モデル性能比較", barmode='group')
fig_summary.write_html(os.path.join(results_folder, 'clusters_comparison.html'))

print("\n" + "=" * 100)
print("【完了】")
print("=" * 100)
print(f"\n生成されたファイル（{results_folder}フォルダ）:")
print("  - all_clusters_summary.csv: 全クラスターの結果サマリー")
print("  - clusters_comparison.html: クラスター間比較グラフ")
print("  - cluster_X/: 各クラスターの詳細結果フォルダ")
print("    ├── model_comparison.csv")
print("    ├── test_predictions.csv")
print("    ├── lightgbm_analysis.html")
print("    ├── random_forest_analysis.html")
print("    ├── feature_importance_lightgbm.csv")
print("    └── feature_importance_random_forest.csv")
print("\n" + "=" * 100)

