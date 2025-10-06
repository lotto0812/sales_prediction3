# -*- coding: utf-8 -*-
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import lightgbm as lgb
import optuna
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import os
warnings.filterwarnings('ignore')

print("=" * 100)
print("売上分類問題")
print("=" * 100)

# 結果フォルダを作成
results_folder = 'sales_classification_results'
os.makedirs(results_folder, exist_ok=True)
print(f"\n結果フォルダ: {results_folder}")

# データを読み込む
print("\n[1/9] データを読み込んでいます...")
input_file = 'final_store_clustering_results/stores_with_cluster.csv'
print(f"  ファイル: {input_file}")
df = pd.read_csv(input_file)
print(f"  データ形状: {df.shape}")

# 目的変数の作成（C1）
print("\n[2/9] 目的変数を作成しています...")
print("  ランク分け: 20~50(low), 50~100(middle), 100~200(high), 200以上(extra)")

def categorize_sales(value):
    if 20 <= value < 50:
        return 'low'
    elif 50 <= value < 100:
        return 'middle'
    elif 100 <= value < 200:
        return 'high'
    elif value >= 200:
        return 'extra'
    else:
        return None  # 20未満は除外

df['sales_category'] = df['target_amount_tableau'].apply(categorize_sales)

# 20未満のデータを除外
df_filtered = df[df['sales_category'].notna()].copy()
print(f"  除外データ: {len(df) - len(df_filtered)}店舗")
print(f"  分析対象: {len(df_filtered)}店舗")

# カテゴリの分布を表示
print("\n  【カテゴリ分布】")
category_dist = df_filtered['sales_category'].value_counts().sort_index()
for cat in ['low', 'middle', 'high', 'extra']:
    if cat in category_dist.index:
        count = category_dist[cat]
        pct = count / len(df_filtered) * 100
        print(f"    {cat}: {count}店舗 ({pct:.1f}%)")

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

print("\n[3/9] 特徴量を選択しています...")
print(f"  特徴量数: {len(feature_cols)}")

# 欠損値の確認
missing_cols = []
for col in feature_cols:
    if col not in df_filtered.columns:
        missing_cols.append(col)

if missing_cols:
    print(f"\n  警告: 以下の列がデータに存在しません: {missing_cols}")
    feature_cols = [col for col in feature_cols if col in df_filtered.columns]
    print(f"  使用可能な特徴量数: {len(feature_cols)}")

# 欠損値を含む行を削除
df_clean = df_filtered.dropna(subset=feature_cols).copy()
print(f"  欠損値を含む行を削除: {len(df_clean)}店舗")

# 特徴量とラベルを分離
X = df_clean[feature_cols].values
y = df_clean['sales_category'].values

# ラベルエンコーディング
label_mapping = {'low': 0, 'middle': 1, 'high': 2, 'extra': 3}
y_encoded = np.array([label_mapping[label] for label in y])

print(f"\n  X shape: {X.shape}")
print(f"  y shape: {y_encoded.shape}")

# クラスウェイトを計算（各ランクの割合の逆数）
from collections import Counter
label_counts = Counter(y_encoded)
total = len(y_encoded)
class_weight_dict = {}

print(f"\n  【クラス分布と逆数ウェイト】")
for i, label in enumerate(['low', 'middle', 'high', 'extra']):
    count = label_counts[i]
    proportion = count / total
    weight = 1.0 / proportion
    class_weight_dict[i] = weight
    print(f"    {label}: {count}件 ({proportion*100:.1f}%) → ウェイト: {weight:.3f}")

# データ分割
print("\n[4/9] データを分割しています...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"  Train: {X_train.shape[0]}店舗")
print(f"  Test: {X_test.shape[0]}店舗")

# クラスタごとの中央値を特徴量として追加
print("\n[5/8] クラスタごとの中央値を特徴量として追加しています...")

# 訓練データでクラスタごとの中央値を計算（Data leakage防止）
train_indices = df_clean.index[:len(X_train)]
test_indices = df_clean.index[len(X_train):]

# cuisine_cluster_id ごとの中央値
cuisine_median_map = df_clean.iloc[train_indices].groupby('cuisine_cluster_id')['target_amount_tableau'].median().to_dict()
print(f"  料理クラスタごとの中央値:")
for cid, median in sorted(cuisine_median_map.items()):
    print(f"    クラスタ {int(cid)}: {median:.2f}円")

# store_cluster ごとの中央値
store_median_map = df_clean.iloc[train_indices].groupby('store_cluster')['target_amount_tableau'].median().to_dict()
print(f"\n  店舗クラスタごとの中央値:")
for sid, median in sorted(store_median_map.items()):
    print(f"    クラスタ {int(sid)}: {median:.2f}円")

# 訓練データに適用
cuisine_cluster_median_train = df_clean.iloc[train_indices]['cuisine_cluster_id'].map(cuisine_median_map).values
store_cluster_median_train = df_clean.iloc[train_indices]['store_cluster'].map(store_median_map).values

# テストデータに適用（訓練データで計算した統計量を使用）
cuisine_cluster_median_test = df_clean.iloc[test_indices]['cuisine_cluster_id'].map(cuisine_median_map).values
store_cluster_median_test = df_clean.iloc[test_indices]['store_cluster'].map(store_median_map).values

# 全データに適用（評価用）
cuisine_cluster_median_all = df_clean['cuisine_cluster_id'].map(cuisine_median_map).values
store_cluster_median_all = df_clean['store_cluster'].map(store_median_map).values

# 新しい特徴量を追加
X_train_with_median = np.column_stack([X_train, cuisine_cluster_median_train, store_cluster_median_train])
X_test_with_median = np.column_stack([X_test, cuisine_cluster_median_test, store_cluster_median_test])
X_all_with_median = np.column_stack([X, cuisine_cluster_median_all, store_cluster_median_all])

print(f"\n  特徴量追加後のshape:")
print(f"    Train: {X_train_with_median.shape}")
print(f"    Test: {X_test_with_median.shape}")

# 標準化（C2）
print("\n[6/8] 特徴量を標準化しています...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_with_median)
X_test_scaled = scaler.transform(X_test_with_median)
X_all_scaled = scaler.transform(X_all_with_median)

# Optunaでハイパーパラメータ最適化
print("\n[7/9] Optunaでハイパーパラメータを最適化しています...")

def objective(trial):
    param = {
        'objective': 'multiclass',
        'num_class': 4,
        'metric': 'multi_logloss',
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
    
    # クラスウェイトを適用したサンプルウェイトを作成
    sample_weights = np.array([class_weight_dict[label] for label in y_train])
    
    # LightGBM データセット作成（ウェイト付き）
    train_data = lgb.Dataset(X_train_scaled, label=y_train, weight=sample_weights)
    
    # Cross-validation
    cv_results = lgb.cv(
        param,
        train_data,
        num_boost_round=1000,
        nfold=5,
        stratified=True,
        callbacks=[lgb.early_stopping(stopping_rounds=50)],
        seed=42
    )
    
    return cv_results['valid multi_logloss-mean'][-1]

# Optunaで最適化
study = optuna.create_study(direction='minimize', study_name='lgbm_classification')
study.optimize(objective, n_trials=50, show_progress_bar=True)

print(f"\n  最適なパラメータ:")
for key, value in study.best_params.items():
    print(f"    {key}: {value}")
print(f"  Best CV score: {study.best_value:.4f}")

# 最適なパラメータでモデルを訓練
print("\n[8/9] 最適なパラメータでモデルを訓練しています...")
best_params = study.best_params
best_params['objective'] = 'multiclass'
best_params['num_class'] = 4
best_params['metric'] = 'multi_logloss'
best_params['verbosity'] = -1

# クラスウェイトを適用したサンプルウェイトを作成
train_sample_weights = np.array([class_weight_dict[label] for label in y_train])
test_sample_weights = np.array([class_weight_dict[label] for label in y_test])

train_data = lgb.Dataset(X_train_scaled, label=y_train, weight=train_sample_weights)
test_data = lgb.Dataset(X_test_scaled, label=y_test, weight=test_sample_weights, reference=train_data)

model = lgb.train(
    best_params,
    train_data,
    num_boost_round=1000,
    valid_sets=[train_data, test_data],
    valid_names=['train', 'test'],
    callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=100)]
)

# 予測
y_pred_train = model.predict(X_train_scaled, num_iteration=model.best_iteration)
y_pred_train_classes = np.argmax(y_pred_train, axis=1)

y_pred_test = model.predict(X_test_scaled, num_iteration=model.best_iteration)
y_pred_test_classes = np.argmax(y_pred_test, axis=1)

# 全データでの予測（すでに作成済みのX_all_scaledを使用）
y_pred_all = model.predict(X_all_scaled, num_iteration=model.best_iteration)
y_pred_all_classes = np.argmax(y_pred_all, axis=1)

# 評価
print("\n[9/9] 結果を評価・保存しています...")

# ラベル名
target_names = ['low', 'middle', 'high', 'extra']

# 予測分布を表示
print("\n【予測分布】")
from collections import Counter
train_pred_dist = Counter(y_pred_train_classes)
test_pred_dist = Counter(y_pred_test_classes)
print("\n  Train予測数:")
for i, label in enumerate(['low', 'middle', 'high', 'extra']):
    count = train_pred_dist.get(i, 0)
    pct = count / len(y_pred_train_classes) * 100
    print(f"    {label}: {count} ({pct:.1f}%)")

print("\n  Test予測数:")
for i, label in enumerate(['low', 'middle', 'high', 'extra']):
    count = test_pred_dist.get(i, 0)
    pct = count / len(y_pred_test_classes) * 100
    print(f"    {label}: {count} ({pct:.1f}%)")

# Classification Report
print("\n" + "=" * 100)
print("【Train データの評価】")
print("=" * 100)
train_report = classification_report(y_train, y_pred_train_classes, target_names=target_names)
print(train_report)

print("\n" + "=" * 100)
print("【Test データの評価】")
print("=" * 100)
test_report = classification_report(y_test, y_pred_test_classes, target_names=target_names)
print(test_report)

print("\n" + "=" * 100)
print("【全データの評価】")
print("=" * 100)
all_report = classification_report(y_encoded, y_pred_all_classes, target_names=target_names)
print(all_report)

# レポートをファイルに保存
with open(os.path.join(results_folder, 'classification_report_train.txt'), 'w', encoding='utf-8') as f:
    f.write(train_report)

with open(os.path.join(results_folder, 'classification_report_test.txt'), 'w', encoding='utf-8') as f:
    f.write(test_report)

with open(os.path.join(results_folder, 'classification_report_all.txt'), 'w', encoding='utf-8') as f:
    f.write(all_report)

print("\n  Classification Reportsを保存しました")

# Confusion Matrix
cm_train = confusion_matrix(y_train, y_pred_train_classes)
cm_test = confusion_matrix(y_test, y_pred_test_classes)
cm_all = confusion_matrix(y_encoded, y_pred_all_classes)

# Confusion Matrixをプロット（Plotly）
def plot_confusion_matrix(cm, title, filename):
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=target_names,
        y=target_names,
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 16},
        hovertemplate='Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Predicted',
        yaxis_title='Actual',
        width=600,
        height=600
    )
    
    fig.write_html(os.path.join(results_folder, filename))

plot_confusion_matrix(cm_train, 'Confusion Matrix - Train Data', 'confusion_matrix_train.html')
plot_confusion_matrix(cm_test, 'Confusion Matrix - Test Data', 'confusion_matrix_test.html')
plot_confusion_matrix(cm_all, 'Confusion Matrix - All Data', 'confusion_matrix_all.html')

print("  Confusion Matrixを保存しました")

# Feature Importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importance(importance_type='gain')
}).sort_values('importance', ascending=False)

feature_importance.to_csv(os.path.join(results_folder, 'feature_importance.csv'), encoding='utf-8-sig', index=False)

# Feature Importanceをプロット（上位20）
fig_importance = px.bar(
    feature_importance.head(20),
    x='importance',
    y='feature',
    orientation='h',
    title='Feature Importance (Top 20)',
    labels={'importance': 'Importance (Gain)', 'feature': 'Feature'}
)
fig_importance.update_layout(height=600, yaxis={'categoryorder': 'total ascending'})
fig_importance.write_html(os.path.join(results_folder, 'feature_importance.html'))

print("  Feature Importanceを保存しました")

# Accuracy比較
accuracies = {
    'Dataset': ['Train', 'Test', 'All'],
    'Accuracy': [
        accuracy_score(y_train, y_pred_train_classes),
        accuracy_score(y_test, y_pred_test_classes),
        accuracy_score(y_encoded, y_pred_all_classes)
    ],
    'F1 Score (Weighted)': [
        f1_score(y_train, y_pred_train_classes, average='weighted'),
        f1_score(y_test, y_pred_test_classes, average='weighted'),
        f1_score(y_encoded, y_pred_all_classes, average='weighted')
    ]
}

accuracy_df = pd.DataFrame(accuracies)
accuracy_df.to_csv(os.path.join(results_folder, 'accuracy_summary.csv'), encoding='utf-8-sig', index=False)

print("\n【精度サマリー】")
print(accuracy_df.to_string(index=False))

# 精度比較グラフ
fig_acc = go.Figure()
fig_acc.add_trace(go.Bar(
    x=accuracy_df['Dataset'],
    y=accuracy_df['Accuracy'],
    name='Accuracy',
    marker_color='lightblue'
))
fig_acc.add_trace(go.Bar(
    x=accuracy_df['Dataset'],
    y=accuracy_df['F1 Score (Weighted)'],
    name='F1 Score (Weighted)',
    marker_color='lightcoral'
))

fig_acc.update_layout(
    title='Model Performance Comparison',
    xaxis_title='Dataset',
    yaxis_title='Score',
    barmode='group',
    height=500,
    width=800
)
fig_acc.write_html(os.path.join(results_folder, 'accuracy_comparison.html'))

print("  精度比較グラフを保存しました")

# モデルを保存
model.save_model(os.path.join(results_folder, 'lgbm_model.txt'))
print("  モデルを保存しました")

# 予測結果を保存
results_df = df_clean.copy()
results_df['predicted_category_encoded'] = y_pred_all_classes
results_df['predicted_category'] = [target_names[i] for i in y_pred_all_classes]
results_df['is_correct'] = (results_df['sales_category'] == results_df['predicted_category'])

results_df.to_csv(os.path.join(results_folder, 'prediction_results.csv'), encoding='utf-8-sig', index=False)
print("  予測結果を保存しました")

print("\n" + "=" * 100)
print("【完了】")
print("=" * 100)
print(f"\n生成されたファイル（{results_folder}フォルダ）:")
print("  1. classification_report_train.txt - Trainデータの分類レポート")
print("  2. classification_report_test.txt - Testデータの分類レポート")
print("  3. classification_report_all.txt - 全データの分類レポート")
print("  4. confusion_matrix_train.html - Trainデータの混同行列")
print("  5. confusion_matrix_test.html - Testデータの混同行列")
print("  6. confusion_matrix_all.html - 全データの混同行列")
print("  7. feature_importance.csv - 特徴量重要度")
print("  8. feature_importance.html - 特徴量重要度グラフ")
print("  9. accuracy_summary.csv - 精度サマリー")
print(" 10. accuracy_comparison.html - 精度比較グラフ")
print(" 11. lgbm_model.txt - 訓練済みモデル")
print(" 12. prediction_results.csv - 予測結果")
print("\n" + "=" * 100)

