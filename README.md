# 売上予測プロジェクト

## 概要
このプロジェクトは、レストランの売上（`target_amount_tableau`）を予測する機械学習モデルです。データの前処理からクラスタリング、複数の回帰モデルの構築まで、一連の分析パイプラインを提供します。

## 📋 プロジェクト構成

本プロジェクトは以下の順序で実行します：

### 1. 料理カテゴリのクラスタリング（`cuisine_clustering.py`）

**目的**: 料理カテゴリ（`CUISINE_CAT_origin`）を売上パターンに基づいてグループ化

**処理内容**:
- 10店舗以下のカテゴリを'other'にまとめて前処理
- 各カテゴリの売上統計（平均、標準偏差）を特徴量として使用
- K-Meansクラスタリングを実施（エルボー法・シルエット分析で最適なクラスタ数を決定）
- クラスタIDを中央値の昇順で再割り当て

**出力**:
- `cuisine_clustering_results/`
  - `cuisine_clustering_analysis.html`: エルボー法とシルエット分析の可視化
  - `cuisine_clustering_result.html`: クラスタリング結果の散布図
  - `cuisine_clustering_results.csv`: 各カテゴリのクラスタ割当結果
  - `cuisine_cluster_statistics.csv/.xlsx`: クラスタ別統計表

---

### 2. 店舗のクラスタリング（`final_store_clustering.py`）

**目的**: 店舗を特性に基づいてグループ化し、類似店舗を識別

**処理内容**:
- 以下の特徴量を使用:
  - `AVG_MONTHLY_POPULATION`: 平均月間人口
  - `NUM_SEATS`: 座席数
  - `DINNER_PRICE`: ディナー価格
  - `IS_FAMILY_FRIENDLY`: ファミリー向けフラグ
  - `cuisine_cluster_id`: 料理クラスタID（手順1の結果）
- K-Meansクラスタリングで店舗をグループ化
- クラスタIDを売上中央値の昇順で再割り当て

**出力**:
- `final_store_clustering_results/`
  - `optimal_k_analysis.html`: 最適クラスタ数の分析
  - `cluster_scatter_*.html`: 各種散布図（座席数vs価格、人口vs売上など）
  - `stores_with_cluster.csv`: 各店舗のクラスタ割当結果
  - `cluster_statistics.csv/.xlsx`: クラスタ別統計表

---

### 3-1. 基本的な売上回帰モデル（`sales_regression.py`）

**目的**: LightGBMとRandom Forestを使用した全店舗の売上予測

**処理内容**:
- 目的変数を対数変換（`log(target_amount_tableau)`）
- 37個の基本特徴量 + クラスタ別中央値（2特徴量）を使用
- Optunaによるハイパーパラメータ最適化
- Train/Test/All データセットでの評価（RMSE, R², MAPE）
- 予測値を`stores_with_cluster.csv`に追加（`lgb_predicted`, `rf_predicted`）

**出力**:
- `sales_regression_results/`
  - `lightgbm_analysis.html` / `random_forest_analysis.html`: 2×2サブプロット（予測vs実測、ヒストグラム、APE、特徴量重要度）
  - `model_comparison.csv`: モデル比較結果
  - `test_predictions.csv`: テスト予測結果
  - `feature_importance_*.csv`: 特徴量重要度

**特徴**:
- インタラクティブなホバー情報（店舗名、都市、料理カテゴリ、各種特徴量を表示）
- 売上単位を「本/月」で表示

---

### 3-2. エンベディング追加モデル（`sales_regression_with_embeddings.py`）

**目的**: カテゴリ変数のTarget EncodingとFrequency Encodingを追加して精度向上

**処理内容**:
- 基本モデル（3-1）に以下の特徴量を追加:
  - `cuisine_origin_median_log`: 料理カテゴリ別の売上中央値（Target Encoding）
  - `city_median_log`: 都市別の売上中央値（Target Encoding）
  - `cuisine_origin_frequency`: 料理カテゴリの出現頻度（Frequency Encoding）
  - `city_frequency`: 都市の出現頻度（Frequency Encoding）
- Data leakage防止のため、訓練データのみで統計量を計算

**出力**:
- `sales_regression_results_with_embeddings/`
  - 3-1と同様の出力ファイル
  - `encoding_maps.json`: カテゴリ変数のエンコーディングマップ（再利用可能）

**期待効果**:
- 料理カテゴリと都市の情報をより効果的に活用
- 基本モデルよりも高い予測精度

---

### 3-3. クラスタ別回帰モデル（`sales_regression_by_cluster.py`）

**目的**: 店舗クラスタごとに個別のモデルを構築し、クラスタ特性に応じた予測

**処理内容**:
- 各`store_cluster`ごとに独立したLightGBM/Random Forestモデルを訓練
- データ数30店舗未満のクラスタはスキップ
- クラスタ内での`cuisine_cluster_id`別中央値を特徴量として追加

**出力**:
- `sales_regression_by_cluster_results/`
  - `cluster_X/`: 各クラスタの詳細結果フォルダ
    - `lightgbm_analysis.html` / `random_forest_analysis.html`
    - `model_comparison.csv`
    - `test_predictions.csv`
    - `feature_importance_*.csv`
  - `all_clusters_summary.csv`: 全クラスタの結果サマリー
  - `clusters_comparison.html`: クラスタ間比較グラフ

**利点**:
- クラスタ特有のパターンを捉えた予測が可能
- 各クラスタに最適化されたモデル

---

### 3-4. 料理カテゴリ別回帰モデル（`sales_regression_by_cuisine.py`）

**目的**: 料理カテゴリ（`CUISINE_CAT_origin`）ごとに個別のモデルを構築

**処理内容**:
- データ数30店舗以上の料理カテゴリのみ処理
- 各カテゴリごとに独立したLightGBM/Random Forestモデルを訓練
- カテゴリ特有の特徴を活かした予測

**出力**:
- `sales_regression_by_cuisine_results/`
  - `cuisine_XXX_YYY/`: 各カテゴリの詳細結果フォルダ（52カテゴリ）
    - `prediction_comparison.html`: LightGBMとRandom Forestの比較
    - `model_comparison.csv`
    - `test_predictions.csv`
  - `all_cuisines_summary.csv`: 全カテゴリの結果サマリー
  - `cuisines_comparison_top20.html`: 上位20カテゴリの比較グラフ

**利点**:
- 料理カテゴリ特有の売上パターンを詳細に分析
- カテゴリごとの最適なモデル

---

## 🚀 実行方法

### 前提条件
```bash
pip install -r requirements.txt
```

### 実行順序
```bash
# 1. 料理カテゴリのクラスタリング
python cuisine_clustering.py

# 2. 店舗のクラスタリング
python final_store_clustering.py

# 3-1. 基本的な回帰モデル
python sales_regression.py

# 3-2. エンベディング追加モデル
python sales_regression_with_embeddings.py

# 3-3. クラスタ別回帰モデル
python sales_regression_by_cluster.py

# 3-4. 料理カテゴリ別回帰モデル
python sales_regression_by_cuisine.py
```

---

## 📊 使用する特徴量

### 基本特徴量（37個）
- `AVG_MONTHLY_POPULATION`: 平均月間人口
- `NEAREST_STATION_INFO_count`: 最寄り駅情報数
- `DISTANCE_VALUE`: 距離値
- `RATING_CNT`: 評価数
- `RATING_SCORE`: 評価スコア
- `IS_FAMILY_FRIENDLY` / `IS_FRIEND_FRIENDLY` / `IS_ALONE_FRIENDLY`: 客層フラグ
- `DINNER_INFO` / `LUNCH_INFO`: 食事情報
- `DINNER_PRICE` / `LUNCH_PRICE`: 価格情報
- `HOLIDAY` / `HOME_PAGE_URL` / `PHONE_NUM`: その他の情報
- `NUM_SEATS`: 座席数
- `AGE_RESTAURANT`: レストラン年齢
- `CITY_count`: 都市カウント
- `CUISINE_CAT_1_*`: 料理カテゴリダミー変数（14種類）
- `rate_count` / `seats_rate_count`: 派生特徴量
- `cuisine_cluster_id` / `store_cluster`: クラスタID

### 追加特徴量（エンベディングモデル）
- `cuisine_cluster_median_log` / `store_cluster_median_log`: クラスタ別中央値
- `cuisine_origin_median_log` / `city_median_log`: カテゴリ別Target Encoding
- `cuisine_origin_frequency` / `city_frequency`: Frequency Encoding

---

## 🤖 使用モデル

### LightGBM
- 勾配ブースティング手法
- Optunaによるハイパーパラメータ最適化（30 trials）
- Early Stopping（50 rounds）
- 高速かつ高精度

### Random Forest
- アンサンブル学習
- Optunaによるハイパーパラメータ最適化（20 trials）
- 過学習に強い
- 特徴量重要度の解釈が容易

---

## 📈 評価指標

- **RMSE** (Root Mean Squared Error): 予測誤差の平均
- **R²** (決定係数): モデルの説明力（1に近いほど良い）
- **MAPE** (Mean Absolute Percentage Error): パーセント誤差の平均

---

## 🎨 可視化機能

すべてのモデルで以下のインタラクティブな可視化を提供：

1. **実測値 vs 予測値の散布図**
   - 45度線（完全予測）との比較
   - ホバーで店舗詳細情報を表示

2. **実測値と予測値のヒストグラム**
   - 分布の比較

3. **実測値 vs 絶対パーセント誤差（APE）**
   - 誤差の傾向分析

4. **特徴量重要度**
   - Top 20特徴量の重要度

**ホバー情報**:
- `RST_TITLE`: 店舗名
- `CITY`: 都市
- `CUISINE_CAT_origin`: 料理カテゴリ
- `store_cluster` / `cuisine_cluster_id`: クラスタID
- `NUM_SEATS`: 座席数
- `AVG_MONTHLY_POPULATION`: 月間平均人口
- `DINNER_PRICE` / `LUNCH_PRICE`: 価格情報
- `IS_FAMILY_FRIENDLY` / `IS_FRIEND_FRIENDLY` / `IS_ALONE_FRIENDLY`: 客層フラグ
- `actual` / `predicted` / `error`: 実測値・予測値・誤差（本/月）

---

## 📝 ライセンス
MIT

