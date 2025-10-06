# 売上予測モデル - テキストエンベディングを使用

## 概要
このプロジェクトは、レストランの売上（target_amount_tableau）を予測するモデルです。

## 特徴

### 🎯 テキストエンベディングによる文字列変数の処理
単純なダミー変数化ではなく、**Sentence Transformers**を使用して意味的な類似性を捉えます。

#### 処理する文字列変数：
1. **CUISINE_CAT_origin**: 料理カテゴリ
   - 「カレー」「カレーライス」「インド料理」などが意味的に近いベクトルになる
   
2. **SAKAYA_DEALER_NAME**: 店舗名
   - チェーン店名が互いに近いベクトルになり、ブランドの類似性を捉える
   
3. **CITY**: 都市名
   - 地理的・文化的に近い都市が近いベクトルになる
   
4. **RST**: レストラン関連情報

### 📊 使用する数値特徴量
- AVG_MONTHLY_POPULATION（平均月間人口）
- NEAREST_STATION_INFO_count（最寄り駅情報数）
- DISTANCE_VALUE（距離値）
- RATING_CNT（評価数）
- RATING_SCORE（評価スコア）
- IS_FAMILY_FRIENDLY（家族向け）
- IS_FRIEND_FRIENDLY（友人向け）
- IS_ALONE_FRIENDLY（一人向け）
- DINNER_INFO（ディナー情報）
- LUNCH_INFO（ランチ情報）
- DINNER_PRICE（ディナー価格）
- LUNCH_PRICE（ランチ価格）
- HOLIDAY（休日情報）
- HOME_PAGE_URL（ホームページURL有無）
- PHONE_NUM（電話番号有無）
- NUM_SEATS（座席数）
- AGE_RESTAURANT（レストラン年齢）
- CITY_count（都市カウント）

## インストール

```bash
pip install -r requirements.txt
```

## 実行方法

```bash
python sales_prediction.py
```

## 出力

実行すると以下のファイルが生成されます：

1. **sales_prediction_results.png**: 予測精度の可視化
   - 予測値 vs 実測値のプロット
   - 特徴量重要度の分析

2. **prediction_results.csv**: 予測結果の詳細
   - 実測値と予測値の比較
   - 予測誤差

3. **feature_importance.csv**: 特徴量重要度
   - どの特徴量が予測に重要かを示す

## テキストエンベディングの仕組み

### なぜテキストエンベディングが有効か？

1. **意味的類似性の保持**
   - 「カレー」と「カレーライス」は異なる文字列だが、意味は近い
   - ダミー変数では完全に独立した変数として扱われてしまう
   - エンベディングでは意味的に近いものは近いベクトルになる

2. **次元の効率性**
   - 100種類の料理カテゴリをダミー変数化すると100次元必要
   - エンベディングでは384次元（モデルによる）で全ての意味情報を表現

3. **未知のカテゴリへの対応**
   - 新しい料理カテゴリが出現しても、意味が近ければ適切に処理できる

### 使用モデル
- **paraphrase-multilingual-MiniLM-L12-v2**
  - 多言語対応（日本語も含む）
  - 軽量で高速
  - 384次元のベクトルを生成

## 機械学習モデル

2つのモデルを使用して予測を行います：

1. **Random Forest Regressor**
   - アンサンブル学習
   - 過学習に強い
   - 特徴量重要度を分析可能

2. **Gradient Boosting Regressor**
   - 高精度
   - 順次誤差を改善していく

## 改善のアイデア

1. **より高度な日本語モデルの使用**
   - `cl-tohoku/bert-base-japanese-v2`
   - `sonoisa/sentence-bert-base-ja-mean-tokens-v2`

2. **Target Encodingとの組み合わせ**
   - エンベディングとターゲットエンコーディングを併用

3. **ディープラーニングモデル**
   - テキストと数値を同時に処理するニューラルネットワーク

4. **特徴量エンジニアリング**
   - チェーン店フラグの作成（SAKAYA_DEALER_NAMEから）
   - 料理カテゴリのグルーピング（CUISINE_CAT_originから）

## ライセンス
MIT

