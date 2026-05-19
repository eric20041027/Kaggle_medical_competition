# 競賽規則 Competition Rules

## 中文版

本次競賽採用公開資料集，請參賽者務必遵守以下規定：

1. **嚴禁**搜尋網際網路或透過任何其他管道取得資料集的標籤資料並用於預測結果上傳評分。
2. **嚴禁**從測試資料集之正確答案中分析取得特徵並用於預測結果上傳評分。
3. 違反上述規定者，成績將以零分計算。

## English Version

This competition utilizes a public dataset. All participants must strictly adhere to the following regulations:

1. It is **strictly forbidden** to search the internet or use any other channels to obtain the ground truth for the dataset and use it in prediction result submissions for scoring.
2. It is **strictly forbidden** to engage in data leakage by analyzing or extracting features from the ground truth of the test set and using them in prediction result submissions for scoring.
3. Any participant found in violation of the aforementioned rules will have their score nullified.

---

## 對我們策略的影響

- **允許**：使用 `kaggle_trainset.csv` 的標籤進行模型評估與監督式訓練（這是合法的訓練資料）。
- **允許**：無監督/Zero-shot 方法（不依賴任何外部標籤）。
- **禁止**：從網路搜尋 `kaggle_testset.csv` 對應的真實標籤。
- **禁止**：利用 test set 的答案反推任何特徵或模型調整。
