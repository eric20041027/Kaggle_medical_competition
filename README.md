# Kaggle 1142 Medical Condition Classification

## 競賽概述

利用未經預處理的醫學文獻摘要（Abstracts），預測病患的 5 種疾病類別。

- **評估指標**：Macro F1-Score
- **目前最佳 LB**：**0.63970**（BiomedBERT-large，原始資料，MAX_LEN=512）

## 類別標籤

| Label | 疾病類別 |
|-------|---------|
| 1 | neoplasms（腫瘤） |
| 2 | digestive system diseases（消化系統疾病） |
| 3 | nervous system diseases（神經系統疾病） |
| 4 | cardiovascular diseases（心血管疾病） |
| 5 | general pathological conditions（一般病理狀況，catch-all） |

## 資料集

| 檔案 | 筆數 | 說明 |
|------|------|------|
| `kaggle_trainset.csv` | 12,994 | 含 label、condition 欄位（含衝突樣本） |
| `kaggle_testset.csv` | 1,444 | 僅含 condition 欄位 |
| `kaggle_testset_submission.csv` | 1,444 | 提交範本 |

### 關鍵資料洞察

- **38.5% 訓練資料（4,999 筆）存在衝突標籤**（2,400 個衝突文本）
- **原始資料 ≠ 清洗資料**：原始 12,994 筆與測試集分布一致（val/LB gap ~0.018）；清洗後 10,395 筆造成 distribution mismatch（gap > 0.13）
- **72% 衝突為「class5 vs 其他類別」**：class5 是 catch-all，由排除法定義
- **腫瘤關鍵詞 100% 可靠**：cancer/tumor/carcinoma 出現在衝突樣本時，命中 neoplasms 的準確率為 100%

## 提交歷史

| 提交 | LB | Val/OOF | 腳本 | 備註 |
|------|-----|---------|------|------|
| **BiomedBERT-large（原始資料）** | **0.63970** | 0.6581 | `kaggle_large_kfold_v2.py` | **目前最佳** |
| NLI+FGM+BiomedBERT ensemble v3 | 0.62229 | 0.7583 | `kaggle_ensemble_v3.py` | 清洗資料，gap=0.136 |
| BiomedBERT+BioLinkBERT ensemble v2 | 0.62013 | 0.7586 | `kaggle_ensemble_v2.py` | 清洗資料，gap=0.138 |
| Ensemble equal（BiomedBERT+BioLinkBERT+BioLinkBERT-FGM） | 0.61473 | — | 本地生成 | 混入清洗資料模型，被拉低 |
| Softlabel mul=0.5 | 0.60859 | 0.7683 | `colab_biomedbert_softlabel.py` | class5 bias，val 虛高 |
| BiomedBERT-large K-fold（移除衝突） | 0.574 | 0.818 | `kaggle_large_kfold.py` | 只用 7,995 筆，OOF 嚴重虛高 |
| BiomedBERT+FGM v2 | 0.56660 | 0.66 | `colab_biomedbert_fgm_v2.py` | 雙重 class5 boost bug |
| Softlabel（原始，1.9x boost） | 0.53463 | — | `colab_biomedbert_softlabel.py` | 嚴重過預測 class5 |

## Colab 訓練腳本

| 腳本 | 模型 | 資料 | 狀態 |
|------|------|------|------|
| `colab_biolinkbert_raw.py` | BioLinkBERT-large + FGM | 原始 12,994 筆 | **進行中** |
| `colab_biomedbert_fgm_v2.py` | BiomedBERT-large + FGM | 原始 12,994 筆 | 完成（LB 失敗） |
| `colab_biomedbert_softlabel.py` | BiomedBERT-large + 軟標籤 | 原始 12,994 筆 | 完成（LB 失敗） |
| `colab_biomedbert_kfold.py` | BiomedBERT-large K-fold | 清洗 10,395 筆 | 備用 |
| `colab_deberta_kfold.py` | DeBERTa-v3-large K-fold | 清洗 10,395 筆 | 備用 |

## 下一步

1. **BioLinkBERT 訓練完成後**：下載 `test_probs_biolinkbert_raw.npy`，與 `test_probs_biomedbert.npy` 做 ensemble
2. **Cascade 推論測試**：`submission_cascade_thr0.35.csv`（class5=17.4%）待提交驗證
3. **R-Drop 訓練**：針對雜訊標籤更根本的解法
4. **關鍵詞前綴注入**：對高信心詞（neoplasms 100% 準確）加文字前綴輔助訓練

## 已確認無效的方向

- **清洗資料訓練後 ensemble**：distribution mismatch 被放大
- **Soft label（KL divergence）**：模型學到「不確定 = class5」偏誤，val 虛高
- **Class5 inference multiplier > 1.0**：必然過預測 class5

## 環境需求

```bash
pip install transformers torch scikit-learn pandas numpy tqdm
```

## 競賽規則

- 嚴禁從網路取得測試集的正確答案
- 嚴禁利用測試集標籤進行 pseudo-labeling（違者成績歸零）
