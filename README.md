# Kaggle 1142 Medical Condition Classification

## 競賽概述

利用未經預處理的醫學文獻摘要（Abstracts），預測病患的 5 種疾病類別。

- **評估指標**：Macro F1-Score
- **參考論文**：Schopf et al. (2022) *Evaluating Unsupervised Text Classification: Zero-shot and Similarity-based Approaches*

## 類別標籤

| Label | 疾病類別 |
|-------|---------|
| 1 | neoplasms（腫瘤） |
| 2 | digestive system diseases（消化系統疾病） |
| 3 | nervous system diseases（神經系統疾病） |
| 4 | cardiovascular diseases（心血管疾病） |
| 5 | general pathological conditions（一般病理狀況） |

## 資料集

| 檔案 | 筆數 | 說明 |
|------|------|------|
| `kaggle_trainset.csv` | 12,994 | 含 label、condition 欄位 |
| `kaggle_testset.csv` | 1,444 | 僅含 condition 欄位 |
| `kaggle_testset_submission.csv` | 1,444 | 提交範本 |

### 資料清洗發現

- 38% 的訓練資料（4,999 筆）存在**衝突標籤**（同一文本對應不同類別）
- 清洗後保留 7,995 筆乾淨樣本
- `general pathological conditions` 衝突比例最高（-45%），為模型最難預測的類別

## 實驗進程與結果

| Phase | 方法 | Val Macro F1 |
|-------|------|-------------|
| Phase 1 | Zero-shot NLI（BART-large-MNLI） | 0.5930 |
| Phase 1 | Similarity-based（S-PubMedBert） | 0.4708 |
| Phase 2 | DeBERTa-v3-large NLI（v4_contrast） | 0.6001 |
| Phase 2 | bge-large-en-v1.5 SBERT（v3_sentences） | 0.5655 |
| Phase 2 | Ensemble（w_A=0.3, w_B=0.7） | 0.6102 |
| Phase 4 | BiomedBERT-base Fine-tune | 0.6397 |
| Phase 5 | Phase4 + Phase2 Ensemble | 0.6643 |
| Kaggle | BiomedBERT-large（T4×2） | 0.6558 |

## 檔案結構

```
├── 本地實驗腳本
│   ├── phase1_baseline.py          # Zero-shot + SBERT 基準測試
│   ├── phase2_label_engineering.py # 標籤描述工程 + 模型掃描
│   ├── phase4_finetune_base_v1.py  # BiomedBERT-base 微調（v1）
│   ├── phase4_finetune_base_v2.py  # BiomedBERT-base 微調（v2，修正過擬合）
│   ├── phase5_ensemble_submit.py   # Ensemble + 生成提交檔
│   └── local_base_kfold.py         # BiomedBERT-base 3-Fold（RTX 3060）
│
├── Kaggle 提交腳本
│   ├── kaggle_base_single.py       # BiomedBERT-base，單模型
│   ├── kaggle_large_single.py      # BiomedBERT-large，單模型（T4×2）
│   └── kaggle_large_kfold.py       # BiomedBERT-large，3-Fold Ensemble ← 主力
│
├── 資料集
│   ├── kaggle_trainset.csv
│   ├── kaggle_testset.csv
│   └── kaggle_testset_submission.csv
│
├── outputs/                        # 訓練輸出（模型權重不含在版控中）
├── Plan.md                         # 競賽計畫書
└── Rule.md                         # 競賽規則
```

## 核心策略

### 1. 無監督基準（Phase 1-2）
遵循官方參考論文，先以 Zero-shot NLI 和 Sentence-BERT Similarity 建立基準，並透過標籤描述工程（4 種描述版本）尋找最佳 prompt。

### 2. 監督式微調（Phase 4）
使用 `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract`，針對過擬合問題做出以下修正：
- LR：`2e-5 → 1e-5`
- LR Schedule：`linear → cosine`
- Label Smoothing：`0.1`
- Early Stopping：`patience=2`

### 3. K-Fold Ensemble（主力）
- 3-Fold Stratified K-Fold
- 移除衝突標籤樣本（-38%噪音）
- 軟機率平均（soft voting）
- Kaggle 版使用 BiomedBERT-large + T4×2 雙 GPU + AMP

## 環境需求

```bash
pip install transformers sentence-transformers torch
pip install scikit-learn pandas numpy matplotlib seaborn tqdm
```

## 競賽規則

- 嚴禁從網路取得測試集的正確答案
- 嚴禁利用測試集答案分析特徵用於預測
- 違者成績歸零
