# Handout：醫學文本分類競賽 — 快速上手指南

> 給下一個接手此專案的 AI / 協作者閱讀。本文件涵蓋：專案現況、資料特性、已嘗試方法、目前最佳成績、技術細節、以及下一步建議。

---

## 1. 競賽基本資訊

| 項目 | 內容 |
|------|------|
| 競賽名稱 | Kaggle 1142 Medical Condition Classification |
| 任務 | 將醫學文獻摘要（condition 欄位）分類為 5 種疾病類別 |
| 評估指標 | **Macro F1-Score**（各類別 F1 均等平均，對類別不平衡敏感） |
| 參考論文 | Schopf et al. (2022) *Evaluating Unsupervised Text Classification* |
| 目標 | 進入前 8 名（領先榜約需 F1 ≥ 0.796） |

### 類別標籤（label 欄位為字串，非整數）

| 提交值 | 字串（訓練集 label 欄位原值） |
|--------|-------------------------------|
| 1 | `neoplasms` |
| 2 | `digestive system diseases` |
| 3 | `nervous system diseases` |
| 4 | `cardiovascular diseases` |
| 5 | `general pathological conditions` |

**重要**：`kaggle_trainset.csv` 的 `label` 欄位是**文字字串**，不是整數。所有腳本都有 `STR_TO_IDX` 映射。

---

## 2. 資料集特性（必讀）

| 檔案 | 筆數 | 說明 |
|------|------|------|
| `kaggle_trainset.csv` | 12,994 | 含 `label`、`condition` 欄位 |
| `kaggle_testset.csv` | 1,444 | 僅含 `condition` 欄位，需預測 |
| `kaggle_testset_submission.csv` | 1,444 | 提交範本，填入 `label` 欄位（整數 1-5） |

### 關鍵發現：38% 噪音標籤

```python
# 同一文本 (condition) 在訓練集中出現多個不同 label → 衝突噪音
text_label_nunique = train_df.groupby("condition")["label"].nunique()
conflict_texts     = text_label_nunique[text_label_nunique > 1].index
df_no_conflict     = train_df[~train_df["condition"].isin(conflict_texts)]
df_clean = df_no_conflict.drop_duplicates(subset="condition").reset_index(drop=True)
# 結果：12,994 → 7,995 筆（移除 4,999 筆衝突 + 重複）
```

各類別衝突移除比例：
- digestive system diseases：-49%（最高）
- general pathological conditions：-45%
- neoplasms：-30%

**所有訓練腳本均已套用此清洗邏輯。**

---

## 3. 實驗進程與成績

| Phase | 方法 | OOF / Val F1 | Kaggle LB | 腳本 |
|-------|------|-------------|-----------|------|
| Phase 1 | Zero-shot NLI（BART-large-MNLI） | 0.5930 | — | `phase1_baseline.py` |
| Phase 1 | Similarity-based（S-PubMedBert） | 0.4708 | — | `phase1_baseline.py` |
| Phase 2 | Ensemble（DeBERTa+bge-large） | 0.6102 | — | `phase2_label_engineering.py` |
| Phase 4 v1 | BiomedBERT-base Fine-tune | 0.6397 | — | `phase4_finetune_base_v1.py` |
| Phase 5 | Phase4 + Phase2 Ensemble | 0.6643 | **0.6643** | `phase5_ensemble_submit.py` |
| Kaggle v1 | BiomedBERT-large 單模型 | — | 0.6558 | `kaggle_large_single.py` |
| Kaggle KFold v1 | BiomedBERT-large 3-Fold（清洗資料） | OOF=0.818 | **0.574** ← 異常低 | `kaggle_large_kfold.py` |

**目前最佳提交：0.6643（Phase 5 Ensemble）**
**KFold v1 LB 異常原因：訓練只用 7,995 乾淨樣本，OOF 虛高，test 遇困難樣本大幅下降**
**目標：≥ 0.796（前 8 名）**

---

## 4. 現有腳本說明

### 本地實驗腳本

| 腳本 | 用途 | 環境 |
|------|------|------|
| `phase1_baseline.py` | Zero-shot NLI + SBERT 基準 | 本地 / Kaggle |
| `phase2_label_engineering.py` | 4 種標籤描述工程 + DeBERTa + bge-large | 本地 / Kaggle |
| `phase4_finetune_base_v1.py` | BiomedBERT-base 初版微調（有過擬合問題） | 本地 |
| `phase4_finetune_base_v2.py` | BiomedBERT-base 修正版（已修正過擬合）| 本地 |
| `phase5_ensemble_submit.py` | Phase4 + Phase2 Ensemble，生成提交檔 | 本地 |
| `local_base_kfold.py` | **主力（本地）** BiomedBERT-base 3-Fold K-Fold | RTX 3060 |

### Kaggle 提交腳本

| 腳本 | 用途 | 硬體需求 |
|------|------|---------|
| `kaggle_base_single.py` | BiomedBERT-base 單模型 | T4 × 1 |
| `kaggle_large_single.py` | BiomedBERT-large 單模型 | T4 × 2 |
| `kaggle_large_kfold.py` | **主力（Kaggle）** BiomedBERT-large 3-Fold K-Fold | T4 × 2 |

---

## 5. 核心技術說明

### 5.1 模型架構

**主力模型（Kaggle）**：`microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract`
- 針對生物醫學摘要預訓練的 BERT-large 變體
- 接上 `AutoModelForSequenceClassification`（5 分類頭）

**主力模型（本地）**：`microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract`
- 同上但 base 尺寸，RTX 3060（12 GB VRAM）可承載

### 5.2 訓練超參數（K-Fold 版，已調優）

| 參數 | 本地（base） | Kaggle（large） |
|------|------------|----------------|
| MAX_LEN | 256 | 256 |
| BATCH_SIZE | 16 | 8 |
| GRAD_ACCUM | 4 | 8 |
| Effective Batch | 64 | 64 |
| LR | 1e-5 | 8e-6 |
| LR Schedule | Cosine + Warmup | Cosine + Warmup |
| WARMUP_RATIO | 20% | 20% |
| LABEL_SMOOTH | 0.1 | 0.1 |
| N_FOLDS | 3 | 3 |
| PATIENCE | 2 | 2 |
| EPOCHS | 10 | 10 |

**LR 選擇原因**：2e-5 在 Epoch 1 即過擬合（F1 = 0.6397 後下降），降至 1e-5 / 8e-6 配合 cosine schedule 才穩定。

### 5.3 損失函數

```python
cw = compute_class_weight("balanced", classes=np.arange(5), y=fold_train_labels)
class_weights = torch.tensor(cw, dtype=torch.float).to(DEVICE)
loss_fn = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
```

- **Class weights**：平衡各類別樣本數不均
- **Label smoothing = 0.1**：減少對訓練集的過度自信

### 5.4 K-Fold Ensemble 流程

```
訓練資料（7,995 筆清洗後）
    → StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    → 每個 fold 訓練一個模型 → 儲存最佳 checkpoint（按 val F1）
    → OOF 預測：每筆訓練資料由不包含它的 fold 預測
    → Test 預測：3 fold 軟機率平均（soft voting）
    → argmax → 映射回 1-5 整數 → 提交
```

### 5.5 記憶體優化（large 模型）

```python
base.gradient_checkpointing_enable()   # 以計算換記憶體
if N_GPUS > 1:
    model = nn.DataParallel(base)      # T4×2 多卡並行
scaler = GradScaler(enabled=USE_AMP)   # FP16 AMP
```

每個 fold 結束後：
```python
del model, base_model, optimizer, scheduler, scaler, loss_fn
torch.cuda.empty_cache(); gc.collect()
```

### 5.6 Phase 2 無監督方法（備用）

**Method A（DeBERTa NLI）**：
- 模型：`MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`
- 最佳標籤描述版本：v4_contrast（對比式描述，明確排除其他類別）
- F1 = 0.6001

**Method B（bge-large SBERT）**：
- 模型：`BAAI/bge-large-en-v1.5`
- 最佳標籤描述版本：v3_sentences（完整句子描述 + 同義詞）
- F1 = 0.5655

**Ensemble**：w_A = 0.3、w_B = 0.7 → F1 = 0.6102

---

## 6. 競賽規則（絕對不能違反）

1. **禁止**：從網路或任何外部管道取得測試集的正確答案
2. **禁止**：利用測試集答案分析特徵用於預測
3. **違者成績歸零**

所有方法必須只使用 `kaggle_trainset.csv` 的標籤進行訓練 / 評估。

---

## 7. 下一步建議（按優先順序）

### 高優先（已建立腳本）

1. **執行 `kaggle_large_kfold_v2.py`（Kaggle T4×2）**
   - 改進點：多數投票保留衝突文本（7,995 → ~9,800）、MAX_LEN=384、PATIENCE=3、EPOCHS=8
   - 解決 v1 的 OOF/LB 落差問題

2. **執行 `kaggle_deberta_kfold.py`（Kaggle T4×2，第二個 Notebook）**
   - 模型：`microsoft/deberta-v3-large`，不同架構 → 互補錯誤
   - 輸出 `test_probs_deberta.npy` 供 ensemble

3. **執行 `kaggle_ensemble_v2.py`**
   - 合併兩模型軟機率，OOF grid search 最佳權重，生成 `submission_ensemble_v2.csv`

### 中優先

4. **Pseudo-labeling**：用 ensemble 高信心（max prob > 0.95）的 test 預測加回訓練集再訓練

5. **MAX_LEN 512**：若 T4 時間允許，試試 512 對長尾文本的影響

---

## 8. 環境需求

```bash
pip install transformers sentence-transformers torch
pip install scikit-learn pandas numpy matplotlib seaborn tqdm
```

**本地環境**：RTX 3060（12 GB VRAM）
**Kaggle 環境**：T4 × 2（各 15 GB VRAM），需開啟 Internet + GPU Accelerator

---

## 9. 輸出檔案說明

```
outputs/
├── local_kfold/
│   ├── best_fold1.pt / best_fold2.pt / best_fold3.pt   # 各 fold 最佳權重
│   ├── oof_probs.npy    # shape (7995, 5)，OOF 軟機率
│   ├── test_probs.npy   # shape (1444, 5)，測試集軟機率
│   └── submission.csv   # 最終提交檔
└── phase5/
    ├── final_submission.csv          # Phase5 Ensemble 提交
    └── submission_phase4_only.csv    # Phase4 單模型提交（備用）
```

`.pt` 和 `.npy` 已排除在 git 追蹤外（見 `.gitignore`）。

---

*文件生成日期：2026-05-19*
