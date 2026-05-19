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

### 關鍵發現：衝突標籤（38% 資料）

訓練集中同一 `condition` 文本可能出現多個不同 `label`（共 2,400 個衝突文本）。

**正確處理方式（多數投票）**：
```python
label_counts = train_df.groupby(["condition", "label"]).size().reset_index(name="cnt")
majority_idx = label_counts.groupby("condition")["cnt"].idxmax()
df_clean = label_counts.loc[majority_idx, ["condition", "label"]].reset_index(drop=True)
# 結果：12,994 筆 → 10,395 筆唯一文本（保留衝突文本，取多數標籤）
```

**錯誤方式（已棄用）**：直接移除所有衝突文本 → 只剩 7,995 筆 → OOF 虛高，LB 暴跌。

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
| Kaggle KFold v1 | BiomedBERT-large 3-Fold（移除衝突） | OOF=0.818 | **0.574** ⚠️ 異常 | `kaggle_large_kfold.py` |
| **訓練中** | BiomedBERT-large v2（Kaggle） | — | — | `kaggle_large_kfold_v2.py` |
| **訓練中** | DeBERTa-v3-large（Colab A100） | val=0.7589 | — | `colab_deberta_kfold.py` |

**目前最佳提交：0.6643（Phase 5 Ensemble）**
**KFold v1 LB 異常原因：訓練只用 7,995 乾淨樣本，OOF 虛高，test 遇困難樣本大幅下降**

---

## 4. 現有腳本說明

### 訓練腳本

| 腳本 | 用途 | 平台 | 狀態 |
|------|------|------|------|
| `kaggle_large_kfold_v2.py` | **主力** BiomedBERT-large 2-Fold | Kaggle T4×2 | 訓練中 |
| `colab_deberta_kfold.py` | **主力** DeBERTa-v3-large 2-Fold | Colab A100/T4 | 訓練中 |
| `kaggle_deberta_kfold.py` | DeBERTa-v3-large（Kaggle 版） | Kaggle T4×2 | 備用 |
| `local_base_kfold.py` | BiomedBERT-base 3-Fold | RTX 3060 / Colab | 備用 |

### 後處理 / 分析腳本

| 腳本 | 用途 |
|------|------|
| `kaggle_ensemble_v2.py` | BiomedBERT + DeBERTa 軟機率 Ensemble，OOF grid search |
| `kaggle_threshold_search.py` | Label-5 單一閾值 grid search（已測試，效果不佳） |
| `local_threshold_simulation.py` | 本地 TF-IDF 代理模型驗證閾值策略 |

---

## 5. 核心技術說明

### 5.1 模型架構

| 模型 | 用途 | 特性 |
|------|------|------|
| `microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract` | 主力 Kaggle | 生醫摘要預訓練，BERT-large |
| `microsoft/deberta-v3-large` | Ensemble 互補 | Disentangled attention，通用強模型 |
| `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract` | 本地實驗 | RTX 3060 可承載 |

### 5.2 訓練超參數（v2 版，已調優）

| 參數 | BiomedBERT-large v2 | DeBERTa Colab |
|------|--------------------|--------------------|
| MAX_LEN | 384 | 384 |
| BATCH_SIZE | 8 | 8 (A100) / 4 (T4) |
| GRAD_ACCUM | 8 | 8 (A100) / 16 (T4) |
| Effective Batch | 64 | 64 |
| LR | 8e-6 | 5e-6 |
| WARMUP_RATIO | 20% | 10% |
| LABEL_SMOOTH | 0.1 | 0.1 |
| N_FOLDS | 2 | 2 |
| PATIENCE | 3 | 3 |
| EPOCHS | 8 | 8 |
| Precision | FP16 AMP | BF16 (A100) / FP32 (T4) |

### 5.3 DeBERTa AMP 相容性問題

DeBERTa-v3 的 disentangled attention 在 PyTorch FP16 AMP（GradScaler）下會產生 FP16 梯度，導致 `ValueError: Attempting to unscale FP16 gradients`。

**解法**：
- A100：使用 `torch.autocast(dtype=torch.bfloat16)`，不需 GradScaler
- T4：`USE_BF16=False`，純 FP32 訓練
- 兩種情況都在 loss 計算前加 `logits.float()`

### 5.4 損失函數

```python
cw = compute_class_weight("balanced", classes=np.arange(5), y=fold_train_labels)
class_weights = torch.tensor(cw, dtype=torch.float).to(DEVICE)
loss_fn = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
```

### 5.5 K-Fold Ensemble 流程

```
訓練資料（10,395 筆多數投票清洗後）
    → StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    → 每個 fold 訓練一個模型 → 儲存最佳 checkpoint（按 val F1）
    → OOF 預測：每筆訓練資料由不包含它的 fold 預測
    → Test 預測：2 fold 軟機率平均（soft voting）
    → 儲存 oof_probs.npy + test_probs.npy
    → kaggle_ensemble_v2.py 合併兩模型
```

### 5.6 Colab 斷點續訓機制

`colab_deberta_kfold.py` 在每個 fold 完成後自動：
1. 存 `oof_probs_deberta.npy` + `test_probs_deberta.npy` 到 Google Drive
2. 記錄已完成的 fold 到 `completed_folds.txt`
3. 重新執行時自動跳過已完成的 fold

---

## 6. 已驗證無效的策略

### Label-5 Confidence Threshold（方案 B）
若 max(P(label1-4)) < τ 則改預測 label 5。本地 TF-IDF 驗證：
- 最佳 τ=0.30 僅帶來 -0.0040（負收益）
- 低信心的 1-4 預測中只有 29.7% 真的是 label 5
- 結論：不採用

---

## 7. 競賽規則（絕對不能違反）

1. **禁止**：從網路或任何外部管道取得測試集的正確答案
2. **禁止**：利用測試集答案分析特徵用於預測
3. **違者成績歸零**

---

## 8. 下一步建議（按優先順序）

### 立即（訓練完成後）

1. **下載 npy 檔**
   - Kaggle：`test_probs_biomedbert.npy` + `oof_probs_biomedbert.npy`
   - Colab/Drive：`test_probs_deberta.npy` + `oof_probs_deberta.npy`

2. **跑 Ensemble**
   - 上傳 4 個 npy 到 Kaggle dataset
   - 執行 `kaggle_ensemble_v2.py` → 提交 `submission_ensemble_v2.csv`

### 短期優化（不需重訓）

3. **Per-class 決策閾值**：用 scipy.optimize 在 OOF 上為每個類別找最佳閾值，加進 ensemble 腳本

### 中期優化（需重訓）

4. **FGM 對抗性訓練**：在現有訓練 loop 加入 FGM（~20 行），預期穩定提升 +0.005~0.01

5. **BioLinkBERT-large**：`michiyasunaga/BioLinkBERT-large`，加入引用連結預訓練，作為第三個 ensemble 成員

---

## 9. 環境需求

```bash
pip install transformers torch scikit-learn pandas numpy tqdm
pip install sentencepiece protobuf  # DeBERTa tokenizer 必需
```

| 平台 | 硬體 | 用途 |
|------|------|------|
| Kaggle | T4 × 2 (各 15.6GB) | BiomedBERT-large 訓練 |
| Google Colab Pro | A100 (40GB) | DeBERTa-v3-large 訓練 |
| 本地 | RTX 3060 (12GB) | 快速實驗 / base 模型 |

---

*文件更新日期：2026-05-19*
