# Kaggle 1142 Medical Condition Classification — 競賽計畫

## 競賽概覽

| 項目 | 內容 |
|------|------|
| 目標 | 利用醫學文獻摘要（Abstracts）預測 5 種疾病類別 |
| 評估指標 | **Macro F1-Score**（各類別 F1 的平均，對類別不平衡敏感） |
| 核心策略 | 無監督學習（Zero-shot & Similarity-based），遵循官方論文暗示 |
| 參考論文 | Schopf et al. (2022) *Evaluating Unsupervised Text Classification* |

### 類別標籤
| Label | 英文 | 說明 |
|-------|------|------|
| 1 | neoplasms | 腫瘤 |
| 2 | digestive system diseases | 消化系統疾病 |
| 3 | nervous system diseases | 神經系統疾病 |
| 4 | cardiovascular diseases | 心血管疾病 |
| 5 | general pathological conditions | 一般病理狀況 |

---

## 整體策略藍圖

```
Phase 1: 無監督基準測試 (Unsupervised Baseline)
    ├── Method A: Zero-shot (NLI-based)
    └── Method B: Similarity-based (Sentence-BERT)
            ↓
Phase 2: 標籤描述工程 (Label Description Engineering)
            ↓
Phase 3: 模型選型與 Ensemble
            ↓
Phase 4: （可選）弱監督微調 (Weak Supervision Fine-tuning)
            ↓
Phase 5: 最終提交與後處理
```

---

## Phase 1：無監督基準測試

### 目標
- 在不使用任何標籤的情況下，評估兩種無監督策略的上限潛力。
- 使用 `kaggle_trainset.csv` 前 1,000 筆作為驗證集（有 ground truth 可對比）。

### Method A：Zero-shot Classification（NLI-based）

**核心概念**：將分類問題轉化為自然語言推論（NLI）任務。

- **工具**：Hugging Face `pipeline("zero-shot-classification")`
- **候選模型**（依優先順序）：
  1. `facebook/bart-large-mnli` — 通用強基準
  2. `cross-encoder/nli-deberta-v3-large` — 推理能力更強
  3. `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` — 多資料集訓練
- **輸入**：Abstract 文字 + 5 個類別的英文名稱作為 `candidate_labels`
- **批次處理**：每批 16 筆，避免 OOM

### Method B：Similarity-based（Sentence-BERT）

**核心概念**：計算 Abstract embedding 與各類別描述 embedding 的 Cosine Similarity，取最高者。

- **工具**：`sentence-transformers`
- **候選模型**（依優先順序）：
  1. `pritamdeka/S-PubMedBert-MS-MARCO` — 醫學領域專用
  2. `sentence-transformers/all-mpnet-base-v2` — 通用強基準
  3. `BAAI/bge-large-en-v1.5` — 近年強力通用模型
- **類別表示**：將每個類別的英文描述（可擴增為多句）encode 為向量
- **批次處理**：每批 64 筆 encode，最後矩陣化計算相似度

### 評估輸出
```
- Macro F1-Score
- Classification Report（每類別的 Precision / Recall / F1）
- Confusion Matrix 視覺化
```

**決策點**：若任一方法在前 1,000 筆達到 F1 > 0.5，繼續優化；若兩者均 < 0.4，考慮提前進入 Phase 4。

---

## Phase 2：標籤描述工程（Label Description Engineering）

**核心洞察**：Zero-shot 和 Similarity-based 的效果，高度依賴「如何描述類別」。

### 策略
1. **擴增類別描述**：將單一標籤名稱擴充為多句醫學描述（e.g., 加入同義詞、典型症狀）
2. **多模板集成**：設計 3-5 種 prompt 模板，對 NLI 方法取 soft vote
3. **類別錨點文本**：從 trainset 中找出每類最「典型」的摘要，作為 SBERT 比較的錨點（Few-shot Retrieval）

### 範例（類別 1：neoplasms）
```
基本版: "neoplasms"
擴增版: "cancer, tumor, neoplasm, malignant growth, oncology, carcinoma"
句子版: "This paper discusses neoplasms, including various types of cancer, 
         tumors, and malignant conditions affecting different body systems."
```

**決策點**：測試不同描述版本的 F1，選擇最佳組合。

---

## Phase 3：模型選型與 Ensemble

### 模型組合策略
| 組合 | Method A 模型 | Method B 模型 | Ensemble |
|------|--------------|--------------|---------|
| 組合 1 | bart-large-mnli | S-PubMedBert | 加權投票 |
| 組合 2 | deberta-v3-large | bge-large-en | Soft voting |
| 組合 3 | — | 多 SBERT 模型 | 平均相似度 |

### Ensemble 方法
- **Hard voting**：多數決
- **Soft voting**：對每類的機率分數取平均後 argmax
- **加權 Ensemble**：根據各模型在前 1,000 筆的 F1 決定權重

---

## Phase 4（可選）：弱監督微調

**前提條件**：Phase 1-3 的最佳結果仍不理想（F1 < 0.7）

### 策略
1. **使用 trainset 進行監督微調**：
   - 以 Phase 1 最佳模型為起點
   - 直接在 `kaggle_trainset.csv` 全量資料（12,994 筆）上進行 fine-tune
   - 模型：`microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract` 或 `allenai/scibert_scivocab_uncased`
2. **K-Fold 交叉驗證**：5-fold，選最佳 checkpoint
3. **Pseudo-labeling**：用高信心的 test set 預測結果擴增訓練資料

---

## Phase 5：最終提交

### 提交流程
1. 用最佳模型對 `kaggle_testset.csv`（1,444 筆）生成預測
2. 映射為 label 1-5
3. 填入 `kaggle_testset_submission.csv` 的 `label` 欄位
4. 驗證提交格式（index 對齊、無缺值）

### 後處理策略
- **類別分布校正**：若預測分布與 trainset 差異過大，考慮 calibration
- **低信心樣本審查**：手動確認 top-5 probability 差異極小的樣本

---

## 執行時程

| Phase | 優先級 | 預估時間 | 完成標誌 |
|-------|-------|---------|---------|
| Phase 1 | ★★★ 必做 | 2-4 小時 | 兩方法的 F1 均已測出 |
| Phase 2 | ★★★ 必做 | 1-2 小時 | 最佳標籤描述確定 |
| Phase 3 | ★★☆ 重要 | 2-3 小時 | Ensemble F1 > Single model |
| Phase 4 | ★☆☆ 選做 | 4-6 小時 | 視 Phase 1-3 結果決定 |
| Phase 5 | ★★★ 必做 | 0.5 小時 | 提交完成 |

---

## 環境需求

```bash
# 必要套件
pip install transformers sentence-transformers
pip install torch  # 建議 CUDA 版本
pip install scikit-learn pandas numpy
pip install matplotlib seaborn  # 視覺化
```

### 建議硬體
- GPU 8GB+ VRAM（用於 bart-large / deberta-large）
- 若無 GPU，Method B (SBERT) 在 CPU 也可接受，Method A 建議用 Kaggle Notebook GPU

---

## 檔案結構規劃

```
Kaggle_competition/
├── Plan.md                          # 本計畫書
├── data/
│   ├── kaggle_trainset.csv
│   ├── kaggle_testset.csv
│   └── kaggle_testset_submission.csv
├── phase1_baseline.py               # Phase 1 程式碼
├── phase2_label_engineering.py      # Phase 2 程式碼
├── phase3_ensemble.py               # Phase 3 程式碼
├── phase4_finetune.py               # Phase 4（選做）
├── phase5_submit.py                 # 生成最終提交檔
└── outputs/
    ├── phase1_results.csv
    ├── phase3_ensemble_results.csv
    └── final_submission.csv
```

---

## 風險與對策

| 風險 | 可能性 | 對策 |
|------|--------|------|
| Zero-shot F1 過低（< 0.4） | 中 | 優先改用監督式微調（Phase 4） |
| GPU 記憶體不足 | 中 | 減小 batch size，或改用較小模型 |
| 類別 5（general pathological）難以區分 | 高 | 強化類別描述，加入負向 prompt |
| 測試集分布與訓練集差異大 | 低 | 加入 calibration 後處理 |

---

*計畫制定日期：2026-05-18*
*下一步：執行 Phase 1，撰寫 `phase1_baseline.py`*
