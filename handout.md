# Handout：醫學文本分類競賽 — 快速上手指南

> 給下一個接手此專案的 AI / 協作者閱讀。涵蓋：專案現況、資料特性、已嘗試方法、目前最佳成績、失敗教訓、下一步建議。

---

## 1. 競賽基本資訊

| 項目 | 內容 |
|------|------|
| 競賽名稱 | Kaggle 1142 Medical Condition Classification |
| 任務 | 將醫學文獻摘要（condition 欄位）分類為 5 種疾病類別 |
| 評估指標 | **Macro F1-Score** |
| 目前最佳 LB | **0.63970** |
| 更新日期 | 2026-05-20 |

### 類別標籤

| 提交值 | 字串 | 特性 |
|--------|------|------|
| 1 | `neoplasms` | 關鍵詞極強（tumor/cancer/carcinoma 100% 可靠） |
| 2 | `digestive system diseases` | 關鍵詞中等（colitis/gallstone 可靠） |
| 3 | `nervous system diseases` | 關鍵詞中等（epileptic/seizure 可靠） |
| 4 | `cardiovascular diseases` | 關鍵詞強（coronary/myocardial 可靠） |
| 5 | `general pathological conditions` | **catch-all，無排他性關鍵詞，由排除法定義** |

---

## 2. 資料集特性（必讀）

| 檔案 | 筆數 | 說明 |
|------|------|------|
| `kaggle_trainset.csv` | 12,994 | 含 `label`（字串）、`condition` 欄位 |
| `kaggle_testset.csv` | 1,444 | 僅含 `condition` 欄位 |
| `kaggle_testset_submission.csv` | 1,444 | 提交範本，填整數 1-5 |

### 最重要發現：原始資料 vs 清洗資料

```
原始資料（12,994 筆，不去重）= 測試集分布  →  val/LB gap ≈ 0.018  ✓
清洗資料（10,395 筆，多數投票去重）         →  val/LB gap ≈ 0.13   ✗
```

**永遠使用原始 12,994 筆訓練。** 清洗後資料會造成 distribution mismatch，LB 大幅低於 val。

### 衝突標籤分析

- **2,400 個衝突文本**（38.5% 訓練樣本）
- 91.9% 是 50/50 tie（無法用多數投票解決）
- **72% 的衝突是「class5 vs 其他某類」**
- 對 neoplasms：高信心關鍵詞（cancer/tumor/carcinoma）在衝突中 100% 準確
- 對 cardiovascular/nervous/digestive：關鍵詞不足以可靠地解決衝突

---

## 3. 提交歷史（完整）

| 提交 | LB | Val/OOF | 備註 |
|------|-----|---------|------|
| **BiomedBERT-large 原始資料** | **0.63970** | 0.6581 | **目前最佳** |
| NLI+FGM+BiomedBERT ensemble v3 | 0.62229 | 0.7583 | 清洗資料，gap=0.136 |
| BiomedBERT+BioLinkBERT ensemble v2 | 0.62013 | 0.7586 | 清洗資料，gap=0.138 |
| Ensemble equal（3模型） | 0.61473 | — | 混入清洗資料模型 |
| Softlabel mul=0.5 | 0.60859 | 0.7683 | class5 bias |
| Softlabel mul=1.0 | 0.58279 | 0.7683 | class5 bias |
| BiomedBERT K-fold（移除衝突） | 0.574 | 0.818 | 只用 7,995 筆，OOF 虛高 |
| BiomedBERT+FGM v2 | 0.56660 | 0.66 | 雙重 class5 boost bug |
| Softlabel 原始（1.9x） | 0.53463 | — | 49% class5 預測 |

---

## 4. 失敗實驗與教訓

### 教訓 1：val F1 高 ≠ LB 高

Val 集來自訓練資料（含衝突），test 集有唯一 ground truth。
模型若對衝突樣本的「噪聲模式」過擬合，val F1 會虛高，但 LB 崩潰。

**判斷標準**：若 val/LB gap > 0.05，代表 distribution mismatch 或模型對噪聲過擬合。

### 教訓 2：Soft label 失敗

- 設計：衝突樣本用 KL divergence 訓練 [0.5, 0.5] 的軟目標
- 結果：模型學到「遇到模糊條件 = 預測 class5」，val F1=0.7683 虛高，LB=0.5828
- 根本原因：val 集含相同衝突條件，val 評估因此虛高；test 集無此特性

### 教訓 3：Class5 boost 雙重疊加

- FGM v2 腳本：CLASS5_BOOST=2.5（訓練）+ CLASS5_INF_MUL=1.9（推論）= 雙重疊加
- val 用原始機率評估（無 1.9x），submission 有 1.9x → 無法從 val 看出問題
- 結果：LB=0.5666，class5 佔 49% 預測
- **規則：val 評估和 submission 推論必須使用完全相同的後處理**

### 教訓 4：Ensemble 需要同分布模型

- 混合「原始資料訓練模型」和「清洗資料訓練模型」→ LB 被拉低
- 只有同樣用原始資料訓練的模型才能有效 ensemble

### 教訓 5：偽標籤違反競賽規則

- 使用測試集預測結果回頭訓練 = 違規，成績歸零
- **永遠不能用 test 預測結果做任何訓練資料**

---

## 5. 現有腳本說明

### Colab 訓練腳本（主力）

| 腳本 | 模型 | 資料 | 狀態 |
|------|------|------|------|
| `colab_biolinkbert_raw.py` | BioLinkBERT-large + FGM | 原始 12,994 | **進行中** |
| `colab_biomedbert_fgm_v2.py` | BiomedBERT-large + FGM | 原始 12,994 | 完成（LB失敗） |
| `colab_biomedbert_softlabel.py` | BiomedBERT-large + KL loss | 原始 12,994 | 完成（LB失敗） |
| `colab_biomedbert_kfold.py` | BiomedBERT-large K-fold | 清洗 10,395 | 備用 |
| `colab_deberta_kfold.py` | DeBERTa-v3-large K-fold | 清洗 10,395 | 備用 |

### 本地 npy 檔案（可直接 ensemble）

| 檔案 | 模型 | 訓練資料 | class5% | 說明 |
|------|------|---------|---------|------|
| `test_probs_biomedbert.npy` | BiomedBERT-large | 原始 | 20.0% | **最佳模型** |
| `test_probs_biolinkbert.npy` | BioLinkBERT-large | 清洗 | 17.5% | 清洗資料，不建議用 |
| `test_probs_biolinkbert_fgm.npy` | BioLinkBERT+FGM | 清洗 | 18.0% | 清洗資料，不建議用 |
| `test_probs_nli.npy` | DeBERTa NLI | — | 15.5% | 零樣本，不建議用 |
| `test_probs_biomedbert_softlabel.npy` | BiomedBERT+軟標籤 | 原始 | 32.7% | class5 bias，不建議用 |

---

## 6. 超參數標準（原始資料版）

```python
MODEL_NAME     = "microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract"
# 或 "michiyasunaga/BioLinkBERT-large"
MAX_LEN        = 512          # 重要！384 會損失資訊
_is_a100       = ...          # A100 偵測
BATCH_SIZE     = 16 if _is_a100 else 8
GRAD_ACCUM     = 4  if _is_a100 else 8   # effective batch = 64
EPOCHS         = 10
LR             = 8e-6
WARMUP_RATIO   = 0.20
LABEL_SMOOTH   = 0.1
VAL_RATIO      = 0.2          # 不用 K-fold，val-split 更省時間
PATIENCE       = 3
FGM_EPSILON    = 1.0          # FGM 對抗訓練
CLASS5_BOOST   = 1.0          # 不額外 boost！
CLASS5_INF_MUL = 1.0          # 不額外 boost！
```

### BF16 推論正確寫法

```python
USE_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=USE_BF16):
    logits = model(...)
probs = torch.softmax(logits.float(), dim=-1)  # 必須加 .float()
```

---

## 7. 資料分析發現

### 各類別高信心關鍵詞

| 類別 | 高信心詞（P>0.90）| 在衝突中可靠度 |
|------|-----------------|--------------|
| neoplasms | carcinoma, metastasis, cancer, cancers, adenocarcinoma | **100%** |
| cardiovascular | antihypertensive, diastolic, angiotensin | 高（但易被腫瘤詞搶先） |
| nervous | epileptic, antiepileptic, spasticity | 高（詞彙少） |
| digestive | colitis, gallstone, gastroesophageal | 高（詞彙少） |
| general | 無可靠排他性關鍵詞 | — |

### Cascade 推論策略

```python
# 如果 class1-4 中有足夠信心，預測 class1-4；否則 fallback 到 class5
non5_max = probs[:, 0:4].max(axis=1)
pred = np.where(non5_max >= threshold,
                probs[:, 0:4].argmax(axis=1) + 1,
                5)
```

threshold=0.35 → class5=17.4%（接近訓練集分布），已生成 `submission_cascade_thr0.35.csv`。

---

## 8. 下一步建議（按優先順序）

### 立即可做

1. **提交 `submission_cascade_thr0.35.csv`**：cascade 推論驗證，預計 LB ≈ 0.64+
2. **BioLinkBERT 訓練完成後**：下載 npy，做 BiomedBERT + BioLinkBERT 雙模型 ensemble

### 短期（需重訓）

3. **BioLinkBERT-large 原始資料**（`colab_biolinkbert_raw.py`）：目前正在 Colab 訓練
4. **R-Drop 訓練**：每 step 兩次前向傳播 + KL 正則化，比 FGM 更適合雜訊標籤場景
5. **關鍵詞前綴注入**：為輸入文字加上 `[NEOPLASM]` 等提示標籤

### 中期（較複雜）

6. **DeBERTa-v3-large 原始資料訓練**：更強的基礎模型
7. **Cascade 訓練**：training time 就區分「class1-4 預測」和「class5 fallback」
8. **Neoplasm 衝突去噪**：305 條高信心 neoplasm 衝突可安全改標籤

---

## 9. 標準 Colab 流程

```python
# Cell 1
from google.colab import drive
drive.mount('/content/drive')

# Cell 2
import os; os.chdir('/content')
!git clone https://github.com/eric20041027/Kaggle_medical_competition.git
os.chdir('/content/Kaggle_medical_competition')
!pip install -q transformers torch scikit-learn pandas numpy tqdm

# Cell 3
!python colab_XXX.py

# Cell 4
from google.colab import files
files.download('/content/drive/MyDrive/kaggle_XXX/test_probs_XXX.npy')
files.download('/content/drive/MyDrive/kaggle_XXX/submission_XXX.csv')
```

**注意**：A100 上 BATCH_SIZE=16, GRAD_ACCUM=4；T4 上 BATCH_SIZE=8, GRAD_ACCUM=8（腳本自動偵測）。

---

## 10. 競賽規則（絕對不能違反）

1. 禁止從網路取得測試集的正確答案
2. 禁止使用測試集預測結果訓練模型（pseudo-labeling = 違規）
3. 違者成績歸零
