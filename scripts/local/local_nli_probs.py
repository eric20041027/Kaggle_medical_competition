"""
零樣本 DeBERTa NLI 機率生成
============================
不需要訓練，直接對訓練集和測試集跑 zero-shot 推論。
輸出 oof_probs_nli.npy + test_probs_nli.npy 供 ensemble 使用。

執行環境：本地（CPU/GPU 皆可）或 Colab
執行時間：約 20-40 分鐘（CPU），5-10 分鐘（GPU）

使用方式：
  python local_nli_probs.py
"""

import os
import numpy as np
import pandas as pd
import torch
from transformers import pipeline
from tqdm import tqdm

TRAIN_PATH  = "kaggle_trainset.csv"
TEST_PATH   = "kaggle_testset.csv"
OUTPUT_DIR  = "."
BATCH_SIZE  = 16   # CPU 用 8，GPU 可用 16（T4 zero-shot 5-label 時 batch*5 會放大記憶體用量）
MODEL_NAME  = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"

LABEL_STR_LIST = [
    "neoplasms",
    "digestive system diseases",
    "nervous system diseases",
    "cardiovascular diseases",
    "general pathological conditions",
]
STR_TO_IDX    = {s: i for i, s in enumerate(LABEL_STR_LIST)}
IDX_TO_SUBMIT = {i: i + 1 for i in range(5)}

# v4_contrast：最佳標籤描述（Phase 2 驗證）
HYPOTHESES = [
    "This abstract specifically studies cancer, tumor, or malignant neoplasm.",
    "This abstract specifically studies a disease of the digestive tract, liver, or gastrointestinal system.",
    "This abstract specifically studies a neurological condition, brain disorder, or spinal disease.",
    "This abstract specifically studies a heart condition, vascular disease, or cardiovascular disorder.",
    "This abstract studies a general medical condition such as inflammation, infection, metabolic disorder, or immune dysfunction, which is not primarily a cancer, digestive, neurological, or cardiovascular disease.",
]

device = 0 if torch.cuda.is_available() else -1
print(f"Device: {'GPU' if device == 0 else 'CPU'}")
print(f"載入模型: {MODEL_NAME}")

classifier = pipeline(
    "zero-shot-classification",
    model=MODEL_NAME,
    device=device,
    batch_size=BATCH_SIZE,
)

def get_nli_probs(texts, desc=""):
    """對一批文本跑 NLI，回傳 (N, 5) softmax 機率矩陣"""
    all_probs = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc=desc):
        batch = texts[i: i + BATCH_SIZE]
        results = classifier(batch, HYPOTHESES, multi_label=False)
        if isinstance(results, dict):
            results = [results]
        for r in results:
            # results 的 labels 順序不固定，需對齊
            label_to_score = dict(zip(r["labels"], r["scores"]))
            probs = np.array([label_to_score[h] for h in HYPOTHESES], dtype=np.float32)
            probs /= probs.sum()   # 確保加總為 1
            all_probs.append(probs)
    return np.vstack(all_probs)

# ── 載入資料 ─────────────────────────────────────────────────────
print("\n載入資料...")
train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)

# 多數投票清洗（與訓練腳本一致）
label_counts = train_df.groupby(["condition", "label"]).size().reset_index(name="cnt")
majority_idx = label_counts.groupby("condition")["cnt"].idxmax()
df_clean     = label_counts.loc[majority_idx, ["condition", "label"]].reset_index(drop=True)

train_texts = df_clean["condition"].tolist()
test_texts  = test_df["condition"].tolist()
print(f"訓練集: {len(train_texts)} 筆  測試集: {len(test_texts)} 筆")

# ── 推論 ──────────────────────────────────────────────────────────
print("\n訓練集推論（作為 OOF 代理）...")
oof_probs = get_nli_probs(train_texts, desc="Train NLI")

print("\n測試集推論...")
test_probs = get_nli_probs(test_texts, desc="Test NLI")

# ── 評估 OOF ─────────────────────────────────────────────────────
from sklearn.metrics import f1_score, classification_report

all_labels  = [STR_TO_IDX[lbl] for lbl in df_clean["label"]]
oof_true    = [IDX_TO_SUBMIT[l] for l in all_labels]
oof_pred    = [IDX_TO_SUBMIT[i] for i in oof_probs.argmax(axis=1)]
oof_f1      = f1_score(oof_true, oof_pred, average="macro")

label_names = [f"{i+1}:{LABEL_STR_LIST[i][:14]}" for i in range(5)]
print(f"\nNLI OOF Macro F1: {oof_f1:.4f}")
print(classification_report(oof_true, oof_pred, target_names=label_names))

# ── 儲存 ──────────────────────────────────────────────────────────
oof_path  = os.path.join(OUTPUT_DIR, "oof_probs_nli.npy")
test_path = os.path.join(OUTPUT_DIR, "test_probs_nli.npy")
np.save(oof_path,  oof_probs)
np.save(test_path, test_probs)
print(f"\n儲存完成：")
print(f"  {oof_path}  shape={oof_probs.shape}")
print(f"  {test_path} shape={test_probs.shape}")
print("\n[後續] 將兩個 npy 加入 kaggle_ensemble_v3.py 做三模型 ensemble")
