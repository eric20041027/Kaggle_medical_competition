"""
Ensemble: BiomedBERT-large v2 + DeBERTa-v3-large
=================================================
使用方式：
  1. 執行 kaggle_large_kfold_v2.py → 下載 test_probs_biomedbert.npy + oof_probs_biomedbert.npy
  2. 執行 kaggle_deberta_kfold.py  → 下載 test_probs_deberta.npy  + oof_probs_deberta.npy
  3. 將兩對 .npy 上傳為 Kaggle Dataset（或掛載至 /kaggle/input/model-probs/）
  4. 執行本腳本：用 OOF 做 grid search，最佳權重套用到 test → 提交

Kaggle 路徑設定（依你上傳的 dataset 名稱調整）：
  PROBS_DIR = "/kaggle/input/model-probs"
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report

# ── 路徑設定 ──────────────────────────────────────────────────
# 在 Kaggle 上執行時，將 PROBS_DIR 改為你上傳的 dataset 路徑
# 在本地執行時，直接放在同一目錄
PROBS_DIR   = "."
SUBMIT_PATH = "kaggle_testset_submission.csv"
OUTPUT_DIR  = "."

LABEL_STR_LIST = [
    "neoplasms",
    "digestive system diseases",
    "nervous system diseases",
    "cardiovascular diseases",
    "general pathological conditions",
]
IDX_TO_SUBMIT = {i: i + 1 for i in range(5)}

# ── 載入 OOF 和 Test 軟機率 ───────────────────────────────────
print("載入軟機率...")
oof_bio   = np.load(os.path.join(PROBS_DIR, "oof_probs_biomedbert.npy"))   # (N_train, 5)
oof_deb   = np.load(os.path.join(PROBS_DIR, "oof_probs_deberta.npy"))      # (N_train, 5)
test_bio  = np.load(os.path.join(PROBS_DIR, "test_probs_biomedbert.npy"))  # (1444, 5)
test_deb  = np.load(os.path.join(PROBS_DIR, "test_probs_deberta.npy"))     # (1444, 5)

print(f"BiomedBERT OOF shape: {oof_bio.shape}")
print(f"DeBERTa    OOF shape: {oof_deb.shape}")
print(f"BiomedBERT Test shape: {test_bio.shape}")
print(f"DeBERTa    Test shape: {test_deb.shape}")

# ── 還原 OOF 真實標籤 ─────────────────────────────────────────
# 從訓練集重建（與兩個訓練腳本使用相同的多數投票清洗）
train_df = pd.read_csv(
    os.path.join("kaggle_trainset.csv")
    if os.path.exists("kaggle_trainset.csv")
    else "/kaggle/input/competitions/1142-medical-condition-classification/kaggle_trainset.csv"
)
label_counts = (
    train_df.groupby(["condition", "label"])
    .size()
    .reset_index(name="cnt")
)
majority_idx = label_counts.groupby("condition")["cnt"].idxmax()
df_clean = label_counts.loc[majority_idx, ["condition", "label"]].reset_index(drop=True)

STR_TO_IDX = {s: i for i, s in enumerate(LABEL_STR_LIST)}
all_labels  = [STR_TO_IDX[lbl] for lbl in df_clean["label"]]
oof_true    = [IDX_TO_SUBMIT[l] for l in all_labels]

assert len(oof_bio) == len(all_labels), \
    f"OOF 長度不符：{len(oof_bio)} vs {len(all_labels)}（確認兩個訓練腳本用了相同清洗邏輯）"

# ── 單模型 OOF 評估 ───────────────────────────────────────────
bio_oof_f1 = f1_score(oof_true, [IDX_TO_SUBMIT[i] for i in oof_bio.argmax(1)], average="macro")
deb_oof_f1 = f1_score(oof_true, [IDX_TO_SUBMIT[i] for i in oof_deb.argmax(1)], average="macro")
print(f"\n單模型 OOF F1:")
print(f"  BiomedBERT-large: {bio_oof_f1:.4f}")
print(f"  DeBERTa-v3-large: {deb_oof_f1:.4f}")

# ── Grid Search 最佳 ensemble 權重 ───────────────────────────
print("\nGrid search 最佳 ensemble 權重 ...")
best_f1, best_w_bio = 0.0, 0.5

for w_bio in np.arange(0.05, 1.0, 0.05):
    w_deb    = 1.0 - w_bio
    combined = w_bio * oof_bio + w_deb * oof_deb
    pred     = [IDX_TO_SUBMIT[i] for i in combined.argmax(axis=1)]
    f1       = f1_score(oof_true, pred, average="macro")
    if f1 > best_f1:
        best_f1, best_w_bio = f1, w_bio

w_bio_best = best_w_bio
w_deb_best = 1.0 - w_bio_best
print(f"\n最佳權重: BiomedBERT={w_bio_best:.2f}  DeBERTa={w_deb_best:.2f}")
print(f"Ensemble OOF Macro F1: {best_f1:.4f}  "
      f"(BiomedBERT alone: {bio_oof_f1:.4f}, DeBERTa alone: {deb_oof_f1:.4f})")

# 最佳權重的 OOF classification report
oof_combined = w_bio_best * oof_bio + w_deb_best * oof_deb
oof_pred_final = [IDX_TO_SUBMIT[i] for i in oof_combined.argmax(axis=1)]
label_names = [f"{i+1}:{LABEL_STR_LIST[i][:14]}" for i in range(5)]
print(f"\nOOF Classification Report (w_bio={w_bio_best:.2f}):")
print(classification_report(oof_true, oof_pred_final, target_names=label_names))

# ── 生成最終提交 ──────────────────────────────────────────────
test_combined    = w_bio_best * test_bio + w_deb_best * test_deb
test_pred_submit = [IDX_TO_SUBMIT[i] for i in test_combined.argmax(axis=1)]

submit_template = pd.read_csv(
    SUBMIT_PATH
    if os.path.exists(SUBMIT_PATH)
    else "/kaggle/input/competitions/1142-medical-condition-classification/kaggle_testset_submission.csv"
)
submit_template["label"] = test_pred_submit
out_path = os.path.join(OUTPUT_DIR, "submission_ensemble_v2.csv")
submit_template.to_csv(out_path, index=False)

print(f"\n提交檔 → {out_path}")
print(f"預測分布:\n{pd.Series(test_pred_submit).value_counts().sort_index()}")
print(f"\n完成！Ensemble OOF Macro F1 = {best_f1:.4f}")

# ── 也輸出各模型單獨提交（供比較分析）──────────────────────────
for name, test_p in [("biomedbert", test_bio), ("deberta", test_deb)]:
    preds = [IDX_TO_SUBMIT[i] for i in test_p.argmax(axis=1)]
    sub = submit_template.copy()
    sub["label"] = preds
    sub.to_csv(os.path.join(OUTPUT_DIR, f"submission_{name}_solo.csv"), index=False)
print("\n各模型單獨提交檔也已輸出（_solo.csv），可個別上傳比對分數")
