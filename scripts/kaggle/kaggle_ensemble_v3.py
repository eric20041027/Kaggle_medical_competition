"""
三模型 Ensemble：BioLinkBERT-FGM + BiomedBERT + NLI
====================================================
所需檔案（放在同目錄或 ~/Downloads）：
  oof_probs_biolinkbert_fgm.npy   （colab_biolinkbert_fgm_kfold.py 輸出）
  test_probs_biolinkbert_fgm.npy
  oof_probs_biomedbert.npy        （已有）
  test_probs_biomedbert.npy       （已有）
  oof_probs_nli.npy               （local_nli_probs.py 輸出）
  test_probs_nli.npy

執行：
  python kaggle_ensemble_v3.py
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report
from itertools import product

LABEL_STR_LIST = [
    "neoplasms",
    "digestive system diseases",
    "nervous system diseases",
    "cardiovascular diseases",
    "general pathological conditions",
]
IDX_TO_SUBMIT = {i: i + 1 for i in range(5)}
STR_TO_IDX    = {s: i for i, s in enumerate(LABEL_STR_LIST)}

# ── 路徑 ──────────────────────────────────────────────────────────
PROBS_DIR   = "."
SUBMIT_PATH = "kaggle_testset_submission.csv"
OUTPUT_DIR  = "."

print("載入軟機率...")
oof_fgm  = np.load(os.path.join(PROBS_DIR, "oof_probs_biolinkbert_fgm.npy"))
oof_bio  = np.load(os.path.join(PROBS_DIR, "oof_probs_biomedbert.npy"))
oof_nli  = np.load(os.path.join(PROBS_DIR, "oof_probs_nli.npy"))
test_fgm = np.load(os.path.join(PROBS_DIR, "test_probs_biolinkbert_fgm.npy"))
test_bio = np.load(os.path.join(PROBS_DIR, "test_probs_biomedbert.npy"))
test_nli = np.load(os.path.join(PROBS_DIR, "test_probs_nli.npy"))

print(f"BioLinkBERT-FGM OOF: {oof_fgm.shape}")
print(f"BiomedBERT      OOF: {oof_bio.shape}")
print(f"NLI             OOF: {oof_nli.shape}")

# ── 還原真實標籤 ──────────────────────────────────────────────────
train_df = pd.read_csv(
    "kaggle_trainset.csv"
    if os.path.exists("kaggle_trainset.csv")
    else "/kaggle/input/competitions/1142-medical-condition-classification/kaggle_trainset.csv"
)
label_counts = train_df.groupby(["condition", "label"]).size().reset_index(name="cnt")
majority_idx = label_counts.groupby("condition")["cnt"].idxmax()
df_clean     = label_counts.loc[majority_idx, ["condition", "label"]].reset_index(drop=True)
all_labels   = [STR_TO_IDX[lbl] for lbl in df_clean["label"]]
oof_true     = [IDX_TO_SUBMIT[l] for l in all_labels]

# ── 單模型基準 ────────────────────────────────────────────────────
fgm_f1 = f1_score(oof_true, [IDX_TO_SUBMIT[i] for i in oof_fgm.argmax(1)], average="macro")
bio_f1 = f1_score(oof_true, [IDX_TO_SUBMIT[i] for i in oof_bio.argmax(1)], average="macro")
nli_f1 = f1_score(oof_true, [IDX_TO_SUBMIT[i] for i in oof_nli.argmax(1)], average="macro")
print(f"\n單模型 OOF F1:")
print(f"  BioLinkBERT-FGM: {fgm_f1:.4f}")
print(f"  BiomedBERT:      {bio_f1:.4f}")
print(f"  NLI (zero-shot): {nli_f1:.4f}")

# ── Grid Search（三權重，步進 0.1）────────────────────────────────
print("\nGrid search 三模型最佳權重 ...")
best_f1, best_w = 0.0, (0.5, 0.4, 0.1)

weights = [round(w * 0.1, 1) for w in range(1, 10)]
for w_fgm in weights:
    for w_bio in weights:
        w_nli = round(1.0 - w_fgm - w_bio, 1)
        if w_nli <= 0:
            continue
        combined = w_fgm * oof_fgm + w_bio * oof_bio + w_nli * oof_nli
        pred = [IDX_TO_SUBMIT[i] for i in combined.argmax(axis=1)]
        f1   = f1_score(oof_true, pred, average="macro")
        if f1 > best_f1:
            best_f1, best_w = f1, (w_fgm, w_bio, w_nli)

w_fgm_b, w_bio_b, w_nli_b = best_w
print(f"\n最佳權重: FGM={w_fgm_b}  Bio={w_bio_b}  NLI={w_nli_b}")
print(f"三模型 Ensemble OOF F1: {best_f1:.4f}")
print(f"(FGM:{fgm_f1:.4f}  Bio:{bio_f1:.4f}  NLI:{nli_f1:.4f})")

# OOF classification report
oof_combined   = w_fgm_b * oof_fgm + w_bio_b * oof_bio + w_nli_b * oof_nli
oof_pred_final = [IDX_TO_SUBMIT[i] for i in oof_combined.argmax(axis=1)]
label_names    = [f"{i+1}:{LABEL_STR_LIST[i][:14]}" for i in range(5)]
print(f"\nOOF Classification Report:")
print(classification_report(oof_true, oof_pred_final, target_names=label_names))

# ── 生成提交 ──────────────────────────────────────────────────────
test_combined    = w_fgm_b * test_fgm + w_bio_b * test_bio + w_nli_b * test_nli
test_pred_submit = [IDX_TO_SUBMIT[i] for i in test_combined.argmax(axis=1)]

submit_template = pd.read_csv(
    SUBMIT_PATH if os.path.exists(SUBMIT_PATH)
    else "/kaggle/input/competitions/1142-medical-condition-classification/kaggle_testset_submission.csv"
)
submit_template["label"] = test_pred_submit
out_path = os.path.join(OUTPUT_DIR, "submission_ensemble_v3.csv")
submit_template.to_csv(out_path, index=False)

print(f"\n提交檔 → {out_path}")
print(f"預測分布:\n{pd.Series(test_pred_submit).value_counts().sort_index()}")
print(f"\n完成！三模型 Ensemble OOF F1 = {best_f1:.4f}")
