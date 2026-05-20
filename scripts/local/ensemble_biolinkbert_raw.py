"""
BiomedBERT + BioLinkBERT（原始資料）Ensemble
=============================================
使用方式：
  1. 從 Colab 下載 test_probs_biolinkbert_raw.npy 到 probs/ 資料夾
  2. python scripts/local/ensemble_biolinkbert_raw.py
"""

import numpy as np
import pandas as pd
import os
from collections import Counter

PROBS_DIR   = "probs"
SUBMIT_PATH = "kaggle_testset_submission.csv"
OUT_DIR     = "submissions"
os.makedirs(OUT_DIR, exist_ok=True)

biomedbert_path    = os.path.join(PROBS_DIR, "test_probs_biomedbert.npy")
biolinkbert_path   = os.path.join(PROBS_DIR, "test_probs_biolinkbert_raw.npy")

assert os.path.exists(biomedbert_path),  f"找不到 {biomedbert_path}"
assert os.path.exists(biolinkbert_path), f"找不到 {biolinkbert_path}，請先從 Colab 下載"

probs_bio  = np.load(biomedbert_path)
probs_link = np.load(biolinkbert_path)

assert probs_bio.shape == probs_link.shape == (1444, 5)

bio_c5  = (probs_bio.argmax(axis=1)  == 4).mean()
link_c5 = (probs_link.argmax(axis=1) == 4).mean()
print(f"BiomedBERT   class5 比率: {bio_c5*100:.1f}%  (LB=0.63970)")
print(f"BioLinkBERT  class5 比率: {link_c5*100:.1f}%")
print()

submission_base = pd.read_csv(SUBMIT_PATH)

variants = {
    "ensemble_equal":   (0.5, 0.5),
    "ensemble_bio60":   (0.6, 0.4),
    "ensemble_link60":  (0.4, 0.6),
}

for name, (wb, wl) in variants.items():
    combined = wb * probs_bio + wl * probs_link
    preds    = combined.argmax(axis=1) + 1
    dist     = Counter(preds)

    sub = submission_base.copy()
    sub["label"] = preds
    out_path = os.path.join(OUT_DIR, f"submission_{name}.csv")
    sub.to_csv(out_path, index=False)

    print(f"[{name}]  bio={wb}  link={wl}")
    print(f"  分布: " + "  ".join([f"class{k}={dist[k]}" for k in sorted(dist)]))
    print(f"  class5 比率: {dist[5]/1444*100:.1f}%")
    print(f"  → {out_path}")
    print()

print("建議提交：submission_ensemble_equal.csv")
print("若 BioLinkBERT val F1 顯著高於 BiomedBERT，可改用 ensemble_link60")
