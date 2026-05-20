"""
Hierarchical + BiomedBERT Ensemble
=====================================
使用方式：
  1. 從 Colab 下載 test_probs_hierarchical.npy 到 probs/ 資料夾
  2. python scripts/local/ensemble_hierarchical.py

會自動根據 hierarchical 的 class5 比率決定 ensemble 權重：
  < 23%：均等 0.5/0.5
  23-28%：BiomedBERT 偏重 0.6/0.4
  > 28%：強制 class5 保正後再 ensemble
"""

import numpy as np
import pandas as pd
import os

PROBS_DIR   = "probs"
SUBMIT_PATH = "kaggle_testset_submission.csv"
OUT_DIR     = "submissions"

os.makedirs(OUT_DIR, exist_ok=True)

biomedbert_path   = os.path.join(PROBS_DIR, "test_probs_biomedbert.npy")
hierarchical_path = os.path.join(PROBS_DIR, "test_probs_hierarchical.npy")

assert os.path.exists(biomedbert_path), f"找不到 {biomedbert_path}"
assert os.path.exists(hierarchical_path), f"找不到 {hierarchical_path}，請先從 Colab 下載"

probs_bio  = np.load(biomedbert_path)   # shape (1444, 5)
probs_hier = np.load(hierarchical_path) # shape (1444, 5)

assert probs_bio.shape == probs_hier.shape == (1444, 5)

hier_c5_rate = (probs_hier.argmax(axis=1) == 4).mean()
print(f"BiomedBERT class5 比率: {(probs_bio.argmax(axis=1)==4).mean()*100:.1f}%")
print(f"Hierarchical class5 比率: {hier_c5_rate*100:.1f}%")

# Hierarchical 保正（若 class5 過高）
probs_hier_adj = probs_hier.copy()
if hier_c5_rate > 0.28:
    print(f"⚠ class5 比率 {hier_c5_rate*100:.1f}% > 28%，套用 P5×0.82 保正")
    probs_hier_adj[:, 4] *= 0.82
    probs_hier_adj = probs_hier_adj / probs_hier_adj.sum(axis=1, keepdims=True)
elif hier_c5_rate > 0.23:
    print(f"⚠ class5 比率 {hier_c5_rate*100:.1f}% > 23%，套用 P5×0.90 保正")
    probs_hier_adj[:, 4] *= 0.90
    probs_hier_adj = probs_hier_adj / probs_hier_adj.sum(axis=1, keepdims=True)

adj_c5_rate = (probs_hier_adj.argmax(axis=1) == 4).mean()
print(f"Hierarchical 保正後 class5 比率: {adj_c5_rate*100:.1f}%")
print()

# Determine ensemble weights
if hier_c5_rate < 0.23:
    w_bio, w_hier = 0.5, 0.5
    weight_desc = "均等 0.5/0.5"
elif hier_c5_rate < 0.28:
    w_bio, w_hier = 0.6, 0.4
    weight_desc = "BiomedBERT 偏重 0.6/0.4"
else:
    w_bio, w_hier = 0.65, 0.35
    weight_desc = "BiomedBERT 強偏重 0.65/0.35（hierarchical class5 過高）"

print(f"Ensemble 策略: {weight_desc}")

# Generate all three variants for comparison
variants = {
    f"hier_ensemble_equal": (0.5, 0.5),
    f"hier_ensemble_bio60": (0.6, 0.4),
    f"hier_ensemble_auto":  (w_bio, w_hier),
}

submission_base = pd.read_csv(SUBMIT_PATH)

for name, (wb, wh) in variants.items():
    combined = wb * probs_bio + wh * probs_hier_adj
    preds = combined.argmax(axis=1) + 1  # 1-indexed

    sub = submission_base.copy()
    sub["label"] = preds
    out_path = os.path.join(OUT_DIR, f"submission_{name}.csv")
    sub.to_csv(out_path, index=False)

    from collections import Counter
    dist = Counter(preds)
    print(f"\n[{name}] w_bio={wb} w_hier={wh}")
    print(f"  分布: " + "  ".join([f"class{k}={dist[k]}" for k in sorted(dist)]))
    print(f"  → {out_path}")

print(f"\n建議提交: submission_hier_ensemble_auto.csv")
print(f"  (自動選擇 {weight_desc})")
