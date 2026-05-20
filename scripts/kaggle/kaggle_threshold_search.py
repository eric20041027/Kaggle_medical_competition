"""
方案 B：Label-5 Confidence Threshold Grid Search
=================================================
原理：若模型對 label 1-4 的最高信心 < τ，改預測 label 5
     在 OOF 上 grid search 找最佳 τ，確認是否真的提升 Macro F1

使用方式：
  接在 kaggle_large_kfold_v2.py 之後執行，或直接在同一 notebook 的最後加上。
  所需檔案：
    - oof_probs_biomedbert.npy   (shape: N_train × 5, 由 v2 腳本輸出)
    - test_probs_biomedbert.npy  (shape: 1444 × 5)
    - kaggle_trainset.csv        (重建 OOF 真實標籤)
    - kaggle_testset_submission.csv

  也可替換成其他模型的 oof_probs.npy / test_probs.npy。
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report

# ── 路徑（Kaggle 環境）────────────────────────────────────────
OUTPUT_DIR  = "/kaggle/working"
DATA_DIR    = "/kaggle/input/competitions/1142-medical-condition-classification"
TRAIN_PATH  = os.path.join(DATA_DIR, "kaggle_trainset.csv")
SUBMIT_PATH = os.path.join(DATA_DIR, "kaggle_testset_submission.csv")

# 若在 OUTPUT_DIR 找不到，嘗試當前目錄（本地測試用）
def load_npy(filename):
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        path = filename
    arr = np.load(path)
    print(f"  載入 {filename}: shape={arr.shape}")
    return arr

print("=== 載入 OOF 軟機率 ===")
oof_probs  = load_npy("oof_probs_biomedbert.npy")   # (N_train, 5)
test_probs = load_npy("test_probs_biomedbert.npy")  # (1444, 5)

# ── 重建 OOF 真實標籤（與 v2 相同的多數投票清洗）─────────────
LABEL_STR_LIST = [
    "neoplasms",
    "digestive system diseases",
    "nervous system diseases",
    "cardiovascular diseases",
    "general pathological conditions",
]
STR_TO_IDX    = {s: i for i, s in enumerate(LABEL_STR_LIST)}
IDX_TO_SUBMIT = {i: i + 1 for i in range(5)}

train_df = pd.read_csv(TRAIN_PATH)
label_counts = (
    train_df.groupby(["condition", "label"])
    .size()
    .reset_index(name="cnt")
)
majority_idx = label_counts.groupby("condition")["cnt"].idxmax()
df_clean     = label_counts.loc[majority_idx, ["condition", "label"]].reset_index(drop=True)

all_labels   = [STR_TO_IDX[lbl] for lbl in df_clean["label"]]
oof_true_idx = np.array(all_labels)                          # 0-based
oof_true_sub = np.array([IDX_TO_SUBMIT[l] for l in all_labels])  # 1-based

assert len(oof_probs) == len(all_labels), (
    f"OOF 長度不符 ({len(oof_probs)} vs {len(all_labels)})。"
    "確認 oof_probs_biomedbert.npy 是由 kaggle_large_kfold_v2.py 產生的。"
)

# ── 基準線（無 threshold）────────────────────────────────────
baseline_pred = oof_probs.argmax(axis=1)
baseline_sub  = np.array([IDX_TO_SUBMIT[i] for i in baseline_pred])
baseline_f1   = f1_score(oof_true_sub, baseline_sub, average="macro")

print(f"\n=== 基準線（無 threshold）===")
print(f"OOF Macro F1: {baseline_f1:.4f}")
label_names = [f"{i+1}:{LABEL_STR_LIST[i][:14]}" for i in range(5)]
print(classification_report(oof_true_sub, baseline_sub, target_names=label_names))

# ── Threshold 函式 ────────────────────────────────────────────
def apply_threshold(probs, tau):
    """
    如果 max(P(class 0-3)) < tau → 改預測 class 4（label 5）
    否則維持原 argmax
    """
    preds = probs.argmax(axis=1).copy()
    max_p14 = probs[:, :4].max(axis=1)       # label 1-4 的最大信心
    preds[max_p14 < tau] = 4                  # 不確定 → label 5
    return preds

# ── Grid Search ──────────────────────────────────────────────
print("=== Grid Search（τ 從 0.30 到 0.95）===")
print(f"{'τ':>6}  {'OOF F1':>8}  {'vs base':>8}  {'#changed':>9}  "
      f"{'F1_1':>6}  {'F1_2':>6}  {'F1_3':>6}  {'F1_4':>6}  {'F1_5':>6}")
print("-" * 80)

results = []
for tau in np.arange(0.30, 0.96, 0.05):
    preds_idx = apply_threshold(oof_probs, tau)
    preds_sub = np.array([IDX_TO_SUBMIT[i] for i in preds_idx])
    macro_f1  = f1_score(oof_true_sub, preds_sub, average="macro")
    per_class = f1_score(oof_true_sub, preds_sub, average=None, labels=[1,2,3,4,5])
    n_changed = int((preds_idx != oof_probs.argmax(axis=1)).sum())
    delta     = macro_f1 - baseline_f1
    marker    = " ◀ 最佳" if macro_f1 == max([r["f1"] for r in results] + [macro_f1]) else ""
    print(f"τ={tau:.2f}  {macro_f1:.4f}  {delta:+.4f}  {n_changed:>9}  "
          f"{per_class[0]:.4f}  {per_class[1]:.4f}  {per_class[2]:.4f}  "
          f"{per_class[3]:.4f}  {per_class[4]:.4f}{marker}")
    results.append({"tau": round(tau, 2), "f1": macro_f1, "delta": delta,
                    "n_changed": n_changed,
                    "f1_per_class": per_class})

# ── 找最佳 τ ─────────────────────────────────────────────────
best = max(results, key=lambda r: r["f1"])
print(f"\n最佳 τ = {best['tau']:.2f}  →  OOF F1 = {best['f1']:.4f}  "
      f"（{'+'if best['delta']>=0 else ''}{best['delta']:.4f} vs 基準）")
print(f"改變了 {best['n_changed']} 筆預測（佔 OOF 的 {best['n_changed']/len(oof_probs)*100:.1f}%）")

# ── 決策 ─────────────────────────────────────────────────────
IMPROVEMENT_THRESHOLD = 0.001   # 至少提升 0.001 才算有意義

if best["delta"] > IMPROVEMENT_THRESHOLD:
    print(f"\n✓ Threshold 策略有效！將最佳 τ={best['tau']:.2f} 套用到 test set")

    # 套用 threshold 到 test 預測
    test_preds_thresh = apply_threshold(test_probs, best["tau"])
    test_pred_submit  = [IDX_TO_SUBMIT[i] for i in test_preds_thresh]

    # 對比基準 test 預測
    test_preds_base   = test_probs.argmax(axis=1)
    n_test_changed    = int((test_preds_thresh != test_preds_base).sum())
    print(f"Test set 共改變 {n_test_changed} 筆預測（佔 {n_test_changed/len(test_probs)*100:.1f}%）")

    # 生成提交檔
    submission = pd.read_csv(SUBMIT_PATH)
    submission["label"] = test_pred_submit
    out_path = os.path.join(OUTPUT_DIR, f"submission_thresh{best['tau']:.2f}.csv")
    submission.to_csv(out_path, index=False)

    print(f"\n提交檔 → {out_path}")
    print(f"預測分布（套用 threshold 後）:\n{pd.Series(test_pred_submit).value_counts().sort_index()}")
    base_dist = pd.Series([IDX_TO_SUBMIT[i] for i in test_preds_base]).value_counts().sort_index()
    print(f"預測分布（基準）:\n{base_dist}")

else:
    print(f"\n✗ Threshold 策略無效（最大提升 {best['delta']:+.4f}，未達 {IMPROVEMENT_THRESHOLD} 門檻）")
    print("  → 不套用 threshold，維持原始 argmax 預測即可")
    print()
    print("  診斷：低信心的 1-4 案例大多不是真正的 label 5，")
    print("        強制改成 5 反而引入更多錯誤。")

# ── 深度分析：哪些樣本被改變了 ───────────────────────────────
best_tau = best["tau"]
preds_with_thresh = apply_threshold(oof_probs, best_tau)
changed_mask = preds_with_thresh != oof_probs.argmax(axis=1)

if changed_mask.sum() > 0:
    print(f"\n=== 被 τ={best_tau:.2f} 改變的樣本分析 ===")
    changed_true  = oof_true_sub[changed_mask]
    changed_orig  = np.array([IDX_TO_SUBMIT[i] for i in oof_probs.argmax(axis=1)])[changed_mask]
    changed_new   = np.array([IDX_TO_SUBMIT[i] for i in preds_with_thresh])[changed_mask]

    correct_orig  = (changed_true == changed_orig).sum()
    correct_new   = (changed_true == changed_new).sum()   # 改成 5 之後對了幾個
    truly_5       = (changed_true == 5).sum()

    print(f"被改動的樣本數: {changed_mask.sum()}")
    print(f"其中真實標籤是 5: {truly_5} ({truly_5/changed_mask.sum()*100:.1f}%)")
    print(f"改動前預測正確: {correct_orig} ({correct_orig/changed_mask.sum()*100:.1f}%)")
    print(f"改動後預測正確: {correct_new} ({correct_new/changed_mask.sum()*100:.1f}%)")
    print(f"  → 淨效益: {correct_new - correct_orig:+d} 筆（正數=改善，負數=惡化）")

    print(f"\n被改動樣本的原始預測分布（改前）:")
    print(pd.Series(changed_orig).value_counts().sort_index().to_string())
    print(f"\n被改動樣本的真實標籤分布:")
    print(pd.Series(changed_true).value_counts().sort_index().to_string())
