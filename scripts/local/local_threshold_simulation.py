"""
本地快速模擬：Label-5 Threshold 策略驗證
==========================================
不需要 GPU，不需要預先訓練的模型。
用 TF-IDF + Logistic Regression 做 3-Fold 交叉驗證，
取得 OOF softmax 機率後，直接跑 threshold grid search。

目的：在送上 Kaggle 之前，先在本地確認 threshold 策略是否有效。
執行時間：約 1-3 分鐘（本地 CPU）

執行方式：
  python3 local_threshold_simulation.py
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import label_binarize

# ── 設定 ─────────────────────────────────────────────────────
TRAIN_PATH = "kaggle_trainset.csv"
SEED       = 42
N_FOLDS    = 3

LABEL_STR_LIST = [
    "neoplasms",
    "digestive system diseases",
    "nervous system diseases",
    "cardiovascular diseases",
    "general pathological conditions",
]
STR_TO_IDX    = {s: i for i, s in enumerate(LABEL_STR_LIST)}
IDX_TO_SUBMIT = {i: i + 1 for i in range(5)}

# ── 載入並清洗資料（多數投票）────────────────────────────────
print("載入訓練集...")
train_df = pd.read_csv(TRAIN_PATH)
label_counts = (
    train_df.groupby(["condition", "label"])
    .size()
    .reset_index(name="cnt")
)
majority_idx = label_counts.groupby("condition")["cnt"].idxmax()
df_clean = label_counts.loc[majority_idx, ["condition", "label"]].reset_index(drop=True)
print(f"清洗後: {len(df_clean)} 筆（多數投票保留衝突文本）")

texts  = df_clean["condition"].tolist()
labels = np.array([STR_TO_IDX[lbl] for lbl in df_clean["label"]])
true_sub = np.array([IDX_TO_SUBMIT[l] for l in labels])

# ── 3-Fold OOF 交叉驗證（TF-IDF + LR）──────────────────────
print(f"\n跑 {N_FOLDS}-Fold 交叉驗證（TF-IDF + Logistic Regression）...")
skf       = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof_probs = np.zeros((len(texts), 5))

for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels)):
    X_train = [texts[i] for i in train_idx]
    X_val   = [texts[i] for i in val_idx]
    y_train = labels[train_idx]

    vec   = TfidfVectorizer(max_features=50000, ngram_range=(1, 2),
                            sublinear_tf=True, min_df=2)
    Xtr   = vec.fit_transform(X_train)
    Xval  = vec.transform(X_val)

    clf   = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED,
                               class_weight="balanced", multi_class="multinomial")
    clf.fit(Xtr, y_train)

    probs = clf.predict_proba(Xval)           # (n_val, 5)
    oof_probs[val_idx] = probs

    fold_f1 = f1_score(true_sub[val_idx],
                       [IDX_TO_SUBMIT[i] for i in probs.argmax(axis=1)],
                       average="macro")
    print(f"  Fold {fold+1}: val F1 = {fold_f1:.4f}")

# ── 基準線 ───────────────────────────────────────────────────
baseline_pred = oof_probs.argmax(axis=1)
baseline_sub  = np.array([IDX_TO_SUBMIT[i] for i in baseline_pred])
baseline_f1   = f1_score(true_sub, baseline_sub, average="macro")

label_names = [f"{i+1}:{LABEL_STR_LIST[i][:14]}" for i in range(5)]
print(f"\n{'='*60}")
print(f"  基準 OOF Macro F1（無 threshold）: {baseline_f1:.4f}")
print(f"{'='*60}")
print(classification_report(true_sub, baseline_sub, target_names=label_names))

# ── Threshold 函式 ────────────────────────────────────────────
def apply_threshold(probs, tau):
    preds   = probs.argmax(axis=1).copy()
    max_p14 = probs[:, :4].max(axis=1)
    preds[max_p14 < tau] = 4
    return preds

# ── Grid Search ──────────────────────────────────────────────
print(f"{'='*60}")
print("  Grid Search（τ 從 0.30 到 0.95）")
print(f"{'='*60}")
print(f"{'τ':>6}  {'OOF F1':>8}  {'vs base':>8}  {'#changed':>9}  "
      f"{'F1_1':>6}  {'F1_2':>6}  {'F1_3':>6}  {'F1_4':>6}  {'F1_5':>6}")
print("-" * 80)

results = []
for tau in np.arange(0.30, 0.96, 0.05):
    preds_idx = apply_threshold(oof_probs, tau)
    preds_sub = np.array([IDX_TO_SUBMIT[i] for i in preds_idx])
    macro_f1  = f1_score(true_sub, preds_sub, average="macro")
    per_class = f1_score(true_sub, preds_sub, average=None, labels=[1,2,3,4,5])
    n_changed = int((preds_idx != baseline_pred).sum())
    delta     = macro_f1 - baseline_f1
    results.append({"tau": round(tau, 2), "f1": macro_f1, "delta": delta,
                    "n_changed": n_changed, "per_class": per_class})

best_f1_so_far = -1
for r in results:
    is_best = r["f1"] > best_f1_so_far
    if is_best:
        best_f1_so_far = r["f1"]
    marker = " ◀ 目前最佳" if is_best else ""
    print(f"τ={r['tau']:.2f}  {r['f1']:.4f}  {r['delta']:+.4f}  "
          f"{r['n_changed']:>9}  "
          f"{r['per_class'][0]:.4f}  {r['per_class'][1]:.4f}  "
          f"{r['per_class'][2]:.4f}  {r['per_class'][3]:.4f}  "
          f"{r['per_class'][4]:.4f}{marker}")

best = max(results, key=lambda r: r["f1"])
print(f"\n{'='*60}")
print(f"  最佳 τ = {best['tau']:.2f}")
print(f"  OOF F1 = {best['f1']:.4f}（基準 {baseline_f1:.4f}，差 {best['delta']:+.4f}）")
print(f"  改變了 {best['n_changed']} / {len(texts)} 筆預測 ({best['n_changed']/len(texts)*100:.1f}%)")
print(f"{'='*60}")

# ── 深度分析：被改變的樣本 ────────────────────────────────────
best_tau  = best["tau"]
new_preds = apply_threshold(oof_probs, best_tau)
changed   = new_preds != baseline_pred

print(f"\n=== 被 τ={best_tau:.2f} 改動的 {changed.sum()} 筆樣本分析 ===")
changed_true = true_sub[changed]
changed_orig = baseline_sub[changed]
changed_new  = np.array([IDX_TO_SUBMIT[i] for i in new_preds])[changed]

truly_5      = (changed_true == 5).sum()
correct_orig = (changed_true == changed_orig).sum()
correct_new  = (changed_true == changed_new).sum()

print(f"真實標籤是 label 5: {truly_5} ({truly_5/changed.sum()*100:.1f}%)")
print(f"改動前正確:         {correct_orig} ({correct_orig/changed.sum()*100:.1f}%)")
print(f"改動後正確:         {correct_new} ({correct_new/changed.sum()*100:.1f}%)")
print(f"淨效益:             {correct_new - correct_orig:+d} 筆")

print(f"\n被改動樣本的原始預測（改前）分布:")
print(pd.Series(changed_orig).value_counts().sort_index().rename(
    {i+1: f"label{i+1}:{LABEL_STR_LIST[i][:12]}" for i in range(5)}))
print(f"\n被改動樣本的真實標籤分布:")
print(pd.Series(changed_true).value_counts().sort_index().rename(
    {i+1: f"label{i+1}:{LABEL_STR_LIST[i][:12]}" for i in range(5)}))

# ── 最終結論 ─────────────────────────────────────────────────
print(f"\n{'='*60}")
print("  結論")
print(f"{'='*60}")
IMPROVEMENT_THRESHOLD = 0.001
if best["delta"] > IMPROVEMENT_THRESHOLD:
    print(f"✓ TF-IDF 代理模型顯示 threshold 策略有效（+{best['delta']:.4f}）")
    print(f"  建議在 Kaggle 真實 OOF 上用 τ={best['tau']:.2f} 驗證後再提交")
    print(f"  注意：TF-IDF 和 BERT 的信心分布不同，最佳 τ 可能需要調整")
else:
    print(f"✗ Threshold 策略無效（最大提升 {best['delta']:+.4f}，低於門檻 {IMPROVEMENT_THRESHOLD}）")
    print()
    print("  原因分析：")
    print(f"  低信心的 label 1-4 預測中，只有 {truly_5/max(changed.sum(),1)*100:.1f}% 真的是 label 5")
    print("  改變這些預測對 F1 的負面影響大於正面影響")
    print()
    print("  替代策略：")
    print("  1. 用多數投票清洗（已在 v2 實作）→ 讓模型自己學模糊案例")
    print("  2. DeBERTa ensemble → 架構多樣性更有效")
