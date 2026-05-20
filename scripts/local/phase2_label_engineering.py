"""
Phase 2: Label Description Engineering + Stronger Models

策略：
1. 擴增標籤描述（豐富同義詞、典型疾病名稱）
2. 試驗更強的 NLI 模型（DeBERTa-v3-large）
3. 試驗更好的 SBERT 模型（bge-large-en-v1.5）
4. Soft Ensemble：A + B 機率加權

重點攻克 Phase 1 弱類別：
- general pathological conditions (F1=0.47)
- nervous system diseases (F1=0.49)
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import f1_score, classification_report, confusion_matrix
from itertools import product

warnings.filterwarnings("ignore")

# ── 設定 ──────────────────────────────────────────────────────────────────────

DATA_PATH = "kaggle_trainset.csv"
EVAL_SAMPLE = 1000
BATCH_SIZE_A = 8     # DeBERTa-large 較大，降低 batch size
BATCH_SIZE_B = 64

LABEL_MAP = {
    1: "neoplasms",
    2: "digestive system diseases",
    3: "nervous system diseases",
    4: "cardiovascular diseases",
    5: "general pathological conditions",
}
STR_TO_INT = {v: k for k, v in LABEL_MAP.items()}

# ── 標籤描述實驗池 ────────────────────────────────────────────────────────────
#
# 每種描述策略是一個 dict，key 是數字 label，value 是送給模型的文字描述。
# Phase 1 發現 general pathological 和 nervous system 最弱，這裡重點強化。

LABEL_DESCRIPTIONS = {

    # ── 版本 v1：Phase 1 原始（基準線）
    "v1_original": {
        1: "neoplasms",
        2: "digestive system diseases",
        3: "nervous system diseases",
        4: "cardiovascular diseases",
        5: "general pathological conditions",
    },

    # ── 版本 v2：加入同義詞與典型疾病
    "v2_synonyms": {
        1: "neoplasms, cancer, tumor, malignant, carcinoma, lymphoma, leukemia, oncology",
        2: "digestive system diseases, gastrointestinal, liver disease, hepatic, bowel, gastric, colorectal, pancreatic",
        3: "nervous system diseases, neurological disorder, brain disease, epilepsy, stroke, dementia, Parkinson, Alzheimer, neuropathy",
        4: "cardiovascular diseases, heart disease, cardiac, coronary artery, hypertension, arrhythmia, myocardial infarction, vascular",
        5: "general pathological conditions, inflammatory disease, infection, immune disorder, metabolic syndrome, genetic disorder, systemic disease, endocrine",
    },

    # ── 版本 v3：完整句子描述（對 NLI 更友善）
    "v3_sentences": {
        1: "This medical abstract is about neoplasms, including cancer, malignant tumors, carcinoma, lymphoma, leukemia, or other oncological conditions.",
        2: "This medical abstract is about digestive system diseases, including gastrointestinal disorders, liver disease, hepatitis, bowel disease, gastric conditions, or pancreatic disorders.",
        3: "This medical abstract is about nervous system diseases, including neurological disorders, brain diseases, epilepsy, stroke, dementia, Parkinson's disease, Alzheimer's disease, or spinal cord conditions.",
        4: "This medical abstract is about cardiovascular diseases, including heart disease, coronary artery disease, hypertension, arrhythmia, myocardial infarction, heart failure, or vascular disorders.",
        5: "This medical abstract is about general pathological conditions, including inflammatory diseases, infections, immune disorders, metabolic syndromes, genetic disorders, or systemic diseases not classified elsewhere.",
    },

    # ── 版本 v4：v3 sentences + 加強 general pathological 的負向特異性
    "v4_contrast": {
        1: "This abstract specifically studies cancer, tumor, or malignant neoplasm.",
        2: "This abstract specifically studies a disease of the digestive tract, liver, or gastrointestinal system.",
        3: "This abstract specifically studies a neurological condition, brain disorder, or spinal disease.",
        4: "This abstract specifically studies a heart condition, vascular disease, or cardiovascular disorder.",
        5: "This abstract studies a general medical condition such as inflammation, infection, metabolic disorder, or immune dysfunction, which is not primarily a cancer, digestive, neurological, or cardiovascular disease.",
    },
}

# ── 資料載入 ──────────────────────────────────────────────────────────────────

def load_eval_data(path: str, n: int):
    df = pd.read_csv(path).head(n)
    texts = df["condition"].tolist()
    labels = [STR_TO_INT[lbl] for lbl in df["label"]]
    return texts, labels


# ── 評估工具 ──────────────────────────────────────────────────────────────────

def eval_metrics(y_true, y_pred, verbose=True):
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    if verbose:
        print(classification_report(
            y_true, y_pred,
            target_names=[LABEL_MAP[i] for i in sorted(LABEL_MAP)],
            labels=sorted(LABEL_MAP.keys()),
        ))
    return macro_f1


def plot_cm(y_true, y_pred, title, save_path):
    cm = confusion_matrix(y_true, y_pred, labels=sorted(LABEL_MAP.keys()))
    tick_labels = [f"{k}:{v[:12]}" for k, v in LABEL_MAP.items()]
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=tick_labels, yticklabels=tick_labels)
    plt.title(title)
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def _has_gpu():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# ── Method A：NLI Zero-shot，掃描所有標籤描述版本 ──────────────────────────────

def run_method_a_sweep(texts, y_true, model_name="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"):
    """對所有標籤描述版本執行 NLI zero-shot，返回各版本 F1 和 soft scores。"""
    from transformers import pipeline

    print(f"\n[Method A Sweep] 模型: {model_name}")
    classifier = pipeline(
        "zero-shot-classification",
        model=model_name,
        device=0 if _has_gpu() else -1,
    )

    results = {}

    for version, desc_map in LABEL_DESCRIPTIONS.items():
        candidate_labels = list(desc_map.values())
        # 描述文字 → 原始 label 數字的對應（不一定等於 STR_TO_INT）
        desc_to_int = {desc: num for num, desc in desc_map.items()}

        y_pred = []
        soft_scores = []   # shape: (N, 5)，每筆樣本對 5 個類別的機率

        t0 = time.time()
        for i in range(0, len(texts), BATCH_SIZE_A):
            batch = texts[i: i + BATCH_SIZE_A]
            preds = classifier(batch, candidate_labels=candidate_labels)
            for res in preds:
                # 重新排序 scores，確保順序對應 label 1-5
                score_dict = dict(zip(res["labels"], res["scores"]))
                ordered_scores = [score_dict[desc_map[j]] for j in sorted(LABEL_MAP.keys())]
                soft_scores.append(ordered_scores)
                best = res["labels"][0]
                y_pred.append(desc_to_int[best])
            print(f"  {version} [{i + len(batch)}/{len(texts)}] {time.time()-t0:.1f}s", end="\r")

        print()
        macro_f1 = f1_score(y_true, y_pred, average="macro")
        print(f"  [{version}] Macro F1 = {macro_f1:.4f}")
        results[version] = {
            "f1": macro_f1,
            "y_pred": y_pred,
            "soft_scores": np.array(soft_scores),  # (N, 5)
        }

    # 印出最佳版本
    best_ver = max(results, key=lambda v: results[v]["f1"])
    print(f"\n  最佳標籤描述版本: {best_ver}  F1={results[best_ver]['f1']:.4f}")
    eval_metrics(y_true, results[best_ver]["y_pred"])
    plot_cm(y_true, results[best_ver]["y_pred"],
            f"Method A (DeBERTa) Best — {best_ver}",
            f"outputs/method_a2_cm_{best_ver}.png")

    return results, best_ver


# ── Method B：SBERT，更換模型 + 掃描標籤描述 ─────────────────────────────────

def run_method_b_sweep(texts, y_true, model_name="BAAI/bge-large-en-v1.5"):
    """用新 SBERT 模型 + 所有標籤描述版本，返回各版本 F1 和 soft scores。"""
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    print(f"\n[Method B Sweep] 模型: {model_name}")
    device = "cuda" if _has_gpu() else "cpu"
    model = SentenceTransformer(model_name, device=device)

    results = {}

    # 先 encode 全部 abstracts（只做一次）
    print("  Encoding abstracts ...")
    text_emb = model.encode(texts, batch_size=BATCH_SIZE_B,
                            show_progress_bar=True, normalize_embeddings=True)

    for version, desc_map in LABEL_DESCRIPTIONS.items():
        candidate_descs = [desc_map[j] for j in sorted(LABEL_MAP.keys())]
        label_emb = model.encode(candidate_descs, batch_size=len(candidate_descs),
                                 show_progress_bar=False, normalize_embeddings=True)

        sim_matrix = cosine_similarity(text_emb, label_emb)  # (N, 5)
        best_indices = np.argmax(sim_matrix, axis=1)
        y_pred = [list(LABEL_MAP.keys())[idx] for idx in best_indices]

        macro_f1 = f1_score(y_true, y_pred, average="macro")
        print(f"  [{version}] Macro F1 = {macro_f1:.4f}")
        results[version] = {
            "f1": macro_f1,
            "y_pred": y_pred,
            "soft_scores": sim_matrix,  # (N, 5)
        }

    best_ver = max(results, key=lambda v: results[v]["f1"])
    print(f"\n  最佳標籤描述版本: {best_ver}  F1={results[best_ver]['f1']:.4f}")
    eval_metrics(y_true, results[best_ver]["y_pred"])
    plot_cm(y_true, results[best_ver]["y_pred"],
            f"Method B (bge-large) Best — {best_ver}",
            f"outputs/method_b2_cm_{best_ver}.png")

    return results, best_ver


# ── Soft Ensemble：A 和 B 的機率加權融合 ──────────────────────────────────────

def run_ensemble(texts, y_true, results_a, best_ver_a, results_b, best_ver_b):
    """對各種 A/B 組合 + 權重組合做 grid search，找最佳 ensemble。"""
    print("\n[Ensemble] 搜尋最佳 A/B 組合與權重 ...")

    # 選最佳版本的 soft scores
    scores_a = results_a[best_ver_a]["soft_scores"]  # (N, 5)
    scores_b = results_b[best_ver_b]["soft_scores"]  # (N, 5)

    # Normalize A scores（NLI 輸出是機率，已歸一化；SBERT 是相似度，需歸一化）
    scores_b_norm = scores_b / scores_b.sum(axis=1, keepdims=True)

    best_f1, best_w = 0, None
    for w_a in np.arange(0.3, 1.0, 0.1):
        w_b = 1.0 - w_a
        combined = w_a * scores_a + w_b * scores_b_norm
        y_pred = [list(LABEL_MAP.keys())[idx] for idx in np.argmax(combined, axis=1)]
        f1 = f1_score(y_true, y_pred, average="macro")
        print(f"  w_A={w_a:.1f} w_B={w_b:.1f} → F1={f1:.4f}")
        if f1 > best_f1:
            best_f1, best_w = f1, (w_a, w_b)
            best_pred = y_pred
            best_combined = combined

    print(f"\n  最佳 Ensemble 權重: w_A={best_w[0]:.1f}, w_B={best_w[1]:.1f}  F1={best_f1:.4f}")
    eval_metrics(y_true, best_pred)
    plot_cm(y_true, best_pred,
            f"Ensemble (w_A={best_w[0]:.1f}) — F1={best_f1:.4f}",
            "outputs/ensemble_cm.png")

    np.save("outputs/ensemble_soft_scores.npy", best_combined)
    pd.DataFrame({
        "text": texts,
        "true_label": y_true,
        "pred_label": best_pred,
        "correct": [t == p for t, p in zip(y_true, best_pred)],
    }).to_csv("outputs/ensemble_predictions.csv", index=False, encoding="utf-8-sig")

    return best_f1, best_w, best_combined


# ── 主程式 ────────────────────────────────────────────────────────────────────

def main():
    os.makedirs("outputs", exist_ok=True)

    print("=" * 60)
    print("  Phase 2: Label Engineering + Stronger Models + Ensemble")
    print("=" * 60)
    print(f"GPU available: {_has_gpu()}\n")

    texts, y_true = load_eval_data(DATA_PATH, EVAL_SAMPLE)

    # Phase 1 基準（供對比）
    PHASE1_BEST = 0.5930

    # ── Method A：DeBERTa-v3-large 掃描所有標籤描述
    results_a, best_ver_a = run_method_a_sweep(texts, y_true)

    # ── Method B：bge-large-en-v1.5 掃描所有標籤描述
    results_b, best_ver_b = run_method_b_sweep(texts, y_true)

    # ── Ensemble
    ens_f1, ens_w, _ = run_ensemble(texts, y_true, results_a, best_ver_a, results_b, best_ver_b)

    # ── 總結
    print("\n" + "=" * 60)
    print("  Phase 2 Results Summary")
    print("=" * 60)
    rows = [
        ("Phase 1 Best (BART NLI)", PHASE1_BEST),
        (f"Method A Best ({best_ver_a})", results_a[best_ver_a]["f1"]),
        (f"Method B Best ({best_ver_b})", results_b[best_ver_b]["f1"]),
        (f"Ensemble (w_A={ens_w[0]:.1f})", ens_f1),
    ]
    for name, f1 in rows:
        bar = "█" * int(f1 * 40)
        delta = f1 - PHASE1_BEST
        sign = "+" if delta >= 0 else ""
        print(f"  {name:<40} F1={f1:.4f}  ({sign}{delta:.4f})  {bar}")

    best_f1_overall = max(f1 for _, f1 in rows[1:])
    if best_f1_overall >= 0.7:
        print("\n  → 效果優異！準備進入 Phase 3 全量資料推論並提交")
    elif best_f1_overall >= 0.6:
        print("\n  → 有顯著提升，可考慮 Phase 4 監督式微調進一步衝高")
    else:
        print("\n  → 提升有限，強烈建議進入 Phase 4 監督式微調")

    # 儲存最佳設定供 Phase 5 使用
    pd.DataFrame([{
        "best_method_a_model": "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
        "best_method_a_desc_version": best_ver_a,
        "best_method_b_model": "BAAI/bge-large-en-v1.5",
        "best_method_b_desc_version": best_ver_b,
        "ensemble_w_a": ens_w[0],
        "ensemble_w_b": ens_w[1],
        "ensemble_f1_eval": ens_f1,
    }]).to_csv("outputs/phase2_best_config.csv", index=False)
    print("\n  最佳設定已儲存 → outputs/phase2_best_config.csv")


if __name__ == "__main__":
    main()
