"""
Phase 1: Unsupervised Baseline
- Method A: Zero-shot Classification (NLI-based)
- Method B: Similarity-based (Sentence-BERT)

評估：對 kaggle_trainset.csv 前 1000 筆計算 Macro F1-Score
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import f1_score, classification_report, confusion_matrix

warnings.filterwarnings("ignore")

# ── 設定 ──────────────────────────────────────────────────────────────────────

DATA_PATH = "kaggle_trainset.csv"
EVAL_SAMPLE = 1000   # 用前 N 筆做快速評估
BATCH_SIZE_A = 16    # Method A (NLI) 的 batch size
BATCH_SIZE_B = 64    # Method B (SBERT) 的 batch size

# Label mapping: 數字 → 疾病描述（提交格式使用數字）
LABEL_MAP = {
    1: "neoplasms",
    2: "digestive system diseases",
    3: "nervous system diseases",
    4: "cardiovascular diseases",
    5: "general pathological conditions",
}
CANDIDATE_LABELS = list(LABEL_MAP.values())
STR_TO_INT = {v: k for k, v in LABEL_MAP.items()}  # 文字描述 → 數字（用於 trainset）
INT_TO_STR = LABEL_MAP                              # 數字 → 文字描述

# ── 資料載入 ──────────────────────────────────────────────────────────────────

def load_eval_data(path: str, n: int) -> tuple[list[str], list[int]]:
    df = pd.read_csv(path)
    print(f"[Data] 共 {len(df)} 筆，取前 {n} 筆做評估")
    print(f"[Data] 欄位: {df.columns.tolist()}")
    print(f"[Data] 標籤分布:\n{df['label'].value_counts()}\n")

    df_eval = df.head(n).copy()

    # condition 欄是 abstract 文本；label 欄是疾病文字描述，需轉為數字
    texts = df_eval["condition"].tolist()
    labels = [STR_TO_INT[lbl] for lbl in df_eval["label"]]
    return texts, labels


# ── 共用工具 ──────────────────────────────────────────────────────────────────

def print_results(method_name: str, y_true: list, y_pred: list) -> float:
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    print(f"\n{'='*60}")
    print(f"  {method_name}")
    print(f"{'='*60}")
    print(f"  Macro F1-Score: {macro_f1:.4f}")
    print()
    print(classification_report(
        y_true, y_pred,
        target_names=[LABEL_MAP[i] for i in sorted(LABEL_MAP)],
        labels=sorted(LABEL_MAP.keys()),
    ))
    return macro_f1


def plot_confusion_matrix(y_true: list, y_pred: list, title: str, save_path: str):
    cm = confusion_matrix(y_true, y_pred, labels=sorted(LABEL_MAP.keys()))
    tick_labels = [f"{k}:{v[:12]}" for k, v in LABEL_MAP.items()]

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=tick_labels, yticklabels=tick_labels,
    )
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Plot] Confusion matrix saved → {save_path}")


# ── Method A: Zero-shot (NLI) ─────────────────────────────────────────────────

def run_method_a(texts: list[str], y_true: list[int]) -> float:
    print("\n[Method A] 載入 Zero-shot pipeline ...")
    from transformers import pipeline

    # facebook/bart-large-mnli 是業界廣泛使用的 zero-shot 基準
    # 可換成 "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli" 取得更高精度
    MODEL_A = "facebook/bart-large-mnli"
    classifier = pipeline(
        "zero-shot-classification",
        model=MODEL_A,
        device=0 if _has_gpu() else -1,  # 有 GPU 自動用 GPU
    )
    print(f"[Method A] 模型: {MODEL_A}  |  batch_size: {BATCH_SIZE_A}")

    y_pred = []
    total = len(texts)
    t0 = time.time()

    for i in range(0, total, BATCH_SIZE_A):
        batch = texts[i : i + BATCH_SIZE_A]
        results = classifier(batch, candidate_labels=CANDIDATE_LABELS)

        for res in results:
            best_label = res["labels"][0]          # 最高分的類別描述
            pred_id = STR_TO_INT[best_label]       # 描述 → 數字
            y_pred.append(pred_id)

        elapsed = time.time() - t0
        print(f"  [{i + len(batch)}/{total}]  已耗時 {elapsed:.1f}s", end="\r")

    print()
    macro_f1 = print_results("Method A: Zero-shot (NLI) — " + MODEL_A, y_true, y_pred)
    plot_confusion_matrix(y_true, y_pred, "Method A — Confusion Matrix", "outputs/method_a_cm.png")

    # 儲存預測結果
    _save_predictions("outputs/method_a_predictions.csv", texts, y_true, y_pred)
    return macro_f1


# ── Method B: Similarity-based (SBERT) ───────────────────────────────────────

def run_method_b(texts: list[str], y_true: list[int]) -> float:
    print("\n[Method B] 載入 Sentence-BERT 模型 ...")
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    # pritamdeka/S-PubMedBert-MS-MARCO 專為醫學文本設計
    # 備選: "sentence-transformers/all-mpnet-base-v2"（通用強基準）
    MODEL_B = "pritamdeka/S-PubMedBert-MS-MARCO"
    device = "cuda" if _has_gpu() else "cpu"
    model = SentenceTransformer(MODEL_B, device=device)
    print(f"[Method B] 模型: {MODEL_B}  |  device: {device}  |  batch_size: {BATCH_SIZE_B}")

    # 將 5 個類別描述 encode 為向量（只做一次）
    print("[Method B] Encoding 類別描述 ...")
    label_embeddings = model.encode(
        CANDIDATE_LABELS,
        batch_size=len(CANDIDATE_LABELS),
        show_progress_bar=False,
        normalize_embeddings=True,  # L2 normalize → cosine sim = dot product
    )  # shape: (5, hidden_dim)

    # 分批 encode abstracts
    print("[Method B] Encoding abstracts ...")
    text_embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE_B,
        show_progress_bar=True,
        normalize_embeddings=True,
    )  # shape: (N, hidden_dim)

    # 計算 cosine similarity 矩陣：(N, 5)
    sim_matrix = cosine_similarity(text_embeddings, label_embeddings)

    # 取相似度最高的類別
    best_indices = np.argmax(sim_matrix, axis=1)  # shape: (N,)
    y_pred = [list(LABEL_MAP.keys())[idx] for idx in best_indices]  # index → 數字 label

    macro_f1 = print_results("Method B: Similarity-based (SBERT) — " + MODEL_B, y_true, y_pred)
    plot_confusion_matrix(y_true, y_pred, "Method B — Confusion Matrix", "outputs/method_b_cm.png")

    # 儲存含 similarity score 的詳細結果
    _save_predictions_with_scores(
        "outputs/method_b_predictions.csv",
        texts, y_true, y_pred, sim_matrix
    )
    return macro_f1


# ── 輔助函式 ─────────────────────────────────────────────────────────────────

def _has_gpu() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _save_predictions(path: str, texts, y_true, y_pred):
    df = pd.DataFrame({
        "text": texts,
        "true_label": y_true,
        "pred_label": y_pred,
        "correct": [t == p for t, p in zip(y_true, y_pred)],
    })
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[Save] {path}")


def _save_predictions_with_scores(path: str, texts, y_true, y_pred, sim_matrix):
    df = pd.DataFrame({
        "text": texts,
        "true_label": y_true,
        "pred_label": y_pred,
        "correct": [t == p for t, p in zip(y_true, y_pred)],
    })
    for i, label_name in enumerate(CANDIDATE_LABELS):
        df[f"sim_{i+1}_{label_name[:10]}"] = sim_matrix[:, i]
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[Save] {path}")


# ── 主程式 ────────────────────────────────────────────────────────────────────

def main():
    os.makedirs("outputs", exist_ok=True)

    print("=" * 60)
    print("  Phase 1: Unsupervised Baseline Evaluation")
    print("=" * 60)
    print(f"GPU available: {_has_gpu()}\n")

    texts, y_true = load_eval_data(DATA_PATH, EVAL_SAMPLE)

    results = {}

    # ── Method A ──
    try:
        results["Method A (Zero-shot NLI)"] = run_method_a(texts, y_true)
    except Exception as e:
        print(f"[Error] Method A 失敗: {e}")

    # ── Method B ──
    try:
        results["Method B (SBERT Similarity)"] = run_method_b(texts, y_true)
    except Exception as e:
        print(f"[Error] Method B 失敗: {e}")

    # ── 總結比較 ──
    print("\n" + "=" * 60)
    print("  Phase 1 Results Summary")
    print("=" * 60)
    for method, f1 in results.items():
        bar = "█" * int(f1 * 40)
        print(f"  {method:<35} F1={f1:.4f}  {bar}")

    best_method = max(results, key=results.get)
    print(f"\n  最佳方法: {best_method}  (F1={results[best_method]:.4f})")

    if results[best_method] >= 0.7:
        print("  → 無監督效果優異，繼續 Phase 2 標籤描述優化")
    elif results[best_method] >= 0.5:
        print("  → 無監督有一定效果，建議 Phase 2 + Phase 3 Ensemble")
    else:
        print("  → 無監督效果有限，考慮提前進入 Phase 4 監督式微調")

    print("\n  所有輸出已儲存至 outputs/ 資料夾")


if __name__ == "__main__":
    main()
