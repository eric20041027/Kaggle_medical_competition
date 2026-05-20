"""
Phase 5: Final Ensemble + Submission

組合：
  - Phase 4 BiomedBERT (fine-tuned, best checkpoint)
  - Phase 2 Ensemble soft scores (DeBERTa v4_contrast + bge-large v3_sentences)

流程：
  1. 用 Phase 4 模型對 val set 推論 → 與 Phase 2 val soft scores 做 grid search
  2. 用 Phase 4 模型對 test set 推論
  3. 重跑 Phase 2 最佳組合對 test set 推論
  4. 套用最佳權重 → 生成最終 submission
"""

import os
import warnings
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import f1_score, classification_report
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── 設定（與 phase4 保持一致）────────────────────────────────────────────────

TRAIN_PATH = "kaggle_trainset.csv"
TEST_PATH  = "kaggle_testset.csv"
MODEL_NAME = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
BEST_MODEL_PATH = "outputs/phase4/best_model.pt"
PHASE2_VAL_SCORES = "outputs/ensemble_soft_scores.npy"   # shape (1000, 5)
OUTPUT_DIR = "outputs/phase5"

MAX_LEN    = 512
BATCH_SIZE = 32
EVAL_N     = 1000    # Phase 2 val scores 對應的前 N 筆

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABEL_STR_LIST = [
    "neoplasms",
    "digestive system diseases",
    "nervous system diseases",
    "cardiovascular diseases",
    "general pathological conditions",
]
STR_TO_IDX   = {s: i for i, s in enumerate(LABEL_STR_LIST)}
IDX_TO_SUBMIT = {i: i + 1 for i in range(5)}

# Phase 2 最佳設定
DEBERTA_MODEL  = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
BGE_MODEL      = "BAAI/bge-large-en-v1.5"

LABEL_DESC_A = {   # v4_contrast (DeBERTa 最佳)
    1: "This abstract specifically studies cancer, tumor, or malignant neoplasm.",
    2: "This abstract specifically studies a disease of the digestive tract, liver, or gastrointestinal system.",
    3: "This abstract specifically studies a neurological condition, brain disorder, or spinal disease.",
    4: "This abstract specifically studies a heart condition, vascular disease, or cardiovascular disorder.",
    5: "This abstract studies a general medical condition such as inflammation, infection, metabolic disorder, or immune dysfunction, which is not primarily a cancer, digestive, neurological, or cardiovascular disease.",
}
LABEL_DESC_B = {   # v3_sentences (bge-large 最佳)
    1: "This medical abstract is about neoplasms, including cancer, malignant tumors, carcinoma, lymphoma, leukemia, or other oncological conditions.",
    2: "This medical abstract is about digestive system diseases, including gastrointestinal disorders, liver disease, hepatitis, bowel disease, gastric conditions, or pancreatic disorders.",
    3: "This medical abstract is about nervous system diseases, including neurological disorders, brain diseases, epilepsy, stroke, dementia, Parkinson's disease, Alzheimer's disease, or spinal cord conditions.",
    4: "This medical abstract is about cardiovascular diseases, including heart disease, coronary artery disease, hypertension, arrhythmia, myocardial infarction, heart failure, or vascular disorders.",
    5: "This medical abstract is about general pathological conditions, including inflammatory diseases, infections, immune disorders, metabolic syndromes, genetic disorders, or systemic diseases not classified elsewhere.",
}

# ── Dataset ───────────────────────────────────────────────────────────────────

class MedDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx], max_length=MAX_LEN,
            padding="max_length", truncation=True, return_tensors="pt",
        )
        item = {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ── Phase 4 推論 ───────────────────────────────────────────────────────────────

def get_phase4_logits(texts, labels=None, desc=""):
    """回傳 softmax 機率 shape (N, 5)"""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=5
    )
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE).eval()

    ds = MedDataset(texts, labels, tokenizer)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    all_probs = []
    print(f"[Phase4] {desc} 推論中 ({len(texts)} 筆) ...")
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Phase4 {desc}", leave=True):
            ids  = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            logits = model(input_ids=ids, attention_mask=mask).logits
            probs = torch.softmax(logits, dim=-1)
            all_probs.append(probs.cpu().numpy())

    del model
    torch.cuda.empty_cache()
    return np.vstack(all_probs)  # (N, 5)


# ── Phase 2 Method A（DeBERTa NLI）推論 ──────────────────────────────────────

def get_deberta_scores(texts, desc=""):
    """回傳每類別的 NLI 機率 shape (N, 5)"""
    from transformers import pipeline
    print(f"[Phase2-A] DeBERTa {desc} 推論中 ({len(texts)} 筆) ...")
    clf = pipeline("zero-shot-classification", model=DEBERTA_MODEL,
                   device=0 if torch.cuda.is_available() else -1)
    candidates = [LABEL_DESC_A[j] for j in sorted(LABEL_DESC_A)]
    all_scores = []
    batches = [texts[i: i + 8] for i in range(0, len(texts), 8)]
    for batch in tqdm(batches, desc=f"DeBERTa {desc}", leave=True):
        results = clf(batch, candidate_labels=candidates)
        for res in results:
            score_dict = dict(zip(res["labels"], res["scores"]))
            row = [score_dict[LABEL_DESC_A[j]] for j in sorted(LABEL_DESC_A)]
            all_scores.append(row)
    del clf
    torch.cuda.empty_cache()
    return np.array(all_scores)  # (N, 5)


# ── Phase 2 Method B（bge-large SBERT）推論 ───────────────────────────────────

def get_sbert_scores(texts, desc=""):
    """回傳 cosine similarity shape (N, 5)，已 L2 normalize"""
    from sentence_transformers import SentenceTransformer
    print(f"[Phase2-B] bge-large {desc} 推論中 ({len(texts)} 筆) ...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(BGE_MODEL, device=device)
    label_descs = [LABEL_DESC_B[j] for j in sorted(LABEL_DESC_B)]
    label_emb = model.encode(label_descs, normalize_embeddings=True, show_progress_bar=False)
    text_emb  = model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True)
    sim = cosine_similarity(text_emb, label_emb)  # (N, 5)
    del model
    torch.cuda.empty_cache()
    return sim


def normalize_rows(arr):
    """每列歸一化為機率分布（SBERT sim → pseudo-prob）"""
    mins = arr.min(axis=1, keepdims=True)
    shifted = arr - mins
    row_sum = shifted.sum(axis=1, keepdims=True) + 1e-9
    return shifted / row_sum


# ── Ensemble Grid Search ──────────────────────────────────────────────────────

def grid_search_weights(y_true, p4_probs, p2_probs):
    """
    搜尋最佳 (w4, w2) 使 val Macro F1 最高
    p4_probs: (N, 5) Phase 4 softmax
    p2_probs: (N, 5) Phase 2 ensemble（已歸一化）
    """
    best_f1, best_w = 0.0, None
    print("\n[Ensemble] Grid search 最佳權重 ...")
    for w4 in np.arange(0.1, 1.0, 0.1):
        w2 = 1.0 - w4
        combined = w4 * p4_probs + w2 * p2_probs
        pred = [list(IDX_TO_SUBMIT.keys())[i] for i in np.argmax(combined, axis=1)]
        pred_submit = [IDX_TO_SUBMIT[p] for p in pred]
        f1 = f1_score(y_true, pred_submit, average="macro")
        print(f"  w_P4={w4:.1f} w_P2={w2:.1f} → F1={f1:.4f}")
        if f1 > best_f1:
            best_f1, best_w = f1, (w4, w2)
    print(f"\n  最佳: w_P4={best_w[0]:.1f}, w_P2={best_w[1]:.1f}  F1={best_f1:.4f}")
    return best_f1, best_w


# ── 主程式 ────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"[Config] Device: {DEVICE}\n")

    # ── 1. 載入 val set（前 1000 筆，與 Phase 2 對齊）
    train_df = pd.read_csv(TRAIN_PATH)
    val_df   = train_df.head(EVAL_N)
    val_texts  = val_df["condition"].tolist()
    val_labels = [IDX_TO_SUBMIT[STR_TO_IDX[lbl]] for lbl in val_df["label"]]  # 1-based

    # ── 2. Phase 4 val 推論
    p4_val = get_phase4_logits(val_texts, desc="Val")

    # ── 3. 載入 Phase 2 val soft scores（已儲存）
    p2_val_raw = np.load(PHASE2_VAL_SCORES)   # shape (1000, 5)
    p2_val = normalize_rows(p2_val_raw)

    # ── 4. Grid search 最佳權重
    best_f1_val, best_w = grid_search_weights(val_labels, p4_val, p2_val)

    print(f"\n[Eval] Ensemble Val Macro F1 = {best_f1_val:.4f}")
    w4, w2 = best_w
    combined_val = w4 * p4_val + w2 * p2_val
    pred_val = [IDX_TO_SUBMIT[i] for i in np.argmax(combined_val, axis=1)]
    print(classification_report(
        val_labels, pred_val,
        target_names=[f"{i+1}:{LABEL_STR_LIST[i][:12]}" for i in range(5)],
        labels=[1, 2, 3, 4, 5],
    ))

    # ── 5. Test set 推論
    test_df   = pd.read_csv(TEST_PATH)
    test_texts = test_df["condition"].tolist()

    print("\n=== Test Set 推論 ===")
    p4_test = get_phase4_logits(test_texts, desc="Test")

    # Phase 2 test: DeBERTa + bge-large
    p2a_test = get_deberta_scores(test_texts, desc="Test")
    p2b_test = get_sbert_scores(test_texts,   desc="Test")

    # Phase 2 test ensemble（使用與 Phase 2 相同的最佳權重 w_A=0.3, w_B=0.7）
    p2b_test_norm = normalize_rows(p2b_test)
    p2a_test_norm = p2a_test  # NLI 輸出已是機率
    p2_test = 0.3 * p2a_test_norm + 0.7 * p2b_test_norm
    p2_test = normalize_rows(p2_test)

    # ── 6. 最終 Ensemble
    final_scores = w4 * p4_test + w2 * p2_test
    final_pred_idx    = np.argmax(final_scores, axis=1)
    final_pred_submit = [IDX_TO_SUBMIT[i] for i in final_pred_idx]

    # ── 7. 生成提交檔
    submission = pd.read_csv("kaggle_testset_submission.csv")
    submission["label"] = final_pred_submit
    submit_path = os.path.join(OUTPUT_DIR, "final_submission.csv")
    submission.to_csv(submit_path, index=False)
    print(f"\n[Submit] 最終提交檔 → {submit_path}")
    print(f"[Submit] 預測分布:\n{pd.Series(final_pred_submit).value_counts().sort_index()}")

    # ── 也輸出純 Phase 4 單模型提交（供比較）
    p4_only_pred = [IDX_TO_SUBMIT[i] for i in np.argmax(p4_test, axis=1)]
    sub_p4 = pd.read_csv("kaggle_testset_submission.csv")
    sub_p4["label"] = p4_only_pred
    sub_p4.to_csv(os.path.join(OUTPUT_DIR, "submission_phase4_only.csv"), index=False)
    print(f"[Submit] Phase4 單模型提交檔（備用）→ outputs/phase5/submission_phase4_only.csv")

    print(f"\n{'='*60}")
    print(f"  最終結果 (Val Macro F1) = {best_f1_val:.4f}")
    print(f"  Phase 4 單模型 Val F1  = 0.6397  (Epoch 1 最佳)")
    print(f"  Ensemble 提升          = {best_f1_val - 0.6397:+.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
