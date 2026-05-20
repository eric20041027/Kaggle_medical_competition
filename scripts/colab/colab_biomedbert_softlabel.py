"""
BiomedBERT-large + FGM + 軟標籤訓練（Colab 版）
================================================
核心改動：
  - 衝突樣本改用軟標籤（50/50 → [0,0,0,0.5,0.5]），不再隨機選一邊
  - 損失函數：KL divergence + per-sample class weight
  - 其餘沿用最佳模型設定：原始 12,994 筆、MAX_LEN=512、FGM

【執行步驟】
  # Cell 1
  from google.colab import drive
  drive.mount('/content/drive')

  # Cell 2
  import os; os.chdir('/content')
  !git clone https://github.com/eric20041027/Kaggle_medical_competition.git
  os.chdir('/content/Kaggle_medical_competition')
  !pip install -q transformers torch scikit-learn pandas numpy tqdm

  # Cell 3
  !python colab_biomedbert_softlabel.py

  # Cell 4
  from google.colab import files
  files.download('/content/drive/MyDrive/kaggle_biomedbert_softlabel/submission_biomedbert_softlabel.csv')
  files.download('/content/drive/MyDrive/kaggle_biomedbert_softlabel/test_probs_biomedbert_softlabel.npy')
"""

import os, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.cuda.amp import GradScaler
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_cosine_schedule_with_warmup,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

REPO_DIR    = "."
TRAIN_PATH  = os.path.join(REPO_DIR, "kaggle_trainset.csv")
TEST_PATH   = os.path.join(REPO_DIR, "kaggle_testset.csv")
SUBMIT_PATH = os.path.join(REPO_DIR, "kaggle_testset_submission.csv")

_DRIVE_DIR = "/content/drive/MyDrive/kaggle_biomedbert_softlabel"
_LOCAL_DIR = "outputs/colab_biomedbert_softlabel"
OUTPUT_DIR = _DRIVE_DIR if os.path.exists("/content/drive/MyDrive") else _LOCAL_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(_LOCAL_DIR, exist_ok=True)

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_GPUS   = torch.cuda.device_count()
USE_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
USE_AMP  = torch.cuda.is_available() and not USE_BF16
print(f"Device: {DEVICE}  |  GPUs: {N_GPUS}  |  BF16: {USE_BF16}  |  FP16 AMP: {USE_AMP}")
if torch.cuda.is_available():
    for i in range(N_GPUS):
        mem = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}  ({mem:.1f} GB)")

MODEL_NAME     = "microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract"
MAX_LEN        = 512
_is_a100       = torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory > 30e9
BATCH_SIZE     = 16 if _is_a100 else 8
GRAD_ACCUM     = 4  if _is_a100 else 8
EPOCHS         = 6
LR             = 8e-6
WARMUP_RATIO   = 0.20
LABEL_SMOOTH   = 0.1
VAL_RATIO      = 0.2
PATIENCE       = 2
FGM_EPSILON    = 1.0
CLASS5_BOOST   = 1.0   # 不額外 boost，軟標籤本身已修正 class5 訓練信號
CLASS5_INF_MUL = 1.0   # 不套用推論乘數，避免雙重 boost
SEED           = 42

print(f"BATCH_SIZE={BATCH_SIZE}  GRAD_ACCUM={GRAD_ACCUM}  MAX_LEN={MAX_LEN}")
print(f"FGM_EPSILON={FGM_EPSILON}  CLASS5_BOOST={CLASS5_BOOST}  CLASS5_INF_MUL={CLASS5_INF_MUL}")

LABEL_STR_LIST = [
    "neoplasms",
    "digestive system diseases",
    "nervous system diseases",
    "cardiovascular diseases",
    "general pathological conditions",
]
STR_TO_IDX    = {s: i for i, s in enumerate(LABEL_STR_LIST)}
IDX_TO_SUBMIT = {i: i + 1 for i in range(5)}


# ── 軟標籤建構 ────────────────────────────────────────────────────────
def build_soft_labels(train_df):
    """
    對每個 condition 統計各標籤的票數比例，回傳 dict[condition → np.array(5)]
    - 無衝突：[0, 0, 0, 1, 0]（one-hot）
    - 50/50：[0, 0, 0, 0.5, 0.5]
    - 三方：[0, 0, 0, 0.67, 0.33] 等比例
    """
    label_counts = (
        train_df.groupby(["condition", "label"])
        .size()
        .reset_index(name="cnt")
    )
    cond_to_soft = {}
    for cond, grp in label_counts.groupby("condition"):
        total = grp["cnt"].sum()
        soft  = np.zeros(5, dtype=np.float32)
        for _, row in grp.iterrows():
            soft[STR_TO_IDX[row["label"]]] += row["cnt"] / total
        cond_to_soft[cond] = soft
    return cond_to_soft


class MedicalDataset(Dataset):
    def __init__(self, texts, soft_labels, tokenizer):
        self.texts       = texts
        self.soft_labels = soft_labels   # list of np.array(5), None = test set
        self.tokenizer   = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item = {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }
        if self.soft_labels is not None:
            item["soft_label"] = torch.tensor(self.soft_labels[idx], dtype=torch.float)
        return item


class FGM:
    def __init__(self, model, emb_name="embeddings.word_embeddings", epsilon=1.0):
        self.model    = model
        self.emb_name = emb_name
        self.epsilon  = epsilon
        self.backup   = {}

    def attack(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    param.data.add_(self.epsilon * param.grad / norm)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                param.data = self.backup[name]
        self.backup = {}


def soft_label_loss(logits, soft_targets, class_weights, label_smooth=0.1):
    """
    KL divergence loss（支援軟標籤）+ per-sample class weight
    soft_targets: (N, C) float tensor，可以是 one-hot 或機率分布
    class_weights: (C,) float tensor
    """
    n_classes = soft_targets.size(-1)

    # Label smoothing：混合均勻分布
    if label_smooth > 0:
        soft_targets = (1.0 - label_smooth) * soft_targets + label_smooth / n_classes

    log_probs = F.log_softmax(logits, dim=-1)

    # Per-sample weight = 軟標籤對 class weight 的期望值，再正規化
    sample_w = (soft_targets * class_weights.unsqueeze(0)).sum(dim=-1)
    sample_w = sample_w / sample_w.mean()

    # Cross-entropy with soft targets per sample
    loss_per = -(soft_targets * log_probs).sum(dim=-1)

    return (loss_per * sample_w).mean()


torch.manual_seed(SEED); np.random.seed(SEED)

# ── 載入資料並建構軟標籤 ───────────────────────────────────────────
print("\n載入原始資料並建構軟標籤...")
train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)

cond_to_soft = build_soft_labels(train_df)

# 統計軟標籤類型
n_hard     = sum(1 for s in cond_to_soft.values() if s.max() == 1.0)
n_soft     = len(cond_to_soft) - n_hard
print(f"唯一 condition: {len(cond_to_soft)}  |  純 one-hot: {n_hard}  |  軟標籤: {n_soft}")

# 原始 12,994 筆，每筆對應其 condition 的軟標籤
all_texts       = train_df["condition"].tolist()
all_soft_labels = [cond_to_soft[c] for c in all_texts]
# 用於 val 評估的 argmax hard label（計算 F1 用）
all_hard_labels = [int(s.argmax()) for s in all_soft_labels]
test_texts      = test_df["condition"].tolist()

print(f"訓練集: {len(all_texts)} 筆  測試集: {len(test_texts)} 筆")
print(f"軟標籤分布（衝突樣本的主標籤）:")
import collections
print(collections.Counter(all_hard_labels))

# Class weights（用 hard label argmax 估算，再 boost class 5）
cw = compute_class_weight("balanced", classes=np.arange(5), y=all_hard_labels)
cw[4] *= CLASS5_BOOST
print(f"\nClass weights: {np.round(cw, 3)}")

print("\n載入 tokenizer ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

test_ds     = MedicalDataset(test_texts, None, tokenizer)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)


def build_model():
    base = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=5, torch_dtype=torch.float32
    )
    base.gradient_checkpointing_enable()
    m = nn.DataParallel(base) if N_GPUS > 1 else base
    m = m.to(DEVICE)
    return base, m


@torch.no_grad()
def get_probs(model, loader, desc=""):
    model.eval()
    all_probs = []
    for batch in tqdm(loader, desc=desc, leave=False):
        ids  = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=USE_BF16):
            logits = model(input_ids=ids, attention_mask=mask).logits
        probs = torch.softmax(logits.float(), dim=-1)
        all_probs.append(probs.cpu().numpy())
    return np.vstack(all_probs)


def train_phase(phase_name, tr_texts, tr_soft, tr_hard,
                vl_texts, vl_soft, vl_hard, fixed_epochs=None):
    print(f"\n{'='*60}\n  {phase_name}\n{'='*60}")

    tr_ds = MedicalDataset(tr_texts, tr_soft, tokenizer)
    tr_ld = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)

    has_val = len(vl_texts) > 0
    if has_val:
        vl_ds = MedicalDataset(vl_texts, vl_soft, tokenizer)
        vl_ld = DataLoader(vl_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    n_epochs     = fixed_epochs if fixed_epochs else EPOCHS
    base_model, model = build_model()
    class_weights_t   = torch.tensor(cw, dtype=torch.float).to(DEVICE)

    total_steps  = (len(tr_ld) // GRAD_ACCUM) * n_epochs
    warmup_steps = int(total_steps * WARMUP_RATIO)
    optimizer    = AdamW(base_model.parameters(), lr=LR, weight_decay=0.01)
    scheduler    = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    scaler  = GradScaler(enabled=USE_AMP)
    fgm     = FGM(base_model, epsilon=FGM_EPSILON)

    print(f"Total steps={total_steps}  Warmup={warmup_steps}")
    print(f"Effective batch={BATCH_SIZE * GRAD_ACCUM * max(N_GPUS, 1)}")

    best_f1    = 0.0
    no_improve = 0
    best_epoch = 1
    best_path  = os.path.join(OUTPUT_DIR, f"best_{phase_name[:6]}.pt")

    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()
        pbar = tqdm(tr_ld, desc=f"Epoch {epoch}/{n_epochs}", leave=True)

        for step, batch in enumerate(pbar):
            ids         = batch["input_ids"].to(DEVICE)
            mask        = batch["attention_mask"].to(DEVICE)
            soft_target = batch["soft_label"].to(DEVICE)   # (B, 5)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=USE_BF16):
                logits = model(input_ids=ids, attention_mask=mask).logits
            loss = soft_label_loss(
                logits.float(), soft_target, class_weights_t, LABEL_SMOOTH
            ) / GRAD_ACCUM

            if USE_AMP: scaler.scale(loss).backward()
            else:       loss.backward()

            # FGM
            fgm.attack()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=USE_BF16):
                logits_adv = model(input_ids=ids, attention_mask=mask).logits
            loss_adv = soft_label_loss(
                logits_adv.float(), soft_target, class_weights_t, LABEL_SMOOTH
            ) / GRAD_ACCUM
            if USE_AMP: scaler.scale(loss_adv).backward()
            else:       loss_adv.backward()
            fgm.restore()

            total_loss += loss.item() * GRAD_ACCUM

            if (step + 1) % GRAD_ACCUM == 0:
                if USE_AMP:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(base_model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(base_model.parameters(), 1.0)
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            pbar.set_postfix({"loss": f"{total_loss/(step+1):.4f}"})

        avg_loss = total_loss / len(tr_ld)

        if has_val:
            val_probs = get_probs(model, vl_ld, desc="Val")
            # 用 hard label 計算 F1（argmax of predicted probs vs argmax of soft target）
            val_pred = val_probs.argmax(axis=1).tolist()
            val_f1   = f1_score(vl_hard, val_pred, average="macro")
            improved = val_f1 > best_f1
            status   = "✓ 新最佳" if improved else f"no_improve={no_improve+1}/{PATIENCE}"
            print(f"Epoch {epoch}/{n_epochs}  loss={avg_loss:.4f}  val_F1={val_f1:.4f}  {status}")
            if improved:
                best_f1    = val_f1
                best_epoch = epoch
                no_improve = 0
                torch.save(base_model.state_dict(), best_path)
            else:
                no_improve += 1
                if no_improve >= PATIENCE and not fixed_epochs:
                    print("  [EarlyStop]")
                    break
        else:
            print(f"Epoch {epoch}/{n_epochs}  loss={avg_loss:.4f}")
            torch.save(base_model.state_dict(), best_path)

    if has_val:
        print(f"\n最佳 epoch={best_epoch}  Val F1={best_f1:.4f}")
        base_model.load_state_dict(torch.load(best_path, map_location=DEVICE))
        val_probs = get_probs(model, vl_ld, desc="Val Final")
        val_pred  = val_probs.argmax(axis=1).tolist()
        label_names = [f"{i+1}:{LABEL_STR_LIST[i][:14]}" for i in range(5)]
        print(classification_report(vl_hard, val_pred, target_names=label_names))

        # 顯示軟標籤 vs hard label 的 val 衝突樣本分析
        n_conflict_val = sum(1 for s in vl_soft if s.max() < 1.0)
        print(f"Val 中含衝突軟標籤的樣本: {n_conflict_val}/{len(vl_soft)}")
    else:
        base_model.load_state_dict(torch.load(best_path, map_location=DEVICE))
        best_epoch = fixed_epochs

    return base_model, model, best_epoch


# ── Train/val split（用 hard label 做 stratify）────────────────────
idx = list(range(len(all_texts)))
tr_idx, vl_idx = train_test_split(
    idx, test_size=VAL_RATIO, stratify=all_hard_labels, random_state=SEED
)

tr_texts = [all_texts[i]       for i in tr_idx]
tr_soft  = [all_soft_labels[i] for i in tr_idx]
tr_hard  = [all_hard_labels[i] for i in tr_idx]
vl_texts = [all_texts[i]       for i in vl_idx]
vl_soft  = [all_soft_labels[i] for i in vl_idx]
vl_hard  = [all_hard_labels[i] for i in vl_idx]

print(f"\nTrain: {len(tr_texts)}  Val: {len(vl_texts)}")

# ── 單階段訓練（val split，早停後直接推論）────────────────────────────
_, best_model, best_epoch = train_phase(
    "Training", tr_texts, tr_soft, tr_hard, vl_texts, vl_soft, vl_hard
)
print(f"\n最佳 epoch = {best_epoch}，直接使用此模型推論（節省時間）")

# ── 推論（class 5 × 1.9）────────────────────────────────────────────
print("\n推論測試集 ...")
test_probs     = get_probs(best_model, test_loader, desc="Test")
test_probs_adj = test_probs.copy()
test_probs_adj[:, 4] *= CLASS5_INF_MUL
test_pred_submit = [IDX_TO_SUBMIT[i] for i in test_probs_adj.argmax(axis=1)]

submission = pd.read_csv(SUBMIT_PATH)
submission["label"] = test_pred_submit
sub_path = os.path.join(OUTPUT_DIR, "submission_biomedbert_softlabel.csv")
submission.to_csv(sub_path, index=False)
np.save(os.path.join(OUTPUT_DIR, "test_probs_biomedbert_softlabel.npy"), test_probs)

print(f"\n提交檔 → {sub_path}")
print(f"預測分布:\n{pd.Series(test_pred_submit).value_counts().sort_index()}")
print(f"\n完成！BiomedBERT + FGM + 軟標籤")
