"""
BiomedBERT-large + FGM + Class5 Boost（Colab 版）
==================================================
核心改動（基於最高分腳本分析）：
  1. 原始資料（不去重，保持與測試集相同分布）
  2. MAX_LEN = 512
  3. FGM 對抗訓練
  4. Class 5 loss weight × 2.5x
  5. 推論時 class 5 × 1.9x
  6. Phase 2：確認最佳 epoch 後全量重訓

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
  !python colab_biomedbert_fgm_v2.py

  # Cell 4
  from google.colab import files
  files.download('/content/drive/MyDrive/kaggle_biomedbert_fgm_v2/submission_biomedbert_fgm_v2.csv')
  files.download('/content/drive/MyDrive/kaggle_biomedbert_fgm_v2/test_probs_biomedbert_fgm_v2.npy')
"""

import os, time, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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
from tqdm import tqdm

warnings.filterwarnings("ignore")

REPO_DIR    = "."
TRAIN_PATH  = os.path.join(REPO_DIR, "kaggle_trainset.csv")
TEST_PATH   = os.path.join(REPO_DIR, "kaggle_testset.csv")
SUBMIT_PATH = os.path.join(REPO_DIR, "kaggle_testset_submission.csv")

_DRIVE_DIR = "/content/drive/MyDrive/kaggle_biomedbert_fgm_v2"
_LOCAL_DIR = "outputs/colab_biomedbert_fgm_v2"
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
EPOCHS         = 10
LR             = 8e-6
WARMUP_RATIO   = 0.20
LABEL_SMOOTH   = 0.1
VAL_RATIO      = 0.2
PATIENCE       = 3
FGM_EPSILON    = 1.0
CLASS5_BOOST   = 2.5
CLASS5_INF_MUL = 1.9
SEED           = 42

print(f"BATCH_SIZE={BATCH_SIZE}  GRAD_ACCUM={GRAD_ACCUM}  MAX_LEN={MAX_LEN}")
print(f"CLASS5_BOOST={CLASS5_BOOST}  CLASS5_INF_MUL={CLASS5_INF_MUL}")

LABEL_STR_LIST = [
    "neoplasms",
    "digestive system diseases",
    "nervous system diseases",
    "cardiovascular diseases",
    "general pathological conditions",
]
STR_TO_IDX    = {s: i for i, s in enumerate(LABEL_STR_LIST)}
IDX_TO_SUBMIT = {i: i + 1 for i in range(5)}


class MedicalDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer

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
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
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


torch.manual_seed(SEED); np.random.seed(SEED)

# ── 原始資料（不去重）──────────────────────────────────────────────
print("\n載入原始資料（不做多數投票去重）...")
train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)
print(f"訓練集: {len(train_df)} 筆  測試集: {len(test_df)} 筆")
print(f"標籤分布:\n{train_df['label'].value_counts()}")

all_texts  = train_df["condition"].tolist()
all_labels = [STR_TO_IDX[lbl] for lbl in train_df["label"]]
test_texts = test_df["condition"].tolist()

print("\n載入 tokenizer ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

test_ds     = MedicalDataset(test_texts, None, tokenizer)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)


def make_class_weights(labels):
    cw = compute_class_weight("balanced", classes=np.arange(5), y=labels)
    cw[4] *= CLASS5_BOOST
    print(f"Class weights: {np.round(cw, 3)}")
    return torch.tensor(cw, dtype=torch.float).to(DEVICE)


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


def train_phase(phase_name, tr_texts, tr_labels, vl_texts, vl_labels, fixed_epochs=None):
    print(f"\n{'='*60}\n  {phase_name}\n{'='*60}")

    tr_ds = MedicalDataset(tr_texts, tr_labels, tokenizer)
    tr_ld = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)

    has_val = len(vl_texts) > 0
    if has_val:
        vl_ds = MedicalDataset(vl_texts, vl_labels, tokenizer)
        vl_ld = DataLoader(vl_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    n_epochs     = fixed_epochs if fixed_epochs else EPOCHS
    base_model, model = build_model()
    class_weights     = make_class_weights(tr_labels)
    loss_fn           = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTH)
    fgm               = FGM(base_model, epsilon=FGM_EPSILON)

    total_steps  = (len(tr_ld) // GRAD_ACCUM) * n_epochs
    warmup_steps = int(total_steps * WARMUP_RATIO)
    optimizer    = AdamW(base_model.parameters(), lr=LR, weight_decay=0.01)
    scheduler    = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    scaler = GradScaler(enabled=USE_AMP)
    print(f"Total steps={total_steps}  Warmup={warmup_steps}  EffBatch={BATCH_SIZE*GRAD_ACCUM*max(N_GPUS,1)}")

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
            ids  = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            lbls = batch["labels"].to(DEVICE)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=USE_BF16):
                logits = model(input_ids=ids, attention_mask=mask).logits
            loss = loss_fn(logits.float(), lbls) / GRAD_ACCUM

            if USE_AMP: scaler.scale(loss).backward()
            else:       loss.backward()

            fgm.attack()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=USE_BF16):
                logits_adv = model(input_ids=ids, attention_mask=mask).logits
            loss_adv = loss_fn(logits_adv.float(), lbls) / GRAD_ACCUM
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
            val_pred  = val_probs.argmax(axis=1).tolist()
            val_f1    = f1_score(vl_labels, val_pred, average="macro")
            improved  = val_f1 > best_f1
            status    = "✓ 新最佳" if improved else f"no_improve={no_improve+1}/{PATIENCE}"
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
        val_probs  = get_probs(model, vl_ld, desc="Val Final")
        val_pred   = val_probs.argmax(axis=1).tolist()
        label_names = [f"{i+1}:{LABEL_STR_LIST[i][:14]}" for i in range(5)]
        print(classification_report(vl_labels, val_pred, target_names=label_names))
    else:
        base_model.load_state_dict(torch.load(best_path, map_location=DEVICE))
        best_epoch = fixed_epochs

    return base_model, model, best_epoch


# ── Phase 1：val split 找最佳 epoch ──────────────────────────────────
tr_texts, vl_texts, tr_labels, vl_labels = train_test_split(
    all_texts, all_labels, test_size=VAL_RATIO, stratify=all_labels, random_state=SEED
)
print(f"\nPhase 1  Train: {len(tr_texts)}  Val: {len(vl_texts)}")
_, _, best_epoch = train_phase("Phase1", tr_texts, tr_labels, vl_texts, vl_labels)
print(f"\n最佳 epoch = {best_epoch}，Phase 2 全量訓練 {best_epoch} epoch")

# ── Phase 2：全量訓練（固定最佳 epoch）────────────────────────────────
_, full_model, _ = train_phase("Phase2", all_texts, all_labels, [], [], fixed_epochs=best_epoch)

# ── 推論 ─────────────────────────────────────────────────────────────
print("\n推論測試集 ...")
test_probs     = get_probs(full_model, test_loader, desc="Test")
test_probs_adj = test_probs.copy()
test_probs_adj[:, 4] *= CLASS5_INF_MUL
test_pred_submit = [IDX_TO_SUBMIT[i] for i in test_probs_adj.argmax(axis=1)]

submission = pd.read_csv(SUBMIT_PATH)
submission["label"] = test_pred_submit
sub_path = os.path.join(OUTPUT_DIR, "submission_biomedbert_fgm_v2.csv")
submission.to_csv(sub_path, index=False)

np.save(os.path.join(OUTPUT_DIR, "test_probs_biomedbert_fgm_v2.npy"), test_probs)

print(f"\n提交檔 → {sub_path}")
print(f"預測分布:\n{pd.Series(test_pred_submit).value_counts().sort_index()}")
print(f"\n完成！")
