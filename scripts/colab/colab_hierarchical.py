"""
階層式分類 + Focal Loss（Colab 版）
=====================================
架構：
  Stage 1 — 二元分類：class5 (general) vs class1-4 (specific)
  Stage 2 — 四分類：neoplasms / digestive / nervous / cardiovascular

兩個階段均使用：
  - BiomedBERT-large + FGM + Focal Loss (gamma=2.0)
  - 原始資料 (12,994 筆，不去重)
  - Val-split 20%，早停

推論邏輯：
  P(class5) = Stage1 輸出的 class5 機率
  P(class1-4) = (1 - P(class5)) × Stage2 各類機率

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
  !python scripts/colab/colab_hierarchical.py

  # Cell 4
  from google.colab import files
  files.download('/content/drive/MyDrive/kaggle_hierarchical/submission_hierarchical.csv')
  files.download('/content/drive/MyDrive/kaggle_hierarchical/test_probs_hierarchical.npy')
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

_DRIVE_DIR = "/content/drive/MyDrive/kaggle_hierarchical"
_LOCAL_DIR = "outputs/colab_hierarchical"
OUTPUT_DIR = _DRIVE_DIR if os.path.exists("/content/drive/MyDrive") else _LOCAL_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(_LOCAL_DIR, exist_ok=True)

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_GPUS   = torch.cuda.device_count()
USE_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
USE_AMP  = torch.cuda.is_available() and not USE_BF16
print(f"Device: {DEVICE}  |  GPUs: {N_GPUS}  |  BF16: {USE_BF16}  |  AMP: {USE_AMP}")
if torch.cuda.is_available():
    for i in range(N_GPUS):
        mem = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({mem:.1f} GB)")

MODEL_NAME     = "microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract"
MAX_LEN        = 512
_is_a100       = torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory > 30e9
BATCH_SIZE     = 16 if _is_a100 else 8
GRAD_ACCUM     = 4  if _is_a100 else 8
EPOCHS         = 8
LR             = 8e-6
WARMUP_RATIO   = 0.20
LABEL_SMOOTH   = 0.1
VAL_RATIO      = 0.2
PATIENCE       = 2
FGM_EPSILON    = 1.0
FOCAL_GAMMA    = 2.0
SEED           = 42

print(f"MODEL: {MODEL_NAME}")
print(f"BATCH={BATCH_SIZE}  ACCUM={GRAD_ACCUM}  MAX_LEN={MAX_LEN}  FOCAL_GAMMA={FOCAL_GAMMA}")

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


class FocalLoss(nn.Module):
    """Focal Loss：聚焦難以分類的樣本（如 class5 邊界案例）"""
    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.weight          = weight
        self.gamma           = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce = F.cross_entropy(
            logits, targets,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


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

# ── 載入資料 ──────────────────────────────────────────────────────
print("\n載入原始資料...")
train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)
print(f"訓練集: {len(train_df)} 筆  測試集: {len(test_df)} 筆")

all_texts  = train_df["condition"].tolist()
all_labels = [STR_TO_IDX[lbl] for lbl in train_df["label"]]  # 0-4
test_texts = test_df["condition"].tolist()

# Stage 1 標籤：0=specific(class1-4), 1=general(class5)
s1_labels = [1 if lbl == 4 else 0 for lbl in all_labels]
# Stage 2 標籤：只用 class1-4 樣本（label 0-3）
s2_mask   = [lbl != 4 for lbl in all_labels]
s2_texts  = [t for t, m in zip(all_texts, s2_mask) if m]
s2_labels = [lbl for lbl, m in zip(all_labels, s2_mask) if m]

print(f"Stage1 資料: {len(all_texts)} 筆 (class5={sum(s1_labels)}, specific={len(s1_labels)-sum(s1_labels)})")
print(f"Stage2 資料: {len(s2_texts)} 筆（僅 class1-4）")

print("\n載入 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

test_ds     = MedicalDataset(test_texts, None, tokenizer)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# ── Val Split ─────────────────────────────────────────────────────
# 用全量資料切 val（保持原始 5 類分層）
tr_idx, vl_idx = train_test_split(
    range(len(all_texts)), test_size=VAL_RATIO,
    stratify=all_labels, random_state=SEED
)
tr_idx, vl_idx = list(tr_idx), list(vl_idx)

# Stage 1
s1_tr_texts  = [all_texts[i] for i in tr_idx]
s1_tr_labels = [s1_labels[i] for i in tr_idx]
s1_vl_texts  = [all_texts[i] for i in vl_idx]
s1_vl_labels = [s1_labels[i] for i in vl_idx]
vl_true_5cls = [all_labels[i] for i in vl_idx]  # 用於最終 5 類 F1

# Stage 2（只取 specific 樣本）
s2_tr_texts  = [all_texts[i] for i in tr_idx if all_labels[i] != 4]
s2_tr_labels = [all_labels[i] for i in tr_idx if all_labels[i] != 4]
s2_vl_texts  = [all_texts[i] for i in vl_idx if all_labels[i] != 4]
s2_vl_labels = [all_labels[i] for i in vl_idx if all_labels[i] != 4]
vl_is_specific = [all_labels[i] != 4 for i in vl_idx]

print(f"\nStage1 Train={len(s1_tr_texts)}  Val={len(s1_vl_texts)}")
print(f"Stage2 Train={len(s2_tr_texts)}  Val={len(s2_vl_texts)}")


def make_class_weights(labels, n_classes):
    cw = compute_class_weight("balanced", classes=np.arange(n_classes), y=labels)
    print(f"  Class weights ({n_classes}): {np.round(cw, 3)}")
    return torch.tensor(cw, dtype=torch.float).to(DEVICE)


def build_model(n_labels):
    base = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=n_labels, torch_dtype=torch.float32
    )
    base.gradient_checkpointing_enable()
    m = nn.DataParallel(base) if N_GPUS > 1 else base
    return base, m.to(DEVICE)


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


def train_stage(stage_name, tr_texts, tr_labels, vl_texts, vl_labels,
                n_labels, save_name):
    print(f"\n{'='*60}")
    print(f"  {stage_name}  ({n_labels} 類)")
    print(f"{'='*60}")

    tr_ds = MedicalDataset(tr_texts, tr_labels, tokenizer)
    tr_ld = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    vl_ds = MedicalDataset(vl_texts, vl_labels, tokenizer)
    vl_ld = DataLoader(vl_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    base_model, model = build_model(n_labels)
    cw      = make_class_weights(tr_labels, n_labels)
    loss_fn = FocalLoss(weight=cw, gamma=FOCAL_GAMMA, label_smoothing=LABEL_SMOOTH)
    fgm     = FGM(base_model, epsilon=FGM_EPSILON)

    total_steps  = (len(tr_ld) // GRAD_ACCUM) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    optimizer    = AdamW(base_model.parameters(), lr=LR, weight_decay=0.01)
    scheduler    = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    scaler    = GradScaler(enabled=USE_AMP)
    best_f1   = 0.0
    no_impr   = 0
    best_path = os.path.join(OUTPUT_DIR, f"best_{save_name}.pt")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()
        pbar = tqdm(tr_ld, desc=f"Epoch {epoch}/{EPOCHS}", leave=True)

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

        vl_probs = get_probs(model, vl_ld, desc="Val")
        vl_pred  = vl_probs.argmax(axis=1).tolist()
        avg      = "binary" if n_labels == 2 else "macro"
        val_f1   = f1_score(vl_labels, vl_pred, average=avg)
        status   = "✓ 新最佳" if val_f1 > best_f1 else f"no_improve={no_impr+1}/{PATIENCE}"
        print(f"Epoch {epoch}/{EPOCHS}  loss={total_loss/len(tr_ld):.4f}  val_F1={val_f1:.4f}  {status}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            no_impr = 0
            torch.save(base_model.state_dict(), best_path)
        else:
            no_impr += 1
            if no_impr >= PATIENCE:
                print("  [EarlyStop]")
                break

    print(f"\n最佳 val_F1={best_f1:.4f}")
    base_model.load_state_dict(torch.load(best_path, map_location=DEVICE))

    # 回傳 val probs 和 test probs
    vl_probs_final = get_probs(model, vl_ld, desc="Val Final")
    test_probs     = get_probs(model, test_loader, desc="Test")
    return base_model, model, vl_probs_final, test_probs


# ── Stage 1：二元分類 ──────────────────────────────────────────────
s1_base, s1_model, s1_vl_probs, s1_test_probs = train_stage(
    "Stage 1：class5 vs specific",
    s1_tr_texts, s1_tr_labels,
    s1_vl_texts, s1_vl_labels,
    n_labels=2, save_name="s1"
)
# s1_probs[:, 1] = P(class5)
s1_vl_p5   = s1_vl_probs[:, 1]   # (n_val,)
s1_test_p5 = s1_test_probs[:, 1]  # (n_test,)

# ── Stage 2：4 分類（class1-4）────────────────────────────────────
s2_base, s2_model, s2_vl_probs, s2_test_probs = train_stage(
    "Stage 2：neoplasms / digestive / nervous / cardiovascular",
    s2_tr_texts, s2_tr_labels,
    s2_vl_texts, s2_vl_labels,
    n_labels=4, save_name="s2"
)
# s2_probs shape: (n_specific_val, 4)

# ── 組合 Val 評估 ─────────────────────────────────────────────────
print("\n\n=== 組合階層式推論 Val 評估 ===")
# 對所有 val 樣本，先取 Stage 2 的機率（針對 specific val 樣本計算）
s2_full_vl_probs = np.zeros((len(vl_idx), 4))
spec_ptr = 0
for i, is_spec in enumerate(vl_is_specific):
    if is_spec:
        s2_full_vl_probs[i] = s2_vl_probs[spec_ptr]
        spec_ptr += 1

# 組合 5 類機率
combined_vl = np.zeros((len(vl_idx), 5))
combined_vl[:, 4]   = s1_vl_p5                              # P(class5)
combined_vl[:, 0:4] = (1 - s1_vl_p5[:, None]) * s2_full_vl_probs  # P(class1-4)

vl_pred_5cls = combined_vl.argmax(axis=1).tolist()
val_f1_5cls  = f1_score(vl_true_5cls, vl_pred_5cls, average="macro")

label_names = [f"{i+1}:{LABEL_STR_LIST[i][:14]}" for i in range(5)]
print(f"\n組合 Val Macro F1 = {val_f1_5cls:.4f}")
print(classification_report(vl_true_5cls, vl_pred_5cls, target_names=label_names))

# ── 組合 Test 推論 ────────────────────────────────────────────────
print("\n推論測試集...")
combined_test = np.zeros((len(test_texts), 5))
combined_test[:, 4]   = s1_test_p5
combined_test[:, 0:4] = (1 - s1_test_p5[:, None]) * s2_test_probs

test_pred_submit = [IDX_TO_SUBMIT[i] for i in combined_test.argmax(axis=1)]

submission = pd.read_csv(SUBMIT_PATH)
submission["label"] = test_pred_submit
sub_path = os.path.join(OUTPUT_DIR, "submission_hierarchical.csv")
submission.to_csv(sub_path, index=False)
np.save(os.path.join(OUTPUT_DIR, "test_probs_hierarchical.npy"), combined_test)

print(f"\n提交檔 → {sub_path}")
dist = pd.Series(test_pred_submit).value_counts().sort_index()
print(f"預測分布:\n{dist}")
c5_pct = (dist.get(5, 0) / len(test_pred_submit)) * 100
print(f"Class5 比例: {c5_pct:.1f}%")
print("\n完成！階層式分類 + Focal Loss + FGM")
