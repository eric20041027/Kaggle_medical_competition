"""
BioLinkBERT-large + FGM + 原始資料（Colab 版）
================================================
重點：
  1. 原始資料（12,994 筆，不去重）
  2. MAX_LEN = 512
  3. FGM 對抗訓練
  4. 無 class5 boost（CLASS5_BOOST=1.0, CLASS5_INF_MUL=1.0）
  5. Val-split 20%，早停後取最佳 checkpoint 直接推論

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
  !python scripts/colab/colab_biolinkbert_raw.py

  # Cell 4
  from google.colab import files
  files.download('/content/drive/MyDrive/kaggle_biolinkbert_raw/submission_biolinkbert_raw.csv')
  files.download('/content/drive/MyDrive/kaggle_biolinkbert_raw/test_probs_biolinkbert_raw.npy')
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
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

REPO_DIR    = "."
TRAIN_PATH  = os.path.join(REPO_DIR, "kaggle_trainset.csv")
TEST_PATH   = os.path.join(REPO_DIR, "kaggle_testset.csv")
SUBMIT_PATH = os.path.join(REPO_DIR, "kaggle_testset_submission.csv")

_DRIVE_DIR = "/content/drive/MyDrive/kaggle_biolinkbert_raw"
_LOCAL_DIR = "outputs/colab_biolinkbert_raw"
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

MODEL_NAME     = "michiyasunaga/BioLinkBERT-large"
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
CLASS5_BOOST   = 1.0
CLASS5_INF_MUL = 1.0
SEED           = 42

print(f"MODEL: {MODEL_NAME}")
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


print("\n訓練集切分（val 20%）...")
tr_texts, vl_texts, tr_labels, vl_labels = train_test_split(
    all_texts, all_labels, test_size=VAL_RATIO, stratify=all_labels, random_state=SEED
)
print(f"Train: {len(tr_texts)}  Val: {len(vl_texts)}")

tr_ds = MedicalDataset(tr_texts, tr_labels, tokenizer)
tr_ld = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
vl_ds = MedicalDataset(vl_texts, vl_labels, tokenizer)
vl_ld = DataLoader(vl_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

base_model, model = build_model()
class_weights     = make_class_weights(tr_labels)
loss_fn           = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTH)
fgm               = FGM(base_model, epsilon=FGM_EPSILON)

total_steps  = (len(tr_ld) // GRAD_ACCUM) * EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)
optimizer    = AdamW(base_model.parameters(), lr=LR, weight_decay=0.01)
scheduler    = get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
)
scaler = GradScaler(enabled=USE_AMP)
print(f"\nTotal steps={total_steps}  Warmup={warmup_steps}  EffBatch={BATCH_SIZE*GRAD_ACCUM*max(N_GPUS,1)}")

best_f1    = 0.0
no_improve = 0
best_epoch = 1
best_path  = os.path.join(OUTPUT_DIR, "best_model.pt")

print(f"\n{'='*60}\n  訓練開始\n{'='*60}")
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

    avg_loss = total_loss / len(tr_ld)

    val_probs = get_probs(model, vl_ld, desc="Val")
    val_pred  = val_probs.argmax(axis=1).tolist()
    val_f1    = f1_score(vl_labels, val_pred, average="macro")
    improved  = val_f1 > best_f1
    status    = "✓ 新最佳" if improved else f"no_improve={no_improve+1}/{PATIENCE}"
    print(f"Epoch {epoch}/{EPOCHS}  loss={avg_loss:.4f}  val_F1={val_f1:.4f}  {status}")

    if improved:
        best_f1    = val_f1
        best_epoch = epoch
        no_improve = 0
        torch.save(base_model.state_dict(), best_path)
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            print("  [EarlyStop]")
            break

print(f"\n最佳 epoch={best_epoch}  Val F1={best_f1:.4f}")
base_model.load_state_dict(torch.load(best_path, map_location=DEVICE))

label_names = [f"{i+1}:{LABEL_STR_LIST[i][:14]}" for i in range(5)]
val_probs_final = get_probs(model, vl_ld, desc="Val Final")
val_pred_final  = val_probs_final.argmax(axis=1).tolist()
print(classification_report(vl_labels, val_pred_final, target_names=label_names))

print("\n推論測試集 ...")
test_probs = get_probs(model, test_loader, desc="Test")
test_probs_adj = test_probs.copy()
test_probs_adj[:, 4] *= CLASS5_INF_MUL
test_pred_submit = [IDX_TO_SUBMIT[i] for i in test_probs_adj.argmax(axis=1)]

submission = pd.read_csv(SUBMIT_PATH)
submission["label"] = test_pred_submit
sub_path = os.path.join(OUTPUT_DIR, "submission_biolinkbert_raw.csv")
submission.to_csv(sub_path, index=False)

np.save(os.path.join(OUTPUT_DIR, "test_probs_biolinkbert_raw.npy"), test_probs)

print(f"\n提交檔 → {sub_path}")
print(f"預測分布:\n{pd.Series(test_pred_submit).value_counts().sort_index()}")
print(f"\n完成！BioLinkBERT-large + FGM + 原始資料")
