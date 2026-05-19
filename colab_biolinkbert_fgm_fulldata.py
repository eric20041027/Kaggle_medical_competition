"""
BioLinkBERT-large + FGM + 全量訓練 + Class5 強化
=================================================
策略：
  1. 不做 K-Fold，全部 10,395 筆都用於訓練（無驗證集）
  2. Class 5（general pathological conditions）loss weight 額外加倍
  3. 固定 epoch 數（根據 K-Fold 最佳 epoch 決定）
  4. 推論時對 class 5 機率乘以 1.9（OOF 校準的最佳值）

【執行步驟】

  # Cell 1：掛載 Google Drive
  from google.colab import drive
  drive.mount('/content/drive')

  # Cell 2：安裝與複製程式
  import os; os.chdir('/content')
  !git clone https://github.com/eric20041027/Kaggle_medical_competition.git
  os.chdir('/content/Kaggle_medical_competition')
  !pip install -q transformers torch scikit-learn pandas numpy tqdm

  # Cell 3：訓練
  !python colab_biolinkbert_fgm_fulldata.py

  # Cell 4：下載結果
  from google.colab import files
  files.download('/content/drive/MyDrive/kaggle_biolinkbert_fgm_fulldata/test_probs_biolinkbert_fgm_fulldata.npy')
  files.download('/content/drive/MyDrive/kaggle_biolinkbert_fgm_fulldata/submission_fgm_fulldata.csv')
"""

import os, warnings, gc
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
from sklearn.metrics import f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

warnings.filterwarnings("ignore")

REPO_DIR    = "."
TRAIN_PATH  = os.path.join(REPO_DIR, "kaggle_trainset.csv")
TEST_PATH   = os.path.join(REPO_DIR, "kaggle_testset.csv")
SUBMIT_PATH = os.path.join(REPO_DIR, "kaggle_testset_submission.csv")

_DRIVE_DIR = "/content/drive/MyDrive/kaggle_biolinkbert_fgm_fulldata"
_LOCAL_DIR = "outputs/colab_biolinkbert_fgm_fulldata"
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

MODEL_NAME      = "michiyasunaga/BioLinkBERT-large"
MAX_LEN         = 384
_is_a100        = torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory > 30e9
BATCH_SIZE      = 16 if _is_a100 else 8
GRAD_ACCUM      = 4  if _is_a100 else 8
EPOCHS          = 6        # K-Fold 最佳 epoch 約在第 5-6 epoch
LR              = 8e-6
WARMUP_RATIO    = 0.10     # 全量無驗證，warmup 縮短
LABEL_SMOOTH    = 0.1
FGM_EPSILON     = 1.0
CLASS5_BOOST    = 2.0      # class 5 loss weight 額外乘以 2x
CLASS5_MULT_INF = 1.9      # 推論時 class 5 機率乘以 1.9（OOF 校準值）
SEED            = 42

print(f"BATCH_SIZE={BATCH_SIZE}  GRAD_ACCUM={GRAD_ACCUM}  EPOCHS={EPOCHS}  CLASS5_BOOST={CLASS5_BOOST}")

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

# ── 載入資料 ──────────────────────────────────────────────────────────
train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)

label_counts = train_df.groupby(["condition", "label"]).size().reset_index(name="cnt")
majority_idx = label_counts.groupby("condition")["cnt"].idxmax()
df_clean     = label_counts.loc[majority_idx, ["condition", "label"]].reset_index(drop=True)
print(f"全量訓練集: {len(df_clean)} 筆  測試集: {len(test_df)} 筆")
print(f"標籤分布:\n{df_clean['label'].value_counts()}")

all_texts  = df_clean["condition"].tolist()
all_labels = [STR_TO_IDX[lbl] for lbl in df_clean["label"]]
test_texts = test_df["condition"].tolist()

print("\n載入 tokenizer ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_ds    = MedicalDataset(all_texts, all_labels, tokenizer)
test_ds     = MedicalDataset(test_texts, None, tokenizer)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# ── Class weights：balanced + class5 boost ────────────────────────────
cw = compute_class_weight("balanced", classes=np.arange(5), y=all_labels)
cw[4] *= CLASS5_BOOST
print(f"\nClass weights (balanced × boost): {np.round(cw, 3)}")
class_weights = torch.tensor(cw, dtype=torch.float).to(DEVICE)

# ── 建立模型 ──────────────────────────────────────────────────────────
print("\n載入模型 ...")
base_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=5, torch_dtype=torch.float32
)
base_model.gradient_checkpointing_enable()
model = nn.DataParallel(base_model) if N_GPUS > 1 else base_model
model = model.to(DEVICE)

loss_fn = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTH)
fgm     = FGM(base_model, emb_name="embeddings.word_embeddings", epsilon=FGM_EPSILON)

total_steps  = (len(train_loader) // GRAD_ACCUM) * EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)
optimizer    = AdamW(base_model.parameters(), lr=LR, weight_decay=0.01)
scheduler    = get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
)
scaler = GradScaler(enabled=USE_AMP)

print(f"Total steps: {total_steps}  Warmup: {warmup_steps}")

# ── 訓練（全量，無驗證）───────────────────────────────────────────────
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=True)

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

    print(f"Epoch {epoch}/{EPOCHS}  avg_loss={total_loss/len(train_loader):.4f}")

    # 每個 epoch 存一次模型（供事後選擇）
    ckpt_path = os.path.join(OUTPUT_DIR, f"fulldata_epoch{epoch}.pt")
    torch.save(base_model.state_dict(), ckpt_path)
    print(f"  → 存檔 {ckpt_path}")

# ── 推論（最終 epoch）────────────────────────────────────────────────
print("\n推論測試集 ...")
model.eval()
all_probs = []
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Test inference"):
        ids  = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=USE_BF16):
            logits = model(input_ids=ids, attention_mask=mask).logits
        probs = torch.softmax(logits.float(), dim=-1)
        all_probs.append(probs.cpu().numpy())

test_probs = np.vstack(all_probs)

# ── 儲存原始機率（供後續 ensemble）────────────────────────────────────
test_path = os.path.join(OUTPUT_DIR, "test_probs_biolinkbert_fgm_fulldata.npy")
np.save(test_path, test_probs)
print(f"原始機率已存：{test_path}")

# ── 套用 class 5 推論倍率（OOF 校準）────────────────────────────────
test_probs_adj = test_probs.copy()
test_probs_adj[:, 4] *= CLASS5_MULT_INF
test_pred_submit = [IDX_TO_SUBMIT[i] for i in test_probs_adj.argmax(axis=1)]

submission = pd.read_csv(SUBMIT_PATH)
submission["label"] = test_pred_submit
sub_path = os.path.join(OUTPUT_DIR, "submission_fgm_fulldata.csv")
submission.to_csv(sub_path, index=False)

print(f"\n提交檔 → {sub_path}")
print(f"預測分布:\n{pd.Series(test_pred_submit).value_counts().sort_index()}")
print(f"\n完成！全量 BioLinkBERT-FGM + Class5 boost × {CLASS5_BOOST} + 推論倍率 × {CLASS5_MULT_INF}")
