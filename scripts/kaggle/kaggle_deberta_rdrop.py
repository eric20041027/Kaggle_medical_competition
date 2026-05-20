"""
DeBERTa-v3-large + R-Drop + 雜訊清理（Kaggle 版）
===================================================
改良重點：
  1. DeBERTa-v3-large：比 BiomedBERT 更強的特徵萃取
  2. R-Drop：每 step 兩次前向傳播 + KL 正則化，專為雜訊標籤設計
  3. 結構標籤清理：移除 RESULTS--、OBJECTIVE: 等論文排版噪音
  4. 原始資料 12,994 筆（不去重）、Val-split 20%

【Kaggle 執行方式】
  - New Notebook → 貼上本腳本 → Run All
  - 需要開啟 Internet（下載 DeBERTa-v3-large 模型）
  - GPU: T4 x1 或 T4 x2 皆可（自動偵測）
  - 預計時間：T4 x1 約 3 小時，T4 x2 約 1.5 小時

【下載結果】
  /kaggle/working/submission_deberta_rdrop.csv
  /kaggle/working/test_probs_deberta_rdrop.npy
"""

import os, re, warnings, glob
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

# ── 路徑設定 ──────────────────────────────────────────────
def find_data_dir():
    candidates = glob.glob("/kaggle/input/**/kaggle_trainset.csv", recursive=True)
    if candidates:
        return os.path.dirname(candidates[0])
    if os.path.exists("kaggle_trainset.csv"):
        return "."
    raise FileNotFoundError("找不到 kaggle_trainset.csv，請確認資料集已掛載")

DATA_DIR    = find_data_dir()
TRAIN_PATH  = os.path.join(DATA_DIR, "kaggle_trainset.csv")
TEST_PATH   = os.path.join(DATA_DIR, "kaggle_testset.csv")
SUBMIT_PATH = os.path.join(DATA_DIR, "kaggle_testset_submission.csv")
OUTPUT_DIR  = "/kaggle/working"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"資料目錄: {DATA_DIR}")

# ── 環境偵測 ──────────────────────────────────────────────
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_GPUS   = torch.cuda.device_count()
# DeBERTa-v3-large 的 Disentangled Attention 在 FP16 下會 overflow → loss=nan
# 強制 float32 訓練，犧牲一點速度換穩定性
USE_BF16 = False
USE_AMP  = False
print(f"Device: {DEVICE}  |  GPUs: {N_GPUS}  |  BF16: {USE_BF16}  |  FP16: {USE_AMP} (DeBERTa 強制 float32)")
for i in range(N_GPUS):
    mem = torch.cuda.get_device_properties(i).total_memory / 1e9
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}  ({mem:.1f} GB)")

# ── 超參數 ────────────────────────────────────────────────
MODEL_NAME    = "microsoft/deberta-v3-large"
MAX_LEN       = 384  # 資料平均 163-199 詞，384 token 覆蓋 75%+ 文本，比 512 快 ~30%
BATCH_SIZE    = 4 if N_GPUS <= 1 else 8     # DeBERTa-v3-large 比 BERT 佔更多記憶體
GRAD_ACCUM    = 16 if N_GPUS <= 1 else 8    # effective batch = 64
EPOCHS        = 8
LR            = 6e-6
WARMUP_RATIO  = 0.20
LABEL_SMOOTH  = 0.1
VAL_RATIO     = 0.2
PATIENCE      = 3
RDROP_ALPHA   = 0.5    # KL 損失權重；0 = 純 CE（等同關閉 R-Drop）
SEED          = 42

print(f"\nMODEL: {MODEL_NAME}")
print(f"BATCH={BATCH_SIZE}  GRAD_ACCUM={GRAD_ACCUM}  EffBatch={BATCH_SIZE*GRAD_ACCUM*max(N_GPUS,1)}")
print(f"RDROP_ALPHA={RDROP_ALPHA}  MAX_LEN={MAX_LEN}")

# ── 標籤對應 ──────────────────────────────────────────────
LABEL_STR_LIST = [
    "neoplasms",
    "digestive system diseases",
    "nervous system diseases",
    "cardiovascular diseases",
    "general pathological conditions",
]
STR_TO_IDX    = {s: i for i, s in enumerate(LABEL_STR_LIST)}
IDX_TO_SUBMIT = {i: i + 1 for i in range(5)}

# ── 雜訊清理 ──────────────────────────────────────────────
_NOISE_RE = re.compile(r'\b[A-Z]{2,}(?:-[A-Z]+)*(?:--|:)\s*')

def clean_text(text: str) -> str:
    """移除醫學摘要中的結構標籤，例如 RESULTS-- OBJECTIVE: BACKGROUND--"""
    return _NOISE_RE.sub(' ', text).strip()

# ── Dataset ───────────────────────────────────────────────
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
        item = {k: v.squeeze(0) for k, v in enc.items() if k in ("input_ids", "attention_mask", "token_type_ids")}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ── R-Drop 損失 ───────────────────────────────────────────
class RDropLoss(nn.Module):
    """
    CE loss（含 label smoothing）+ 對稱 KL 散度
    每個 step 需要兩次前向傳播，各自有不同的 dropout mask
    """
    def __init__(self, weight=None, alpha=0.5, label_smoothing=0.1):
        super().__init__()
        self.alpha          = alpha
        self.label_smoothing = label_smoothing
        self.weight         = weight

    def forward(self, logits1, logits2, targets):
        ce1 = F.cross_entropy(logits1, targets, weight=self.weight,
                              label_smoothing=self.label_smoothing)
        ce2 = F.cross_entropy(logits2, targets, weight=self.weight,
                              label_smoothing=self.label_smoothing)
        ce  = (ce1 + ce2) / 2

        if self.alpha == 0:
            return ce

        p1 = F.softmax(logits1.float(), dim=-1)
        p2 = F.softmax(logits2.float(), dim=-1)
        kl = (F.kl_div(p1.log(), p2, reduction="batchmean") +
              F.kl_div(p2.log(), p1, reduction="batchmean")) / 2

        return ce + self.alpha * kl


# ── 推論 ──────────────────────────────────────────────────
@torch.no_grad()
def get_probs(model, loader, desc=""):
    model.eval()
    all_probs = []
    for batch in tqdm(loader, desc=desc, leave=False):
        ids  = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        kwargs = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in batch:
            kwargs["token_type_ids"] = batch["token_type_ids"].to(DEVICE)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=USE_BF16):
            logits = model(**kwargs).logits
        probs = torch.softmax(logits.float(), dim=-1)
        all_probs.append(probs.cpu().numpy())
    return np.vstack(all_probs)


# ── 主流程 ────────────────────────────────────────────────
torch.manual_seed(SEED)
np.random.seed(SEED)

print("\n載入資料（原始 12,994 筆）...")
train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)
print(f"訓練: {len(train_df)}  測試: {len(test_df)}")

# 清理文字
train_df["condition"] = train_df["condition"].apply(clean_text)
test_df["condition"]  = test_df["condition"].apply(clean_text)
print("結構標籤清理完成（RESULTS-- / OBJECTIVE: 等）")

all_texts  = train_df["condition"].tolist()
all_labels = [STR_TO_IDX[l] for l in train_df["label"]]
test_texts = test_df["condition"].tolist()

print("\n載入 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

test_ds = MedicalDataset(test_texts, None, tokenizer)
test_ld = DataLoader(test_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
                     num_workers=2, pin_memory=True)

print("\nVal-split 20%...")
tr_texts, vl_texts, tr_labels, vl_labels = train_test_split(
    all_texts, all_labels, test_size=VAL_RATIO, stratify=all_labels, random_state=SEED
)
print(f"Train: {len(tr_texts)}  Val: {len(vl_texts)}")

tr_ds = MedicalDataset(tr_texts, tr_labels, tokenizer)
tr_ld = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,
                   num_workers=2, pin_memory=True)
vl_ds = MedicalDataset(vl_texts, vl_labels, tokenizer)
vl_ld = DataLoader(vl_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
                   num_workers=2, pin_memory=True)

# 建立模型
print("\n載入模型...")
base = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=5, ignore_mismatched_sizes=True
)
base.gradient_checkpointing_enable()
model = nn.DataParallel(base) if N_GPUS > 1 else base
model = model.to(DEVICE)

# 類別權重
cw = compute_class_weight("balanced", classes=np.arange(5), y=tr_labels)
print(f"Class weights: {np.round(cw, 3)}")
class_weights = torch.tensor(cw, dtype=torch.float).to(DEVICE)

loss_fn = RDropLoss(weight=class_weights, alpha=RDROP_ALPHA,
                    label_smoothing=LABEL_SMOOTH)

total_steps  = (len(tr_ld) // GRAD_ACCUM) * EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)
optimizer    = AdamW(base.parameters(), lr=LR, weight_decay=0.01, eps=1e-6)
scheduler    = get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
)
scaler = GradScaler(enabled=USE_AMP)

print(f"Total steps={total_steps}  Warmup={warmup_steps}")

best_f1    = 0.0
no_improve = 0
best_epoch = 1
best_path  = os.path.join(OUTPUT_DIR, "best_deberta_rdrop.pt")

print(f"\n{'='*60}\n  訓練開始（DeBERTa-v3-large + R-Drop）\n{'='*60}")

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    pbar = tqdm(tr_ld, desc=f"Epoch {epoch}/{EPOCHS}", leave=True)

    for step, batch in enumerate(pbar):
        ids  = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        lbls = batch["labels"].to(DEVICE)
        kwargs = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in batch:
            kwargs["token_type_ids"] = batch["token_type_ids"].to(DEVICE)

        # R-Drop：同一 batch 過兩次（dropout 不同）
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=USE_BF16):
            logits1 = model(**kwargs).logits
            logits2 = model(**kwargs).logits

        loss = loss_fn(logits1.float(), logits2.float(), lbls) / GRAD_ACCUM

        if USE_AMP:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        total_loss += loss.item() * GRAD_ACCUM

        if (step + 1) % GRAD_ACCUM == 0:
            if USE_AMP:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(base.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(base.parameters(), 1.0)
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
        torch.save(base.state_dict(), best_path)
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            print("  [EarlyStop]")
            break

print(f"\n最佳 epoch={best_epoch}  Val F1={best_f1:.4f}")
base.load_state_dict(torch.load(best_path, map_location=DEVICE))

label_names = [f"{i+1}:{LABEL_STR_LIST[i][:14]}" for i in range(5)]
val_final = get_probs(model, vl_ld, desc="Val Final")
print(classification_report(vl_labels, val_final.argmax(axis=1), target_names=label_names))

print("\n推論測試集...")
test_probs = get_probs(model, test_ld, desc="Test")
test_pred  = [IDX_TO_SUBMIT[i] for i in test_probs.argmax(axis=1)]

submission = pd.read_csv(SUBMIT_PATH)
submission["label"] = test_pred
sub_path = os.path.join(OUTPUT_DIR, "submission_deberta_rdrop.csv")
submission.to_csv(sub_path, index=False)
np.save(os.path.join(OUTPUT_DIR, "test_probs_deberta_rdrop.npy"), test_probs)

from collections import Counter
dist = Counter(test_pred)
print(f"\n提交檔 → {sub_path}")
print(f"預測分布: " + "  ".join([f"class{k}={dist[k]}" for k in sorted(dist)]))
print(f"Class5 比例: {dist[5]/len(test_pred)*100:.1f}%")
print("\n完成！DeBERTa-v3-large + R-Drop + 雜訊清理")
