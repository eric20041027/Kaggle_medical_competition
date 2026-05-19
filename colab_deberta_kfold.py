"""
DeBERTa-v3-large K-Fold Fine-tuning（Google Colab 版）
=====================================================
【第一次使用】在 Colab 執行以下 cells：

  # Cell 1：掛載 Google Drive（防止斷線後資料消失）
  from google.colab import drive
  drive.mount('/content/drive')

  # Cell 2：安裝與複製程式
  !git clone https://github.com/eric20041027/Kaggle_medical_competition.git
  %cd Kaggle_medical_competition
  !pip install -q transformers torch scikit-learn pandas numpy tqdm sentencepiece protobuf

  # Cell 3：開始訓練
  !python colab_deberta_kfold.py

  # Cell 4：訓練完成後下載（或直接從 Drive 取檔）
  from google.colab import files
  files.download('outputs/colab_deberta/test_probs_deberta.npy')
  files.download('outputs/colab_deberta/oof_probs_deberta.npy')

【斷線後重跑】已完成的 fold checkpoint 存在 Drive，重連後執行：
  from google.colab import drive
  drive.mount('/content/drive')
  !git clone https://github.com/eric20041027/Kaggle_medical_competition.git
  %cd Kaggle_medical_competition
  !pip install -q transformers torch scikit-learn pandas numpy tqdm sentencepiece protobuf
  !python colab_deberta_kfold.py   # 自動跳過已完成的 fold

注意：DeBERTa-v3-large 在 Colab 免費 T4 (16GB) 上可運行，
      若出現 CUDA OOM 可將 BATCH_SIZE 改為 2（GRAD_ACCUM 改為 32）。
"""

import os, warnings, gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.cuda.amp import GradScaler  # kept for reference, not used in BF16/FP32 path
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_cosine_schedule_with_warmup,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── Colab 路徑 ────────────────────────────────────────────────────
# 輸出同步到 Google Drive，防止斷線後資料遺失
REPO_DIR    = "."
TRAIN_PATH  = os.path.join(REPO_DIR, "kaggle_trainset.csv")
TEST_PATH   = os.path.join(REPO_DIR, "kaggle_testset.csv")
SUBMIT_PATH = os.path.join(REPO_DIR, "kaggle_testset_submission.csv")

_DRIVE_DIR = "/content/drive/MyDrive/kaggle_deberta"
_LOCAL_DIR = "outputs/colab_deberta"
# Drive 已掛載就存到 Drive，否則只存本地
OUTPUT_DIR = _DRIVE_DIR if os.path.exists("/content/drive/MyDrive") else _LOCAL_DIR

DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_GPUS  = torch.cuda.device_count()
# BF16：A100/V100 原生支援，不需要 GradScaler，避開 DeBERTa FP16 梯度問題
USE_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
print(f"Device: {DEVICE}  |  GPUs: {N_GPUS}  |  BF16: {USE_BF16}")
if torch.cuda.is_available():
    for i in range(N_GPUS):
        mem = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}  ({mem:.1f} GB)")

# ── A100 (40GB) 最佳化；T4 fallback 維持 BATCH_SIZE=4 ──────────
MODEL_NAME   = "microsoft/deberta-v3-large"
MAX_LEN      = 384
_is_a100     = torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory > 30e9
BATCH_SIZE   = 8  if _is_a100 else 4
GRAD_ACCUM   = 8  if _is_a100 else 16   # effective batch = 64
N_FOLDS      = 3  if _is_a100 else 2    # A100 夠快跑 3 fold
EPOCHS       = 8
LR           = 5e-6
WARMUP_RATIO = 0.10
LABEL_SMOOTH = 0.1
PATIENCE     = 3
SEED         = 42

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


os.makedirs(OUTPUT_DIR, exist_ok=True)
torch.manual_seed(SEED); np.random.seed(SEED)

train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)
print(f"原始訓練集: {len(train_df)} 筆 ({train_df['condition'].nunique()} 唯一文本)")

# ── 衝突標籤多數投票 ─────────────────────────────────────────────
label_counts = (
    train_df.groupby(["condition", "label"])
    .size()
    .reset_index(name="cnt")
)
majority_idx = label_counts.groupby("condition")["cnt"].idxmax()
df_clean = (
    label_counts.loc[majority_idx, ["condition", "label"]]
    .reset_index(drop=True)
)
print(f"清洗後訓練集: {len(df_clean)} 筆（多數投票保留衝突文本）")
print(f"\n清洗後標籤分布:\n{df_clean['label'].value_counts()}")
print(f"\n測試集: {len(test_df)} 筆")

all_texts  = df_clean["condition"].tolist()
all_labels = [STR_TO_IDX[lbl] for lbl in df_clean["label"]]
test_texts = test_df["condition"].tolist()

print("\n載入 DeBERTa tokenizer ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

test_ds     = MedicalDataset(test_texts, None, tokenizer)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# 斷點續訓：若 Drive 上已有部分結果，直接載入
_oof_path  = os.path.join(OUTPUT_DIR, "oof_probs_deberta.npy")
_test_path = os.path.join(OUTPUT_DIR, "test_probs_deberta.npy")
_done_path = os.path.join(OUTPUT_DIR, "completed_folds.txt")

if os.path.exists(_oof_path) and os.path.exists(_test_path):
    oof_probs  = np.load(_oof_path)
    test_probs = np.load(_test_path)
    print(f"  ✓ 載入上次中斷的部分結果")
else:
    oof_probs  = np.zeros((len(all_texts), 5))
    test_probs = np.zeros((len(test_texts), 5))

completed_folds = set()
if os.path.exists(_done_path):
    with open(_done_path) as f:
        completed_folds = {int(l.strip()) for l in f if l.strip()}
    print(f"  ✓ 已完成 fold: {sorted(completed_folds)}，跳過重跑")

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)


def build_model():
    base = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=5, torch_dtype=torch.float32
    )
    # gradient_checkpointing 與 AMP 在 DeBERTa 上會產生 FP16 梯度衝突，不啟用
    # BATCH_SIZE=4 + FP32 在 T4 16GB 上已足夠省記憶體
    if N_GPUS > 1:
        m = nn.DataParallel(base)
    else:
        m = base
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
        probs = torch.softmax(logits, dim=-1)
        all_probs.append(probs.cpu().numpy())
    return np.vstack(all_probs)


for fold, (train_idx, val_idx) in enumerate(skf.split(all_texts, all_labels)):
    if fold in completed_folds:
        print(f"\n  [SKIP] Fold {fold+1} 已完成，跳過")
        continue

    print(f"\n{'='*60}")
    print(f"  FOLD {fold+1}/{N_FOLDS}  |  Train: {len(train_idx)}  Val: {len(val_idx)}")
    print(f"{'='*60}")

    fold_train_texts  = [all_texts[i]  for i in train_idx]
    fold_train_labels = [all_labels[i] for i in train_idx]
    fold_val_texts    = [all_texts[i]  for i in val_idx]
    fold_val_labels   = [all_labels[i] for i in val_idx]

    train_ds = MedicalDataset(fold_train_texts, fold_train_labels, tokenizer)
    val_ds   = MedicalDataset(fold_val_texts,   fold_val_labels,   tokenizer)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    base_model, model = build_model()

    cw            = compute_class_weight("balanced", classes=np.arange(5), y=fold_train_labels)
    class_weights = torch.tensor(cw, dtype=torch.float).to(DEVICE)
    loss_fn       = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTH)

    total_steps  = (len(train_loader) // GRAD_ACCUM) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    optimizer    = AdamW(base_model.parameters(), lr=LR, weight_decay=0.01)
    scheduler    = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    # BF16 不需要 GradScaler（動態範圍足夠），FP32 fallback 同樣不需要
    best_f1    = 0.0
    no_improve = 0
    best_path  = os.path.join(OUTPUT_DIR, f"deberta_fold{fold+1}.pt")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Fold{fold+1} Epoch{epoch}/{EPOCHS}", leave=True)

        for step, batch in enumerate(pbar):
            ids  = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            lbls = batch["labels"].to(DEVICE)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=USE_BF16):
                logits = model(input_ids=ids, attention_mask=mask).logits
            loss = loss_fn(logits.float(), lbls) / GRAD_ACCUM  # float32 for stable loss
            loss.backward()
            total_loss += loss.item() * GRAD_ACCUM
            if (step + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(base_model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            pbar.set_postfix({"loss": f"{total_loss / (step + 1):.4f}"})

        val_p = get_probs(model, val_loader, desc="Val")
        val_pred = [IDX_TO_SUBMIT[i] for i in val_p.argmax(axis=1)]
        val_true = [IDX_TO_SUBMIT[l] for l in fold_val_labels]
        val_f1   = f1_score(val_true, val_pred, average="macro")

        improved = val_f1 > best_f1
        status   = "✓ 新最佳" if improved else f"no_improve={no_improve+1}/{PATIENCE}"
        print(f"  Epoch {epoch}/{EPOCHS}  loss={total_loss/len(train_loader):.4f}  val_F1={val_f1:.4f}  {status}")

        if improved:
            best_f1    = val_f1
            no_improve = 0
            torch.save(base_model.state_dict(), best_path)
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print("  [EarlyStop] 停止訓練")
                break

    print(f"\n  Fold {fold+1} 最佳 Val F1 = {best_f1:.4f}，生成預測中 ...")
    base_model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    oof_probs[val_idx]  = get_probs(model, val_loader,  desc=f"OOF  Fold{fold+1}")
    test_probs         += get_probs(model, test_loader, desc=f"Test Fold{fold+1}") / N_FOLDS

    del model, base_model, optimizer, scheduler, scaler, loss_fn
    del train_ds, val_ds, train_loader, val_loader
    torch.cuda.empty_cache()
    gc.collect()

    # 每完成一個 fold 立即存到 Drive，防止斷線損失進度
    np.save(_oof_path,  oof_probs)
    np.save(_test_path, test_probs)
    with open(_done_path, "a") as f:
        f.write(f"{fold}\n")
    completed_folds.add(fold)
    print(f"  Fold {fold+1} 完成，GPU 記憶體已釋放")

# ── 最終結果 ──────────────────────────────────────────────────────
print(f"\n{'='*60}\n  DeBERTa-v3-large K-Fold 訓練完成（Colab）\n{'='*60}")

oof_pred = [IDX_TO_SUBMIT[i] for i in oof_probs.argmax(axis=1)]
oof_true = [IDX_TO_SUBMIT[l] for l in all_labels]
oof_f1   = f1_score(oof_true, oof_pred, average="macro")

label_names = [f"{i+1}:{LABEL_STR_LIST[i][:14]}" for i in range(5)]
print(f"\nOOF Macro F1: {oof_f1:.4f}")
print(classification_report(oof_true, oof_pred, target_names=label_names))

test_pred_submit = [IDX_TO_SUBMIT[i] for i in test_probs.argmax(axis=1)]
submission = pd.read_csv(SUBMIT_PATH)
submission["label"] = test_pred_submit
submission.to_csv(os.path.join(OUTPUT_DIR, "submission_deberta.csv"), index=False)

np.save(os.path.join(OUTPUT_DIR, "oof_probs_deberta.npy"),  oof_probs)
np.save(os.path.join(OUTPUT_DIR, "test_probs_deberta.npy"), test_probs)

print(f"\n提交檔 → {OUTPUT_DIR}/submission_deberta.csv")
print(f"預測分布:\n{pd.Series(test_pred_submit).value_counts().sort_index()}")
print(f"\n完成！OOF Macro F1 = {oof_f1:.4f}")
print(f"\n[後續] 下載 test_probs_deberta.npy 與 oof_probs_deberta.npy 供 ensemble 使用")
