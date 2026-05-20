"""
BiomedBERT-large + FGM + Class5 Boost（Kaggle Notebook 版）
============================================================
基於最高分腳本的核心設計，加入：
  1. FGM 對抗訓練（提升泛化）
  2. Class 5 loss weight × 2.5x（解決 recall=0.35 問題）
  3. 推論時 class 5 機率 × 1.9x（OOF 校準）
  4. 原始資料（不做多數投票去重，保持與測試集相同分布）
  5. MAX_LEN = 512（完整上下文）
  6. Phase 2：在最佳 epoch 確認後，全量重訓提交

使用方式（Kaggle Notebook）：
  將此檔案內容貼入 Kaggle Notebook 的 Code cell 並執行
"""

import os, time, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_cosine_schedule_with_warmup,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from tqdm.notebook import tqdm

warnings.filterwarnings("ignore")

DATA_DIR    = "/kaggle/input/competitions/1142-medical-condition-classification"
TRAIN_PATH  = os.path.join(DATA_DIR, "kaggle_trainset.csv")
TEST_PATH   = os.path.join(DATA_DIR, "kaggle_testset.csv")
SUBMIT_PATH = os.path.join(DATA_DIR, "kaggle_testset_submission.csv")
OUTPUT_DIR  = "/kaggle/working"

DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_GPUS  = torch.cuda.device_count()
USE_AMP = torch.cuda.is_available()
print(f"Device: {DEVICE}  |  GPUs: {N_GPUS}  |  AMP: {USE_AMP}")
for i in range(N_GPUS):
    mem = torch.cuda.get_device_properties(i).total_memory / 1e9
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}  ({mem:.1f} GB)")

MODEL_NAME     = "microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract"
MAX_LEN        = 512
BATCH_SIZE     = 8
GRAD_ACCUM     = 8
EPOCHS         = 10
LR             = 8e-6
WARMUP_RATIO   = 0.20
LABEL_SMOOTH   = 0.1
VAL_RATIO      = 0.2
PATIENCE       = 3
FGM_EPSILON    = 1.0
CLASS5_BOOST   = 2.5   # class 5 loss weight 額外乘以 2.5x
CLASS5_INF_MUL = 1.9   # 推論時 class 5 機率乘以 1.9x
SEED           = 42

print(f"MAX_LEN={MAX_LEN}  FGM_EPSILON={FGM_EPSILON}  CLASS5_BOOST={CLASS5_BOOST}  CLASS5_INF_MUL={CLASS5_INF_MUL}")

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

# ── 載入原始資料（不做多數投票去重）────────────────────────────────
train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)
print(f"\n原始訓練集: {len(train_df)} 筆（含重複/衝突，保持與測試集相同分布）")
print(f"測試集: {len(test_df)} 筆")
print(f"標籤分布:\n{train_df['label'].value_counts()}")

all_texts  = train_df["condition"].tolist()
all_labels = [STR_TO_IDX[lbl] for lbl in train_df["label"]]
test_texts = test_df["condition"].tolist()

# ── Phase 1：train/val split（確認最佳 epoch）────────────────────────
train_texts, val_texts, train_labels, val_labels = train_test_split(
    all_texts, all_labels,
    test_size=VAL_RATIO, stratify=all_labels, random_state=SEED,
)
print(f"\nPhase 1  Train: {len(train_texts)}  Val: {len(val_texts)}")

print("\n載入 tokenizer ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

test_ds     = MedicalDataset(test_texts, None, tokenizer)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)


def make_loaders(tr_texts, tr_labels, vl_texts, vl_labels):
    tr_ds = MedicalDataset(tr_texts, tr_labels, tokenizer)
    vl_ds = MedicalDataset(vl_texts, vl_labels, tokenizer)
    tr_ld = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    vl_ld = DataLoader(vl_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    return tr_ld, vl_ld


def make_class_weights(labels):
    cw = compute_class_weight("balanced", classes=np.arange(5), y=labels)
    cw[4] *= CLASS5_BOOST
    print(f"Class weights (balanced × boost): {np.round(cw, 3)}")
    return torch.tensor(cw, dtype=torch.float).to(DEVICE)


def build_model():
    base = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=5)
    base.gradient_checkpointing_enable()
    m = nn.DataParallel(base) if N_GPUS > 1 else base
    m = m.to(DEVICE)
    return base, m


@torch.no_grad()
def get_probs(model, loader):
    model.eval()
    all_probs = []
    for batch in loader:
        ids  = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        with autocast(enabled=USE_AMP):
            logits = model(input_ids=ids, attention_mask=mask).logits
        probs = torch.softmax(logits.float(), dim=-1)
        all_probs.append(probs.cpu().numpy())
    return np.vstack(all_probs)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    preds_all, labels_all = [], []
    for batch in loader:
        ids  = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        lbls = batch["labels"]
        with autocast(enabled=USE_AMP):
            logits = model(input_ids=ids, attention_mask=mask).logits
        preds_all.extend(logits.argmax(dim=-1).cpu().numpy())
        labels_all.extend(lbls.numpy())
    return f1_score(labels_all, preds_all, average="macro"), labels_all, preds_all


def train_phase(phase_name, tr_loader, vl_loader, tr_labels, fixed_epochs=None):
    print(f"\n{'='*60}\n  {phase_name}\n{'='*60}")
    base_model, model = build_model()
    class_weights = make_class_weights(tr_labels)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTH)
    fgm     = FGM(base_model, emb_name="embeddings.word_embeddings", epsilon=FGM_EPSILON)

    n_epochs     = fixed_epochs if fixed_epochs else EPOCHS
    total_steps  = (len(tr_loader) // GRAD_ACCUM) * n_epochs
    warmup_steps = int(total_steps * WARMUP_RATIO)
    optimizer    = AdamW(base_model.parameters(), lr=LR, weight_decay=0.01)
    scheduler    = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    scaler = GradScaler(enabled=USE_AMP)

    print(f"Total steps: {total_steps}  Warmup: {warmup_steps}")
    print(f"Effective batch: {BATCH_SIZE * GRAD_ACCUM * max(N_GPUS,1)}")

    best_f1    = 0.0
    no_improve = 0
    best_epoch = 1
    best_path  = os.path.join(OUTPUT_DIR, f"best_{phase_name.replace(' ','_')}.pt")

    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()
        pbar = tqdm(tr_loader, desc=f"Epoch {epoch}/{n_epochs}", leave=True)

        for step, batch in enumerate(pbar):
            ids  = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            lbls = batch["labels"].to(DEVICE)

            with autocast(enabled=USE_AMP):
                logits = model(input_ids=ids, attention_mask=mask).logits
            loss = loss_fn(logits.float(), lbls) / GRAD_ACCUM
            scaler.scale(loss).backward()

            # FGM 對抗訓練
            fgm.attack()
            with autocast(enabled=USE_AMP):
                logits_adv = model(input_ids=ids, attention_mask=mask).logits
            loss_adv = loss_fn(logits_adv.float(), lbls) / GRAD_ACCUM
            scaler.scale(loss_adv).backward()
            fgm.restore()

            total_loss += loss.item() * GRAD_ACCUM

            if (step + 1) % GRAD_ACCUM == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(base_model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
            pbar.set_postfix({"loss": f"{total_loss/(step+1):.4f}"})

        if vl_loader is not None:
            val_f1, val_true, val_pred = evaluate(model, vl_loader)
            improved = val_f1 > best_f1
            status   = "✓ 新最佳" if improved else f"no_improve={no_improve+1}/{PATIENCE}"
            print(f"Epoch {epoch}/{n_epochs}  loss={total_loss/len(tr_loader):.4f}  val_F1={val_f1:.4f}  {status}")
            if improved:
                best_f1    = val_f1
                best_epoch = epoch
                no_improve = 0
                torch.save(base_model.state_dict(), best_path)
            else:
                no_improve += 1
                if no_improve >= PATIENCE and not fixed_epochs:
                    print("  [EarlyStop] 停止訓練")
                    break
        else:
            print(f"Epoch {epoch}/{n_epochs}  loss={total_loss/len(tr_loader):.4f}")
            torch.save(base_model.state_dict(), best_path)

    if vl_loader is not None:
        print(f"\n載入最佳模型 (epoch={best_epoch}, Val F1={best_f1:.4f}) ...")
        base_model.load_state_dict(torch.load(best_path, map_location=DEVICE))
        val_f1, val_true, val_pred = evaluate(model, vl_loader)
        label_names = [f"{i+1}:{LABEL_STR_LIST[i][:14]}" for i in range(5)]
        print(classification_report(val_true, val_pred, target_names=label_names))

    return base_model, model, best_epoch


# ── Phase 1：val split 確認最佳 epoch ────────────────────────────────
tr_loader, vl_loader = make_loaders(train_texts, train_labels, val_texts, val_labels)
_, _, best_epoch = train_phase("Phase1 val-split", tr_loader, vl_loader, train_labels)
print(f"\n最佳 epoch = {best_epoch}，Phase 2 將全量訓練 {best_epoch} epoch")

# ── Phase 2：全量訓練（固定最佳 epoch）────────────────────────────────
full_loader, _ = make_loaders(all_texts, all_labels, [], [])
# 全量沒有 val_loader，固定 epoch 數
full_ds = MedicalDataset(all_texts, all_labels, tokenizer)
full_loader = DataLoader(full_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
_, full_model, _ = train_phase("Phase2 full-data", full_loader, None, all_labels, fixed_epochs=best_epoch)

# ── 推論（class 5 × 1.9x）────────────────────────────────────────────
print("\n推論測試集 ...")
test_probs = get_probs(full_model, test_loader)

test_probs_adj = test_probs.copy()
test_probs_adj[:, 4] *= CLASS5_INF_MUL

test_pred_submit = [IDX_TO_SUBMIT[i] for i in test_probs_adj.argmax(axis=1)]

submission = pd.read_csv(SUBMIT_PATH)
submission["label"] = test_pred_submit
submit_out = os.path.join(OUTPUT_DIR, "submission_biomedbert_fgm_v2.csv")
submission.to_csv(submit_out, index=False)

np.save(os.path.join(OUTPUT_DIR, "test_probs_biomedbert_fgm_v2.npy"), test_probs)

print(f"\n提交檔 → {submit_out}")
print(f"預測分布:\n{pd.Series(test_pred_submit).value_counts().sort_index()}")
print(f"\n完成！")
