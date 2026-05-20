import os, time, warnings, gc
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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

warnings.filterwarnings("ignore")

TRAIN_PATH  = "kaggle_trainset.csv"
TEST_PATH   = "kaggle_testset.csv"
SUBMIT_PATH = "kaggle_testset_submission.csv"
OUTPUT_DIR  = "outputs/local_kfold"

DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()
print(f"Device: {DEVICE}  |  AMP: {USE_AMP}")
if torch.cuda.is_available():
    mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  GPU: {torch.cuda.get_device_name(0)}  ({mem:.1f} GB)")

# RTX 3060（12GB）適用：base 模型 + 較大 batch
MODEL_NAME   = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
MAX_LEN      = 384        # 256→384：與 Kaggle v2 一致，RTX 3060 base 模型 12GB 完全夠用
BATCH_SIZE   = 16
GRAD_ACCUM   = 4          # effective batch = 64
EPOCHS       = 8          # 10→8：收斂點通常在 epoch 7-8
LR           = 1e-5
WARMUP_RATIO = 0.20
LABEL_SMOOTH = 0.1
N_FOLDS      = 3
PATIENCE     = 3          # 2→3：避免過早停止
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
print(f"原始訓練集: {len(train_df)} 筆")

# 資料清洗：衝突標籤改用多數投票（移除會讓 LB 暴跌，詳見 handout）
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
print(f"清洗後（多數投票）: {len(df_clean)} 筆")
print(f"\n清洗後標籤分布:\n{df_clean['label'].value_counts()}")

all_texts  = df_clean["condition"].tolist()
all_labels = [STR_TO_IDX[lbl] for lbl in df_clean["label"]]
test_texts = test_df["condition"].tolist()

print("\n載入 tokenizer ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

test_ds     = MedicalDataset(test_texts, None, tokenizer)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

oof_probs  = np.zeros((len(all_texts), 5))
test_probs = np.zeros((len(test_texts), 5))
fold_best_f1s = []

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

def build_model():
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=5)
    return model.to(DEVICE)

@torch.no_grad()
def get_probs(model, loader, desc=""):
    model.eval()
    all_probs = []
    for batch in tqdm(loader, desc=desc, leave=False):
        ids  = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        with autocast(enabled=USE_AMP):
            logits = model(input_ids=ids, attention_mask=mask).logits
        probs = torch.softmax(logits, dim=-1)
        all_probs.append(probs.cpu().numpy())
    return np.vstack(all_probs)

for fold, (train_idx, val_idx) in enumerate(skf.split(all_texts, all_labels)):
    print(f"\n{'='*60}")
    print(f"  FOLD {fold+1}/{N_FOLDS}  |  Train: {len(train_idx)}  Val: {len(val_idx)}")
    print(f"{'='*60}")

    fold_train_texts  = [all_texts[i]  for i in train_idx]
    fold_train_labels = [all_labels[i] for i in train_idx]
    fold_val_texts    = [all_texts[i]  for i in val_idx]
    fold_val_labels   = [all_labels[i] for i in val_idx]

    train_ds = MedicalDataset(fold_train_texts, fold_train_labels, tokenizer)
    val_ds   = MedicalDataset(fold_val_texts,   fold_val_labels,   tokenizer)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    model = build_model()

    classes       = np.arange(5)
    cw            = compute_class_weight("balanced", classes=classes, y=fold_train_labels)
    class_weights = torch.tensor(cw, dtype=torch.float).to(DEVICE)
    loss_fn       = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTH)

    total_steps  = (len(train_loader) // GRAD_ACCUM) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    optimizer    = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler    = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    scaler    = GradScaler(enabled=USE_AMP)
    best_f1   = 0.0
    no_improve = 0
    best_path = os.path.join(OUTPUT_DIR, f"best_fold{fold+1}.pt")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Fold{fold+1} Epoch{epoch}/{EPOCHS}", leave=True)

        for step, batch in enumerate(pbar):
            ids  = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            lbls = batch["labels"].to(DEVICE)
            with autocast(enabled=USE_AMP):
                logits = model(input_ids=ids, attention_mask=mask).logits
                loss   = loss_fn(logits, lbls) / GRAD_ACCUM
            scaler.scale(loss).backward()
            total_loss += loss.item() * GRAD_ACCUM
            if (step + 1) % GRAD_ACCUM == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
            pbar.set_postfix({"loss": f"{total_loss / (step + 1):.4f}"})

        val_p           = get_probs(model, val_loader, desc="Val")
        val_pred_labels = [IDX_TO_SUBMIT[i] for i in val_p.argmax(axis=1)]
        val_true_labels = [IDX_TO_SUBMIT[l] for l in fold_val_labels]
        val_f1          = f1_score(val_true_labels, val_pred_labels, average="macro")

        improved = val_f1 > best_f1
        status   = "✓ 新最佳" if improved else f"no_improve={no_improve+1}/{PATIENCE}"
        print(f"  Epoch {epoch}/{EPOCHS}  loss={total_loss/len(train_loader):.4f}  val_F1={val_f1:.4f}  {status}")

        if improved:
            best_f1    = val_f1
            no_improve = 0
            torch.save(model.state_dict(), best_path)
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  [EarlyStop] 停止訓練")
                break

    print(f"\n  Fold {fold+1} 最佳 Val F1 = {best_f1:.4f}，生成預測中 ...")
    fold_best_f1s.append(best_f1)
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    oof_probs[val_idx]  = get_probs(model, val_loader,  desc=f"OOF  Fold{fold+1}")
    test_probs         += get_probs(model, test_loader, desc=f"Test Fold{fold+1}") / N_FOLDS

    del model, optimizer, scheduler, scaler, loss_fn
    del train_ds, val_ds, train_loader, val_loader
    torch.cuda.empty_cache()
    gc.collect()
    print(f"  Fold {fold+1} 完成，GPU 記憶體已釋放")

print(f"\n{'='*60}")
print(f"  K-Fold 訓練完成")
print(f"{'='*60}")
for i, f1 in enumerate(fold_best_f1s):
    print(f"  Fold {i+1} 最佳 Val F1: {f1:.4f}")
print(f"  平均 Val F1: {np.mean(fold_best_f1s):.4f}")

oof_pred_labels = [IDX_TO_SUBMIT[i] for i in oof_probs.argmax(axis=1)]
oof_true_labels = [IDX_TO_SUBMIT[l] for l in all_labels]
oof_f1          = f1_score(oof_true_labels, oof_pred_labels, average="macro")

label_names = [f"{i+1}:{LABEL_STR_LIST[i][:14]}" for i in range(5)]
print(f"\nOOF Macro F1 (全量訓練集): {oof_f1:.4f}")
print(classification_report(oof_true_labels, oof_pred_labels, target_names=label_names))

test_pred_submit = [IDX_TO_SUBMIT[i] for i in test_probs.argmax(axis=1)]
submission = pd.read_csv(SUBMIT_PATH)
submission["label"] = test_pred_submit
submit_out = os.path.join(OUTPUT_DIR, "submission.csv")
submission.to_csv(submit_out, index=False)

np.save(os.path.join(OUTPUT_DIR, "oof_probs_base.npy"),  oof_probs)
np.save(os.path.join(OUTPUT_DIR, "test_probs_base.npy"), test_probs)

print(f"\n提交檔 → {submit_out}")
print(f"預測分布:\n{pd.Series(test_pred_submit).value_counts().sort_index()}")
print(f"\n完成！OOF Macro F1 = {oof_f1:.4f}")
