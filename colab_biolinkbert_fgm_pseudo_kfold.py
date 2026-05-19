"""
BioLinkBERT-large + FGM + Pseudo-Labeling K-Fold（Colab 版）
=============================================================
在 FGM 訓練基礎上加入 pseudo-labeled test data（799 筆，threshold=0.75）。

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
  !python colab_biolinkbert_fgm_pseudo_kfold.py

  # Cell 4：下載結果
  from google.colab import files
  files.download('/content/drive/MyDrive/kaggle_biolinkbert_fgm_pseudo/oof_probs_biolinkbert_fgm_pseudo.npy')
  files.download('/content/drive/MyDrive/kaggle_biolinkbert_fgm_pseudo/test_probs_biolinkbert_fgm_pseudo.npy')
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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

warnings.filterwarnings("ignore")

REPO_DIR    = "."
TRAIN_PATH  = os.path.join(REPO_DIR, "kaggle_trainset.csv")
TEST_PATH   = os.path.join(REPO_DIR, "kaggle_testset.csv")
PSEUDO_PATH = os.path.join(REPO_DIR, "pseudo_labels.csv")
SUBMIT_PATH = os.path.join(REPO_DIR, "kaggle_testset_submission.csv")

_DRIVE_DIR = "/content/drive/MyDrive/kaggle_biolinkbert_fgm_pseudo"
_LOCAL_DIR = "outputs/colab_biolinkbert_fgm_pseudo"
OUTPUT_DIR = _DRIVE_DIR if os.path.exists("/content/drive/MyDrive") else _LOCAL_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(_LOCAL_DIR, exist_ok=True)

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_GPUS   = torch.cuda.device_count()
USE_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
USE_AMP  = torch.cuda.is_available() and not USE_BF16
print(f"Device: {DEVICE}  |  GPUs: {N_GPUS}  |  BF16: {USE_BF16}  |  FP16 AMP: {USE_AMP}")

MODEL_NAME   = "michiyasunaga/BioLinkBERT-large"
MAX_LEN      = 384
_is_a100     = torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory > 30e9
BATCH_SIZE   = 16 if _is_a100 else 8
GRAD_ACCUM   = 4  if _is_a100 else 8
N_FOLDS      = 2
EPOCHS       = 8
LR           = 8e-6
WARMUP_RATIO = 0.20
LABEL_SMOOTH = 0.1
FGM_EPSILON  = 1.0
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


class FGM:
    def __init__(self, model, emb_name="embeddings.word_embeddings", epsilon=1.0):
        self.model     = model
        self.emb_name  = emb_name
        self.epsilon   = epsilon
        self.backup    = {}

    def attack(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


torch.manual_seed(SEED); np.random.seed(SEED)

# ── 載入資料 ──────────────────────────────────────────────────────────
train_df  = pd.read_csv(TRAIN_PATH)
test_df   = pd.read_csv(TEST_PATH)
pseudo_df = pd.read_csv(PSEUDO_PATH)

print(f"原始訓練集: {len(train_df)} 筆")
print(f"Pseudo labels: {len(pseudo_df)} 筆（threshold=0.75）")

# 多數投票清洗原始訓練集
label_counts = train_df.groupby(["condition", "label"]).size().reset_index(name="cnt")
majority_idx = label_counts.groupby("condition")["cnt"].idxmax()
df_clean     = label_counts.loc[majority_idx, ["condition", "label"]].reset_index(drop=True)
print(f"清洗後原始訓練集: {len(df_clean)} 筆")

# 合併 pseudo labels（直接附加，不重複清洗）
df_combined = pd.concat([df_clean, pseudo_df], ignore_index=True)
print(f"合併後總訓練集: {len(df_combined)} 筆")
print(f"\n標籤分布:\n{df_combined['label'].value_counts()}")
print(f"\n測試集: {len(test_df)} 筆")

all_texts  = df_combined["condition"].tolist()
all_labels = [STR_TO_IDX[lbl] for lbl in df_combined["label"]]
test_texts = test_df["condition"].tolist()

# OOF 只在原始訓練集上評估
orig_n = len(df_clean)

print("\n載入 tokenizer ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

test_ds     = MedicalDataset(test_texts, None, tokenizer)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

_oof_path  = os.path.join(OUTPUT_DIR, "oof_probs_biolinkbert_fgm_pseudo.npy")
_test_path = os.path.join(OUTPUT_DIR, "test_probs_biolinkbert_fgm_pseudo.npy")
_done_path = os.path.join(OUTPUT_DIR, "completed_folds.txt")

if os.path.exists(_oof_path) and os.path.exists(_test_path):
    oof_probs  = np.load(_oof_path)
    test_probs = np.load(_test_path)
    print("  ✓ 載入上次中斷的部分結果")
else:
    oof_probs  = np.zeros((orig_n, 5))
    test_probs = np.zeros((len(test_texts), 5))

completed_folds = set()
if os.path.exists(_done_path):
    with open(_done_path) as f:
        completed_folds = {int(l.strip()) for l in f if l.strip()}
    print(f"  ✓ 已完成 fold: {sorted(completed_folds)}，跳過重跑")

# K-Fold 只對原始訓練集做 split，pseudo data 全部加入 train
orig_labels_for_split = [STR_TO_IDX[lbl] for lbl in df_clean["label"]]
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)


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


for fold, (orig_train_idx, orig_val_idx) in enumerate(
    skf.split(range(orig_n), orig_labels_for_split)
):
    if fold in completed_folds:
        print(f"\n  [SKIP] Fold {fold+1} 已完成，跳過")
        continue

    print(f"\n{'='*60}")
    print(f"  FOLD {fold+1}/{N_FOLDS}")
    print(f"{'='*60}")

    # train = fold的原始訓練部分 + 全部pseudo data
    pseudo_indices = list(range(orig_n, len(all_texts)))
    train_idx_full = list(orig_train_idx) + pseudo_indices
    val_idx_orig   = list(orig_val_idx)

    fold_train_texts  = [all_texts[i]  for i in train_idx_full]
    fold_train_labels = [all_labels[i] for i in train_idx_full]
    fold_val_texts    = [all_texts[i]  for i in val_idx_orig]
    fold_val_labels   = [all_labels[i] for i in val_idx_orig]

    print(f"  Train: {len(fold_train_texts)} (原始{len(orig_train_idx)} + pseudo{len(pseudo_indices)})  Val: {len(fold_val_texts)}")

    train_ds = MedicalDataset(fold_train_texts, fold_train_labels, tokenizer)
    val_ds   = MedicalDataset(fold_val_texts,   fold_val_labels,   tokenizer)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    base_model, model = build_model()

    cw            = compute_class_weight("balanced", classes=np.arange(5), y=fold_train_labels)
    class_weights = torch.tensor(cw, dtype=torch.float).to(DEVICE)
    loss_fn       = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTH)
    fgm           = FGM(base_model, emb_name="embeddings.word_embeddings", epsilon=FGM_EPSILON)

    total_steps  = (len(train_loader) // GRAD_ACCUM) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    optimizer    = AdamW(base_model.parameters(), lr=LR, weight_decay=0.01)
    scheduler    = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    scaler = GradScaler(enabled=USE_AMP)

    best_f1    = 0.0
    no_improve = 0
    best_path  = os.path.join(OUTPUT_DIR, f"biolinkbert_fgm_pseudo_fold{fold+1}.pt")

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
            loss = loss_fn(logits.float(), lbls) / GRAD_ACCUM

            if USE_AMP:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # FGM 對抗訓練
            fgm.attack()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=USE_BF16):
                logits_adv = model(input_ids=ids, attention_mask=mask).logits
            loss_adv = loss_fn(logits_adv.float(), lbls) / GRAD_ACCUM
            if USE_AMP:
                scaler.scale(loss_adv).backward()
            else:
                loss_adv.backward()
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
            pbar.set_postfix({"loss": f"{total_loss / (step + 1):.4f}"})

        val_p    = get_probs(model, val_loader, desc="Val")
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
    oof_probs[orig_val_idx] = get_probs(model, val_loader,  desc=f"OOF  Fold{fold+1}")
    test_probs             += get_probs(model, test_loader, desc=f"Test Fold{fold+1}") / N_FOLDS

    del model, base_model, optimizer, scheduler, scaler, loss_fn, fgm
    del train_ds, val_ds, train_loader, val_loader
    torch.cuda.empty_cache()
    gc.collect()

    np.save(_oof_path,  oof_probs)
    np.save(_test_path, test_probs)
    with open(_done_path, "a") as f:
        f.write(f"{fold}\n")
    completed_folds.add(fold)
    print(f"  Fold {fold+1} 完成，已存至 {OUTPUT_DIR}")

# ── 最終結果 ──────────────────────────────────────────────────────────
print(f"\n{'='*60}\n  BioLinkBERT-FGM + Pseudo-Label 訓練完成\n{'='*60}")

orig_labels_true = [IDX_TO_SUBMIT[l] for l in orig_labels_for_split]
oof_pred = [IDX_TO_SUBMIT[i] for i in oof_probs.argmax(axis=1)]
oof_f1   = f1_score(orig_labels_true, oof_pred, average="macro")

label_names = [f"{i+1}:{LABEL_STR_LIST[i][:14]}" for i in range(5)]
print(f"\nOOF Macro F1 (原始訓練集): {oof_f1:.4f}")
print(classification_report(orig_labels_true, oof_pred, target_names=label_names))

test_pred_submit = [IDX_TO_SUBMIT[i] for i in test_probs.argmax(axis=1)]
submission = pd.read_csv(SUBMIT_PATH)
submission["label"] = test_pred_submit
submission.to_csv(os.path.join(OUTPUT_DIR, "submission_fgm_pseudo.csv"), index=False)

np.save(_oof_path,  oof_probs)
np.save(_test_path, test_probs)

print(f"\n提交檔 → {OUTPUT_DIR}/submission_fgm_pseudo.csv")
print(f"預測分布:\n{pd.Series(test_pred_submit).value_counts().sort_index()}")
print(f"\n完成！OOF Macro F1 = {oof_f1:.4f}")
print(f"\n[後續] 下載 npy 檔用於 ensemble")
