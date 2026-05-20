"""
Phase 4: Supervised Fine-tuning
使用 kaggle_trainset.csv 全量資料（12,994 筆）微調醫學 BERT 模型

模型：microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract
策略：
- 80/20 stratified train/val split
- 使用 class weights 處理類別不平衡
- 每個 epoch 計算 val Macro F1，儲存最佳 checkpoint
- 訓練完後對 testset 生成最終預測
"""

import os
import time
import warnings
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")

# ── 設定 ──────────────────────────────────────────────────────────────────────

TRAIN_PATH = "kaggle_trainset.csv"
TEST_PATH  = "kaggle_testset.csv"
OUTPUT_DIR = "outputs/phase4"

MODEL_NAME = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"

MAX_LEN    = 512
BATCH_SIZE = 16
GRAD_ACCUM = 2       # 等效 batch size = 32
EPOCHS     = 5
LR         = 2e-5
WARMUP_RATIO = 0.1
VAL_RATIO  = 0.2
SEED       = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Label 對應：文字 → 模型整數（0-based）→ 提交數字（1-based）
LABEL_STR_LIST = [
    "neoplasms",
    "digestive system diseases",
    "nervous system diseases",
    "cardiovascular diseases",
    "general pathological conditions",
]
STR_TO_IDX = {s: i for i, s in enumerate(LABEL_STR_LIST)}
IDX_TO_SUBMIT = {i: i + 1 for i in range(5)}  # 0→1, 1→2, ..., 4→5
SUBMIT_TO_IDX = {v: k for k, v in IDX_TO_SUBMIT.items()}

# ── 資料集 ────────────────────────────────────────────────────────────────────

class MedicalDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels       # None for test set
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
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


# ── 資料載入與切分 ────────────────────────────────────────────────────────────

def load_train_data():
    df = pd.read_csv(TRAIN_PATH)
    print(f"[Data] 訓練集: {len(df)} 筆")
    print(f"[Data] 標籤分布:\n{df['label'].value_counts()}\n")

    texts  = df["condition"].tolist()
    labels = [STR_TO_IDX[lbl] for lbl in df["label"]]

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels,
        test_size=VAL_RATIO,
        stratify=labels,
        random_state=SEED,
    )
    print(f"[Data] Train: {len(train_texts)}  |  Val: {len(val_texts)}")
    return train_texts, train_labels, val_texts, val_labels


def load_test_data():
    df = pd.read_csv(TEST_PATH)
    print(f"[Data] 測試集: {len(df)} 筆")
    return df["condition"].tolist(), df


# ── 訓練工具 ──────────────────────────────────────────────────────────────────

def compute_class_weights(labels):
    classes = np.arange(len(LABEL_STR_LIST))
    weights = compute_class_weight("balanced", classes=classes, y=labels)
    return torch.tensor(weights, dtype=torch.float).to(DEVICE)


def train_one_epoch(model, loader, optimizer, scheduler, loss_fn, grad_accum, epoch, total_epochs):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [Train]", leave=True)
    for step, batch in enumerate(pbar):
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels         = batch["labels"].to(DEVICE)

        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        loss   = loss_fn(logits, labels) / grad_accum
        loss.backward()
        total_loss += loss.item() * grad_accum

        if (step + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        pbar.set_postfix({"loss": f"{total_loss / (step + 1):.4f}"})

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []

    for batch in loader:
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels         = batch["labels"].to(DEVICE)

        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        preds  = logits.argmax(dim=-1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    return macro_f1, all_labels, all_preds


@torch.no_grad()
def predict(model, loader):
    model.eval()
    all_logits = []

    for batch in loader:
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        all_logits.append(logits.cpu().numpy())

    return np.vstack(all_logits)  # (N, 5)


# ── 主訓練流程 ────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    print(f"[Config] Device: {DEVICE}  |  Model: {MODEL_NAME}")

    # 1. 載入資料
    train_texts, train_labels, val_texts, val_labels = load_train_data()
    test_texts, test_df = load_test_data()

    # 2. Tokenizer
    print("[Setup] 載入 tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 3. Dataset / DataLoader
    train_ds = MedicalDataset(train_texts, train_labels, tokenizer, MAX_LEN)
    val_ds   = MedicalDataset(val_texts,   val_labels,   tokenizer, MAX_LEN)
    test_ds  = MedicalDataset(test_texts,  None,         tokenizer, MAX_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    # 4. 模型
    print("[Setup] 載入預訓練模型 ...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABEL_STR_LIST)
    ).to(DEVICE)

    # 5. Class weights（處理類別不平衡）
    class_weights = compute_class_weights(train_labels)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    print(f"[Setup] Class weights: {class_weights.cpu().numpy().round(3)}")

    # 6. Optimizer & Scheduler
    total_steps  = (len(train_loader) // GRAD_ACCUM) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    print(f"[Setup] Total steps: {total_steps}  |  Warmup: {warmup_steps}\n")

    # 7. 訓練迴圈
    best_f1   = 0.0
    best_path = os.path.join(OUTPUT_DIR, "best_model.pt")
    history   = []

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, loss_fn, GRAD_ACCUM, epoch, EPOCHS)
        val_f1, val_true, val_pred = evaluate(model, val_loader)
        elapsed = time.time() - t0

        print(f"Epoch {epoch}/{EPOCHS}  loss={train_loss:.4f}  val_F1={val_f1:.4f}  ({elapsed:.0f}s)")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_f1": val_f1})

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), best_path)
            print(f"  ✓ 新最佳模型已儲存 (F1={best_f1:.4f})")

    # 8. 最佳模型評估
    print(f"\n[Eval] 載入最佳模型 (Val F1={best_f1:.4f}) ...")
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    val_f1, val_true, val_pred = evaluate(model, val_loader)

    label_names = [f"{i+1}:{LABEL_STR_LIST[i][:12]}" for i in range(5)]
    print("\n" + "=" * 60)
    print(f"  Phase 4 Best Val Macro F1: {val_f1:.4f}")
    print("=" * 60)
    print(classification_report(val_true, val_pred, target_names=label_names))

    # 儲存訓練歷史
    pd.DataFrame(history).to_csv(
        os.path.join(OUTPUT_DIR, "training_history.csv"), index=False
    )

    # 9. 對 testset 推論
    print("[Predict] 對測試集生成預測 ...")
    test_logits = predict(model, test_loader)          # (N, 5)
    test_pred_idx = test_logits.argmax(axis=1)         # 0-based
    test_pred_submit = [IDX_TO_SUBMIT[i] for i in test_pred_idx]   # 1-based

    # 10. 生成提交檔
    submission = pd.read_csv("kaggle_testset_submission.csv")
    submission["label"] = test_pred_submit
    submit_path = os.path.join(OUTPUT_DIR, "submission_phase4.csv")
    submission.to_csv(submit_path, index=False)
    print(f"[Submit] 提交檔已儲存 → {submit_path}")
    print(f"[Submit] 預測分布:\n{pd.Series(test_pred_submit).value_counts().sort_index()}")

    # 11. 同時儲存 soft scores（供後續 Ensemble 使用）
    np.save(os.path.join(OUTPUT_DIR, "test_logits.npy"), test_logits)

    print(f"\n[Done] Phase 4 完成！Best Val F1 = {best_f1:.4f}")
    if best_f1 >= 0.80:
        print("  → 效果優異，直接提交！")
    elif best_f1 >= 0.70:
        print("  → 效果良好，可考慮與 Phase 2 Ensemble 進一步提升")
    else:
        print("  → 建議嘗試更大的模型或更長的訓練")


if __name__ == "__main__":
    main()
