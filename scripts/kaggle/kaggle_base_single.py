# ============================================================
# Cell 1: 安裝套件
# ============================================================
# !pip install -q sentence-transformers


# ============================================================
# Cell 2: Import & 設定
# ============================================================
import os, time, warnings, numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    get_cosine_schedule_with_warmup,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from tqdm.notebook import tqdm   # Kaggle notebook 用這個版本顯示較好看

warnings.filterwarnings("ignore")

# ── 資料路徑 ─────────────────────────────────────────────────
data_folder = "/kaggle/input/competitions/1142-medical-condition-classification"

TRAIN_PATH  = os.path.join(data_folder, "kaggle_trainset.csv")
TEST_PATH   = os.path.join(data_folder, "kaggle_testset.csv")
SUBMIT_PATH = os.path.join(data_folder, "kaggle_testset_submission.csv")
OUTPUT_DIR  = "/kaggle/working"

print(f"資料路徑: {data_folder}")
print(f"  train : {TRAIN_PATH}")
print(f"  test  : {TEST_PATH}")
print(f"  submit: {SUBMIT_PATH}")

# ── Hyperparameters（Phase 4v2 優化版）────────────────────────
MODEL_NAME   = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
MAX_LEN      = 512
BATCH_SIZE   = 16
GRAD_ACCUM   = 2          # 等效 batch=32
EPOCHS       = 10
LR           = 1e-5
WARMUP_RATIO = 0.20
LABEL_SMOOTH = 0.1
VAL_RATIO    = 0.2
PATIENCE     = 3
SEED         = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {DEVICE}")

LABEL_STR_LIST = [
    "neoplasms",
    "digestive system diseases",
    "nervous system diseases",
    "cardiovascular diseases",
    "general pathological conditions",
]
STR_TO_IDX    = {s: i for i, s in enumerate(LABEL_STR_LIST)}
IDX_TO_SUBMIT = {i: i + 1 for i in range(5)}


# ============================================================
# Cell 3: Dataset
# ============================================================
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


# ============================================================
# Cell 4: 載入資料
# ============================================================
torch.manual_seed(SEED); np.random.seed(SEED)

train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)

print(f"訓練集: {len(train_df)} 筆")
print(f"測試集: {len(test_df)} 筆")
print(f"\n標籤分布:\n{train_df['label'].value_counts()}")

all_texts  = train_df["condition"].tolist()
all_labels = [STR_TO_IDX[lbl] for lbl in train_df["label"]]
test_texts = test_df["condition"].tolist()

train_texts, val_texts, train_labels, val_labels = train_test_split(
    all_texts, all_labels,
    test_size=VAL_RATIO, stratify=all_labels, random_state=SEED,
)
print(f"\nTrain: {len(train_texts)}  Val: {len(val_texts)}")


# ============================================================
# Cell 5: Tokenizer & DataLoader
# ============================================================
print("載入 tokenizer ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_ds = MedicalDataset(train_texts, train_labels, tokenizer)
val_ds   = MedicalDataset(val_texts,   val_labels,   tokenizer)
test_ds  = MedicalDataset(test_texts,  None,         tokenizer)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

print(f"Train batches: {len(train_loader)}  Val batches: {len(val_loader)}")


# ============================================================
# Cell 6: 模型、Loss、Optimizer
# ============================================================
print("載入預訓練模型 ...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=5
).to(DEVICE)

# Class weights（處理類別不平衡）
classes      = np.arange(5)
cw           = compute_class_weight("balanced", classes=classes, y=train_labels)
class_weights = torch.tensor(cw, dtype=torch.float).to(DEVICE)
print(f"Class weights: {cw.round(3)}")

loss_fn = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTH)

total_steps  = (len(train_loader) // GRAD_ACCUM) * EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)

optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
scheduler = get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
)
print(f"Total steps: {total_steps}  Warmup: {warmup_steps}")


# ============================================================
# Cell 7: 訓練函式
# ============================================================
def train_one_epoch(epoch):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]", leave=True)

    for step, batch in enumerate(pbar):
        ids   = batch["input_ids"].to(DEVICE)
        mask  = batch["attention_mask"].to(DEVICE)
        lbls  = batch["labels"].to(DEVICE)

        logits = model(input_ids=ids, attention_mask=mask).logits
        loss   = loss_fn(logits, lbls) / GRAD_ACCUM
        loss.backward()
        total_loss += loss.item() * GRAD_ACCUM

        if (step + 1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        pbar.set_postfix({"loss": f"{total_loss / (step + 1):.4f}"})

    return total_loss / len(train_loader)


@torch.no_grad()
def evaluate():
    model.eval()
    preds_all, labels_all = [], []
    for batch in val_loader:
        ids  = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        lbls = batch["labels"].to(DEVICE)
        pred = model(input_ids=ids, attention_mask=mask).logits.argmax(dim=-1)
        preds_all.extend(pred.cpu().numpy())
        labels_all.extend(lbls.cpu().numpy())
    return f1_score(labels_all, preds_all, average="macro"), labels_all, preds_all


@torch.no_grad()
def predict_test():
    model.eval()
    all_logits = []
    for batch in tqdm(test_loader, desc="Predicting test", leave=True):
        ids  = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        logits = model(input_ids=ids, attention_mask=mask).logits
        all_logits.append(logits.cpu().numpy())
    return np.vstack(all_logits)


# ============================================================
# Cell 8: 訓練迴圈
# ============================================================
best_f1    = 0.0
no_improve = 0
best_path  = os.path.join(OUTPUT_DIR, "best_model.pt")
history    = []

for epoch in range(1, EPOCHS + 1):
    t0         = time.time()
    train_loss = train_one_epoch(epoch)
    val_f1, val_true, val_pred = evaluate()
    elapsed    = time.time() - t0

    improved = val_f1 > best_f1
    status   = "✓ 新最佳" if improved else f"no_improve={no_improve+1}/{PATIENCE}"
    print(f"Epoch {epoch}/{EPOCHS}  loss={train_loss:.4f}  val_F1={val_f1:.4f}  ({elapsed:.0f}s)  {status}")
    history.append({"epoch": epoch, "train_loss": train_loss, "val_f1": val_f1})

    if improved:
        best_f1    = val_f1
        no_improve = 0
        torch.save(model.state_dict(), best_path)
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            print(f"\n[EarlyStop] {PATIENCE} 個 epoch 無進步，停止訓練")
            break

pd.DataFrame(history).to_csv(os.path.join(OUTPUT_DIR, "training_history.csv"), index=False)


# ============================================================
# Cell 9: 最終評估 & 提交
# ============================================================
print(f"\n載入最佳模型 (Val F1={best_f1:.4f}) ...")
model.load_state_dict(torch.load(best_path, map_location=DEVICE))
val_f1, val_true, val_pred = evaluate()

label_names = [f"{i+1}:{LABEL_STR_LIST[i][:14]}" for i in range(5)]
print("\n" + "=" * 60)
print(f"  最終 Val Macro F1: {val_f1:.4f}")
print("=" * 60)
print(classification_report(val_true, val_pred, target_names=label_names))

# 對測試集推論
test_logits      = predict_test()
test_pred_submit = [IDX_TO_SUBMIT[i] for i in test_logits.argmax(axis=1)]

# 生成提交檔
submission = pd.read_csv(SUBMIT_PATH)
submission["label"] = test_pred_submit
submit_out = os.path.join(OUTPUT_DIR, "submission.csv")
submission.to_csv(submit_out, index=False)

print(f"\n提交檔已儲存 → {submit_out}")
print(f"預測分布:\n{pd.Series(test_pred_submit).value_counts().sort_index()}")
print(f"\n完成！Best Val F1 = {best_f1:.4f}")
