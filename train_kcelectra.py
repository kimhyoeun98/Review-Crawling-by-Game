"""
KcELECTRA (beomi/kcelectra-base-v2022) 기반 Steam 리뷰 감성 분류 파인튜닝.

실행:
    python train_kcelectra.py

결과물:
    model/kcelectra/  — 파인튜닝된 모델 + 토크나이저
"""

import json
import os
import re
import sys
import torch
import numpy as np

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# ── 설정 ─────────────────────────────────────────────────────────────────────
MODEL_NAME   = "beomi/kcelectra-base-v2022"
DATA_FILE    = "./data/steam_raw_reviews.jsonl"
SAVE_DIR     = "./model/kcelectra"
MAX_LEN      = 128
BATCH_SIZE   = 64        # A4000 16GB + FP16 여유
EPOCHS       = 2
LR           = 2e-5
MAX_PER_CLASS = 30_000


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[\r\n\xa0]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def load_data() -> tuple[list[str], list[int]]:
    """추천/비추천을 클래스별 MAX_PER_CLASS 건씩 균등 로드."""
    pos_texts, neg_texts = [], []

    with open(DATA_FILE, encoding='utf-8') as f:
        for line in f:
            if len(pos_texts) >= MAX_PER_CLASS and len(neg_texts) >= MAX_PER_CLASS:
                break
            try:
                d = json.loads(line)
                text = clean_text(d.get('리뷰 내용', ''))
                if not text or len(text) < 5:
                    continue
                label = 1 if d.get('평점') == '추천' else 0
                if label == 1 and len(pos_texts) < MAX_PER_CLASS:
                    pos_texts.append(text)
                elif label == 0 and len(neg_texts) < MAX_PER_CLASS:
                    neg_texts.append(text)
            except Exception:
                pass

    texts  = pos_texts + neg_texts
    labels = [1] * len(pos_texts) + [0] * len(neg_texts)
    return texts, labels


class ReviewDataset(Dataset):
    """토큰화만 수행(패딩 X). 패딩은 DataCollator가 배치마다 동적으로 처리."""

    def __init__(self, texts: list[str], labels: list[int],
                 tokenizer, max_len: int):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_len,
        )
        enc['labels'] = self.labels[idx]
        return enc


def evaluate(model, loader, device) -> tuple[float, str]:
    model.eval()
    use_amp = (device.type == 'cuda')
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(**batch).logits
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch['labels'].cpu().numpy())
    acc    = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=['비추천', '추천'])
    return acc, report


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"디바이스: {device}")
    if device.type == 'cpu':
        print(" GPU가 없으면 학습에 수 시간이 소요됩니다. CUDA 환경을 권장합니다.")

    # 1. 데이터 로딩
    print(f"\n[1단계] 데이터 로딩 (클래스별 최대 {MAX_PER_CLASS:,}건)...")
    texts, labels = load_data()
    pos = sum(labels)
    neg = len(labels) - pos
    print(f"  총 {len(texts):,}건 | 추천: {pos:,} / 비추천: {neg:,}")

    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, test_size=0.1, random_state=42, stratify=labels
    )
    print(f"  학습: {len(X_train):,}건 / 검증: {len(X_val):,}건")

    # 2. 토크나이저 & 모델 로딩
    print(f"\n[2단계] 토크나이저/모델 로딩: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    )
    model.to(device)

    # 3. DataLoader (동적 패딩 collator)
    print("\n[3단계] DataLoader 구성 중...")
    collator = DataCollatorWithPadding(tokenizer, padding='longest')
    train_ds = ReviewDataset(X_train, y_train, tokenizer, MAX_LEN)
    val_ds   = ReviewDataset(X_val,   y_val,   tokenizer, MAX_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collator, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE * 2, shuffle=False,
                              collate_fn=collator, num_workers=0, pin_memory=True)

    # 4. Optimizer & Scheduler
    optimizer    = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps  = len(train_loader) * EPOCHS
    scheduler    = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps,
    )

    # 5. 학습 루프 (FP16 혼합정밀)
    use_amp = (device.type == 'cuda')
    scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)
    print(f"\n[4단계] 학습 시작 ({EPOCHS} epochs, batch={BATCH_SIZE}, lr={LR}, FP16={use_amp})...")
    best_acc = 0.0
    os.makedirs(SAVE_DIR, exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for step, batch in enumerate(train_loader, 1):
            batch  = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss = model(**batch).loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running_loss += loss.item()

            if step % 200 == 0:
                avg = running_loss / step
                print(f"  Epoch {epoch+1}/{EPOCHS} | step {step}/{len(train_loader)} | loss {avg:.4f}")

        acc, report = evaluate(model, val_loader, device)
        print(f"\n  ── Epoch {epoch+1} 검증 결과 ──")
        print(f"  정확도: {acc:.4f} ({acc*100:.1f}%)")
        print(report)

        if acc > best_acc:
            best_acc = acc
            model.save_pretrained(SAVE_DIR)
            tokenizer.save_pretrained(SAVE_DIR)
            print(f"  최고 모델 저장: {SAVE_DIR}  (acc={best_acc:.4f})\n")

    print(f"학습 완료 — 최종 최고 정확도: {best_acc:.4f}")
    print(f"모델 경로: {SAVE_DIR}")


if __name__ == "__main__":
    train()
