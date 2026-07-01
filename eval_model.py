"""
저장된 KcELECTRA 모델을 학습 당시와 동일한 검증셋(10%)으로 재평가.
학습 로그가 지워져서 검증 지표(정밀도/재현율/F1/혼동행렬)를 다시 생성하기 위함.

* 학습 당시와 동일 조건: 2nd Crawling Data, 클래스별 3만, random_state=42, test_size=0.1
"""

import json
import os
import re
import sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

MODEL_DIR     = "./model/kcelectra"
DATA_FILE     = "./data/_archive_crawls/2nd Crawling Data/steam_raw_reviews_2.jsonl"  # 학습 당시 원본
MAX_LEN       = 128
BATCH         = 256
MAX_PER_CLASS = 30_000


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[\r\n\xa0]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def load_data():
    pos, neg = [], []
    with open(DATA_FILE, encoding='utf-8') as f:
        for line in f:
            if len(pos) >= MAX_PER_CLASS and len(neg) >= MAX_PER_CLASS:
                break
            try:
                d = json.loads(line)
                t = clean_text(d.get('리뷰 내용', ''))
                if not t or len(t) < 5:
                    continue
                lab = 1 if d.get('평점') == '추천' else 0
                if lab == 1 and len(pos) < MAX_PER_CLASS:
                    pos.append(t)
                elif lab == 0 and len(neg) < MAX_PER_CLASS:
                    neg.append(t)
            except Exception:
                pass
    texts = pos + neg
    labels = [1] * len(pos) + [0] * len(neg)
    return texts, labels


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = (device.type == 'cuda')
    print(f"디바이스: {device}")

    texts, labels = load_data()
    # 학습과 동일한 분할 재현
    _, X_val, _, y_val = train_test_split(
        texts, labels, test_size=0.1, random_state=42, stratify=labels
    )
    print(f"검증셋: {len(X_val):,}건 (추천 {sum(y_val):,} / 비추천 {len(y_val)-sum(y_val):,})")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()

    preds = []
    for start in range(0, len(X_val), BATCH):
        batch = X_val[start:start + BATCH]
        enc = tokenizer(batch, truncation=True, max_length=MAX_LEN,
                        padding=True, return_tensors='pt')
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(**enc).logits
        preds.extend(logits.argmax(dim=-1).cpu().tolist())

    acc = accuracy_score(y_val, preds)
    print(f"\n=== 검증 결과 ===")
    print(f"정확도(Accuracy): {acc:.4f} ({acc*100:.1f}%)")
    print(classification_report(y_val, preds, target_names=['비추천(부정)', '추천(긍정)'], digits=4))
    cm = confusion_matrix(y_val, preds)
    print("혼동행렬 [행=실제, 열=예측]  (비추천, 추천)")
    print(f"  실제 비추천: {cm[0].tolist()}")
    print(f"  실제 추천  : {cm[1].tolist()}")


if __name__ == "__main__":
    main()
