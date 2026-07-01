"""
전체 리뷰를 KcELECTRA 모델로 감성 분석하여 점수를 미리 계산(캐시)합니다.
카테고리 통계가 '평점(추천/비추천)'이 아닌 '모델이 분석한 본문 감성'을 쓰도록 하기 위함.

실행:
    python predict_all_sentiments.py

입력:  data/steam_raw_reviews.jsonl
출력:  data/steam_raw_reviews_scored.jsonl
        (game, rating, model_label, pos_prob, review)
        - rating      : 원본 평점 (1=추천, 0=비추천)  ← 비교용
        - model_label : 모델 예측 (1=긍정, 0=부정)
"""

import json
import os
import re
import sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

MODEL_DIR = "./model/kcelectra"
IN_FILE   = "./data/steam_raw_reviews.jsonl"
OUT_FILE  = "./data/steam_raw_reviews_scored.jsonl"
MAX_LEN   = 128
BATCH     = 256


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[\r\n\xa0]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = (device.type == 'cuda')
    print(f"디바이스: {device} | FP16: {use_amp}")

    print(f"모델 로딩: {MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()

    # 1. 전체 리뷰 로드
    print("리뷰 로딩 중...")
    records = []
    with open(IN_FILE, encoding='utf-8') as f:
        for line in f:
            try:
                d = json.loads(line)
                records.append({
                    'game': d.get('게임 이름', ''),
                    'rating': 1 if d.get('평점') == '추천' else 0,
                    'review': d.get('리뷰 내용', ''),
                })
            except Exception:
                pass
    total = len(records)
    print(f"  총 {total:,}건")

    # 2. 배치 추론
    print(f"감성 분석 중 (batch={BATCH})...")
    done = 0
    agree = 0
    with open(OUT_FILE, 'w', encoding='utf-8') as out:
        for start in range(0, total, BATCH):
            batch = records[start:start + BATCH]
            texts = [clean_text(r['review']) or "." for r in batch]

            enc = tokenizer(
                texts, truncation=True, max_length=MAX_LEN,
                padding=True, return_tensors='pt',
            )
            enc = {k: v.to(device) for k, v in enc.items()}

            with torch.no_grad():
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    logits = model(**enc).logits
            probs = F.softmax(logits, dim=-1).float().cpu()
            pos_probs = probs[:, 1].tolist()

            for r, p in zip(batch, pos_probs):
                model_label = 1 if p >= 0.5 else 0
                if model_label == r['rating']:
                    agree += 1
                out.write(json.dumps({
                    'game': r['game'],
                    'rating': r['rating'],
                    'model_label': model_label,
                    'pos_prob': round(float(p), 4),
                    'review': r['review'],
                }, ensure_ascii=False) + '\n')

            done += len(batch)
            if (start // BATCH) % 50 == 0 or done >= total:
                print(f"  {done:,}/{total:,} ({done/total*100:.0f}%)")

    rate = agree / total * 100
    print(f"\n완료 → {OUT_FILE}")
    print(f"평점 vs 모델 일치율: {agree:,}/{total:,} = {rate:.1f}%")
    print(f"불일치(평점과 본문 감성이 다른 리뷰): {total-agree:,}건 ({100-rate:.1f}%)")


if __name__ == "__main__":
    main()
