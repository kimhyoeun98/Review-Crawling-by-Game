"""
KcELECTRA 기반 감성 분석기.

model/kcelectra/ 에 파인튜닝 모델이 있으면 그것을 사용하고,
없으면 HuggingFace 허브의 beomi/kcelectra-base-v2022 를 사용합니다.
(파인튜닝 없이 base 모델만으로는 정확도가 낮습니다 — 먼저 train_kcelectra.py 를 실행하세요.)
"""

import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

BASE_DIR        = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FINETUNED_DIR   = os.path.join(BASE_DIR, "model", "kcelectra")
PRETRAINED_NAME = "beomi/kcelectra-base-v2022"
MAX_LEN         = 128


class SentimentAnalyzer:
    """
    KcELECTRA 기반 Steam 리뷰 감성 분류기.

    사용 예:
        analyzer = SentimentAnalyzer()
        result   = analyzer.analyze("이 게임 정말 재미있어요!")
        # {'label': '긍정', 'pos_prob': 0.97, 'neg_prob': 0.03}
    """

    def __init__(self):
        self._tokenizer = None
        self._model     = None
        self._device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _ensure_loaded(self):
        if self._model is not None:
            return

        if os.path.isdir(FINETUNED_DIR) and os.path.exists(
            os.path.join(FINETUNED_DIR, "config.json")
        ):
            model_path = FINETUNED_DIR
        else:
            # 파인튜닝 모델이 없으면 허브에서 base 모델 로드 (정확도 낮음)
            model_path = PRETRAINED_NAME

        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._model     = AutoModelForSequenceClassification.from_pretrained(
            model_path, num_labels=2
        )
        self._model.to(self._device)
        self._model.eval()

    def analyze(self, text: str) -> dict:
        """
        Returns:
            {'label': '긍정'|'부정'|'판단불가', 'pos_prob': float, 'neg_prob': float}
        """
        self._ensure_loaded()

        if not isinstance(text, str) or not text.strip():
            return {'label': '판단불가', 'pos_prob': 0.5, 'neg_prob': 0.5}

        enc = self._tokenizer(
            text,
            truncation=True,
            max_length=MAX_LEN,
            padding='max_length',
            return_tensors='pt',
        )
        enc = {k: v.to(self._device) for k, v in enc.items()}

        with torch.no_grad():
            logits = self._model(**enc).logits

        probs    = F.softmax(logits, dim=-1)[0].cpu().tolist()
        neg_prob, pos_prob = probs[0], probs[1]
        label    = '긍정' if pos_prob >= 0.5 else '부정'
        return {'label': label, 'pos_prob': pos_prob, 'neg_prob': neg_prob}

    def is_too_negative(self, text: str, threshold: float = 0.70) -> tuple[bool, float]:
        """부정 확률이 threshold 이상이면 (True, neg_prob) 반환."""
        result   = self.analyze(text)
        neg_prob = result['neg_prob']
        return neg_prob >= threshold, neg_prob
