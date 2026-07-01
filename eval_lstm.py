"""
LSTM을 KcELECTRA·TF-IDF와 '동일한' 60k 데이터·검증 분할로 학습/평가.
기존 방식 재현: Okt 형태소 토큰화 + Embedding + LSTM.
공정 비교: random_state=42, test_size=0.1 동일.
"""
import json, re, sys, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from sklearn.model_selection import train_test_split

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

DATA_FILE = "./data/_archive_crawls/2nd Crawling Data/steam_raw_reviews_2.jsonl"
MAX_PER_CLASS = 30_000
VOCAB = 20_000
MAXLEN = 50
EPOCHS = 4
BATCH = 128


def clean_text(t):
    if not isinstance(t, str):
        return ""
    t = re.sub(r'[\r\n\xa0]+', ' ', t)
    t = re.sub(r'[^가-힣ㄱ-ㅎㅏ-ㅣa-zA-Z\s]', '', t)
    return re.sub(r'\s+', ' ', t).strip()


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
    return pos + neg, [1] * len(pos) + [0] * len(neg)


print("[1] 데이터 로드")
texts, labels = load_data()
X_tr, X_val, y_tr, y_val = train_test_split(
    texts, labels, test_size=0.1, random_state=42, stratify=labels)
print(f"  학습 {len(X_tr):,} / 검증 {len(X_val):,}")

print("[2] Okt 형태소 토큰화 (시간 소요)...")
import konlpy.jvm as kjvm
kjvm.init_jvm(max_heap_size=4096)  # 기본 1024MB → 4GB로 상향 (OOM 방지)
from konlpy.tag import Okt
okt = Okt()

def tok(seq):
    out = []
    for i, t in enumerate(seq):
        out.append(okt.morphs(t))
        if (i + 1) % 10000 == 0:
            print(f"    {i+1:,}건 토큰화")
    return out

tr_tok = tok(X_tr)
val_tok = tok(X_val)

print("[3] 시퀀스 인코딩")
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

tk = Tokenizer(num_words=VOCAB, oov_token='<OOV>')
tk.fit_on_texts(tr_tok)
Xtr = pad_sequences(tk.texts_to_sequences(tr_tok), maxlen=MAXLEN)
Xval = pad_sequences(tk.texts_to_sequences(val_tok), maxlen=MAXLEN)
ytr, yval = np.array(y_tr), np.array(y_val)

print("[4] LSTM 학습")
import tempfile, os as _os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, SpatialDropout1D
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score, f1_score, classification_report

model = Sequential([
    Embedding(VOCAB, 100, input_length=MAXLEN),
    SpatialDropout1D(0.2),
    LSTM(64, dropout=0.2, recurrent_dropout=0.0),
    Dense(1, activation='sigmoid'),
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

ckpt = _os.path.join(tempfile.gettempdir(), 'lstm_best.weights.h5')
cb = ModelCheckpoint(ckpt, monitor='val_accuracy', save_best_only=True, save_weights_only=True)
model.fit(Xtr, ytr, validation_data=(Xval, yval),
          epochs=EPOCHS, batch_size=BATCH, verbose=2, callbacks=[cb])

# 최고 성능(best val_accuracy) 가중치로 동일 예측에서 정확도·F1 산출
model.load_weights(ckpt)
proba = model.predict(Xval, batch_size=BATCH, verbose=0).ravel()
pred = (proba >= 0.5).astype(int)
acc = accuracy_score(yval, pred)
f1 = f1_score(yval, pred, average='macro')

print(f"\n[LSTM (Okt + Embedding + LSTM)]")
print(f"  검증 정확도: {acc:.4f} ({acc*100:.1f}%)")
print(f"  Macro F1   : {f1:.4f}")
print(classification_report(yval, pred, target_names=['비추천(부정)', '추천(긍정)'], digits=4))
print(f"=== 3-방식 비교 (동일 검증셋 6,000건) ===")
print(f"  TF-IDF + LogReg : 87.2% / F1 0.872")
print(f"  LSTM            : {acc*100:.1f}% / F1 {f1:.3f}")
print(f"  KcELECTRA       : 89.3% / F1 0.893")
