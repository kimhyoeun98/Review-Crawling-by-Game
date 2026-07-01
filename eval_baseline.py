"""
KcELECTRA와 '동일한' 60k 데이터·검증 분할에서 기존 방식(TF-IDF) 정확도를 측정.
공정한 비교를 위해 random_state=42, test_size=0.1 동일.
"""
import json, re, sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

DATA_FILE = "./data/_archive_crawls/2nd Crawling Data/steam_raw_reviews_2.jsonl"
MAX_PER_CLASS = 30_000


def clean_text(t):
    if not isinstance(t, str):
        return ""
    t = re.sub(r'[\r\n\xa0]+', ' ', t)
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


texts, labels = load_data()
X_tr, X_val, y_tr, y_val = train_test_split(
    texts, labels, test_size=0.1, random_state=42, stratify=labels
)
print(f"학습 {len(X_tr):,} / 검증 {len(X_val):,}")

# 한국어 형태소 없이도 강한 char n-gram TF-IDF
vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4),
                      min_df=3, max_features=100_000, sublinear_tf=True)
Xtr = vec.fit_transform(X_tr)
Xval = vec.transform(X_val)

clf = LogisticRegression(C=4.0, max_iter=1000, n_jobs=-1)
clf.fit(Xtr, y_tr)
pred = clf.predict(Xval)

acc = accuracy_score(y_val, pred)
f1 = f1_score(y_val, pred, average='macro')
print(f"\n[TF-IDF(char 2~4) + LogisticRegression]")
print(f"  검증 정확도: {acc:.4f} ({acc*100:.1f}%)")
print(f"  Macro F1   : {f1:.4f}")
print(f"\n[비교] KcELECTRA: 정확도 89.3% / Macro F1 0.893")
print(f"        차이      : {(0.893-acc)*100:+.1f}%p")
