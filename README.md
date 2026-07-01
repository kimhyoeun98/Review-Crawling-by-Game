# 🎮 Steam 리뷰 감성 분석 서비스

> **스팀 게임 리뷰를 대량 수집하고, KcELECTRA 딥러닝 모델로 본문 감성을 분석하는 웹 서비스**

Steam 리뷰를 장르별로 크롤링해 한국어 감성 분석 모델(KcELECTRA)을 학습하고, 장르·게임별 긍정/부정 경향을 시각화하며, 부정적인 리뷰는 등록 단계에서 자동으로 걸러내는 Streamlit 기반 서비스입니다.

---

## ✨ 핵심 기능

| 기능 | 설명 |
|------|------|
| 📊 **카테고리 통계** | 7개 장르의 리뷰 긍정/부정 비율을 **모델이 분석한 본문 감성** 기준으로 시각화. 막대 클릭 시 실제 리뷰 열람 |
| 🔥 **실시간 인기 게임** | Steam API로 동시접속자 TOP 게임을 조회하고, 수집된 리뷰의 긍정률과 함께 비교 |
| ✍️ **리뷰 감성 검사** | 입력한 리뷰를 KcELECTRA가 분석해 긍정/부정 판정. 부정 확률이 임계값을 넘으면 등록 차단 |

---

## 🧠 모델 성능

동일한 60,000건 데이터·검증셋(6,000건, `random_state=42`)으로 비교:

| 방식 | 검증 정확도 | 비고 |
|------|:----------:|------|
| TF-IDF + LogisticRegression | 87.2% | char n-gram |
| LSTM (Okt + Embedding) | 85.7% | 과적합 경향 |
| **KcELECTRA (채택)** | **89.3%** | Macro F1 0.893 |

> **핵심 발견**: 전체 리뷰의 **13.9%(77,795건)** 는 평점(추천/비추천)과 본문 감성이 달랐습니다. → 평점 집계가 아닌 **본문 감성 분석**이 필요한 이유.

---

* **v1.3.0(current)**
  * **데이터 정제 및 정규화 시작**: 모델 학습의 정확도를 높이기 위해 수집된 리뷰 텍스트의 본격적인 전처리 파이프라인 구축
  * **평점 이진 라벨링**: 감성 분석 모델 학습에 바로 활용할 수 있도록 평점 데이터를 0과 1의 이진 레이블로 변환
  * **텍스트 노이즈 정제**: 불필요한 제어 문자를 공백으로 치환하고 숫자 및 특수기호를 제거하여 순수 텍스트 데이터만 추출
  * **공백 및 중복 데이터 처리**: 다중 공백을 하나의 공백으로 치환하여 텍스트를 표준화하고, 학습 데이터의 편향을 막기 위해 중복 수집된 리뷰를 제거하는 로직 추가 
* **v1.2.0**
  * **데이터 표준화**: 향후 동시 접속자 수(CCU) 데이터와의 원활한 병합(Merge)을 위해 수집 컬럼에 '게임 이름(공식 명칭)' 매핑 로직 추가
  * Steam API 응답 속도 최적화 및 `cursor` 기반 무한 스크롤 누락 방지 로직 개선
* **v1.1.0**
  * 프로그램 강제 종료 및 에러 발생 시를 대비한 체크포인트 저장 시스템(`completed_appids.txt`) 도입
  * 대용량 텍스트 처리를 위한 JSON Lines(`jsonl`) 저장 방식 채택 및 백업 기능 구현
* **v1.0.0**
  * 단일 스크립트에서 객체지향(OOP) 구조로 모듈화 (`main.py`, `SteamAppIDExtractor.py`, `SteamMassiveReviewCrawler.py` 분리)
  * Selenium을 활용한 카테고리별 '최고 인기 제품' AppID 자동 추출(헤드리스 모드 지원) 기능 구현

## 📂 프로젝트 구조

```text
project_webCrawling/
├── main.py                       # 크롤링 → 전처리 파이프라인 진입점
├── lib/                          # 데이터 수집 클래스 모듈
│   ├── SteamAppIDExtractor.py        # Selenium 기반 appid 추출
│   ├── SteamMassiveReviewCrawler.py  # Steam API 리뷰 수집 (appid·카테고리 저장)
│   └── ReviewPreprocessor.py         # 정제 + Okt 토큰화
├── service/                      # 서비스 클래스 모듈
│   ├── app.py                        # Streamlit 메인 앱 (port 8501)
│   ├── SentimentAnalyzer.py          # KcELECTRA 추론 + 차단 판정
│   ├── analytics.py                  # 본문 감성 통계 집계
│   └── live_stats.py                 # Steam 실시간 접속자 API
├── train_kcelectra.py            # KcELECTRA 파인튜닝
├── predict_all_sentiments.py     # 전체 리뷰 모델 감성 분석(캐시 생성)
├── build_category_map.py         # 게임명 → 카테고리 매핑 빌드
├── eval_model.py / eval_baseline.py / eval_lstm.py  # 3개 모델 검증 비교
├── eval_model.py / eval_baseline.py / eval_lstm.py  # 3개 모델 검증 비교
├── model/kcelectra/              # 파인튜닝된 모델 가중치 (Git LFS · 490MB)
├── data/                         # 데이터
│   ├── steam_raw_reviews.jsonl        # 수집 리뷰 (560,520건)
│   ├── steam_raw_reviews_scored.jsonl # 모델 감성 분석 캐시
│   ├── game_category_map.json        # 게임명 → 카테고리
│   ├── completed_appids.txt          # 크롤링 체크포인트
│   └── steam_appid/                  # 장르별 appid 목록
└── docs/                         # 문서
    ├── Steam_리뷰_감성분석_테크니컬_리포트 # 테크니컬 리포트 (설계 문서)
    ├── Steam_리뷰_감성분석_발표.pdf   # 발표 자료 
    └── demo.mp4                      # 시연 영상
```

> 📦 `model/`, `data/steam_dataset.zip`, `docs/demo.mp4` 은 **Git LFS** 로 관리됩니다.

---

## 🚀 실행 방법

### 0. 클론 (Git LFS 필수)
모델·데이터가 LFS로 저장되어 있어 **먼저 Git LFS를 설치**해야 합니다.
```bash
git lfs install
git clone <repo-url>
cd project_webCrawling
git lfs pull            # model/, data/steam_dataset.zip, docs/demo.mp4 내려받기
```

### 1. 환경 설정
```bash
pip install -r requirements.txt
# GPU 사용 시 CUDA 빌드 torch 별도 설치 권장:
# pip install torch --index-url https://download.pytorch.org/whl/cu128
```

### 2. 서비스 실행
```bash
streamlit run service/app.py
# 브라우저에서 http://localhost:8501 접속
```

### 3. 처음부터 재현하기
```bash
# (1) 리뷰 크롤링 + 전처리
python main.py

# (2) 카테고리 매핑 빌드
python build_category_map.py

# (3) KcELECTRA 학습
python train_kcelectra.py

# (4) 전체 리뷰 감성 분석 캐시 생성
python predict_all_sentiments.py

# (5) 서비스 실행
streamlit run service/app.py
```

---

## 🛠 기술 스택

| 분류 | 사용 기술 |
|------|----------|
| **언어** | Python 3.9 |
| **크롤링** | Selenium, Requests, Steam Web API |
| **딥러닝** | PyTorch, Transformers (KcELECTRA `beomi/kcelectra-base-v2022`) |
| **머신러닝/비교** | scikit-learn (TF-IDF, LogReg), TensorFlow/Keras (LSTM), KoNLPy(Okt) |
| **데이터 처리** | pandas |
| **웹/시각화** | Streamlit, Plotly |

---

## 📊 수집 데이터

- **560,520건** 리뷰 · **7개 액션 장르** · 한국어
- 장르: FPS / TPS / 핵앤슬래시 / 리듬·아케이드 / 런앤점프 / 슈팅(shmup) / 격투
- 형식: JSON Lines (`appid`, `게임 이름`, `카테고리`, `작성자`, `리뷰 내용`, `평점`)

---

## 📑 문서

| 문서 | 경로 |
|------|------|
| 테크니컬 리포트 (설계 문서) | [`docs/Steam_리뷰_감성분석_테크니컬 리포트.docx`](docs/Steam_리뷰_감성분석_테크니컬 리포트.docx) · `Steam_리뷰_감성분석_테크니컬 리포트.docx` |
| 발표 자료 (PDF) | `docs/Steam_리뷰_감성분석_발표.pdf` |
| 시연 영상 | `docs/demo.mp4` |

> `data/steam_processed_reviews.csv`(전처리 중간 산출물)와 `steam_raw_reviews_full_961k.jsonl`(초기 대형 크롤)은
> 용량 절감을 위해 저장소에서 제외했습니다. 전자는 `python main.py`(전처리)로, 데이터는 크롤러로 재생성할 수 있습니다.
