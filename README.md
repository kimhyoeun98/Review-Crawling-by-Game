# 🎮 Steam Review & CCU Correlation Analyzer 
> **스팀 게임 리뷰 대량 수집 및 감성-동시접속자 상관관계 분석 프로젝트**

이 프로젝트는 스팀(Steam) 상점의 카테고리별 인기 게임 목록과 사용자 리뷰를 대량으로 수집하는 파이썬 기반 크롤링 엔진입니다. 

## 🎯 프로젝트 최종 목표
본 크롤러를 통해 수집된 '사용자 리뷰 감성 데이터'는 향후 **스팀 동시 접속자 수(CCU, Concurrent Users) 데이터와 결합**하여 다음과 같은 인사이트를 도출하는 데 사용됩니다.
* 📈 **상관관계 분석**: 게임의 긍정적/부정적 리뷰 흐름(Sentiment)이 동시 접속자 수 증감에 미치는 영향 파악
* 🔍 **유저 이탈 및 유입 원인 분석**: 패치나 업데이트 시점의 리뷰 감성 변화가 실제 플레이어 수 유지(Retention)에 미치는 파급력 시각화

---

## ✨ 주요 기능
* **카테고리별 AppID 자동 추출**: 셀레니움(Selenium)을 활용해 특정 장르의 인기 게임 리스트를 자동으로 스캔합니다.
* **대량 리뷰 크롤링**: Steam API를 사용하여 한국어 리뷰를 `cursor` 페이징 방식으로 누락 없이 수집합니다.
* **체크포인트 시스템**: 수집 완료된 AppID를 추적하여 중복 저장을 방지하고 효율적인 이어받기를 지원합니다.
* **분석 친화적 데이터 포맷**: 대규모 텍스트 마이닝 및 감성 분석 모델(NLP) 학습에 즉시 활용할 수 있도록 JSON Lines(`jsonl`) 포맷으로 저장합니다.

## 🛠 기술 스택
**Language**
* `Python 3.9.25`

**Web Crawling & API**
* `Selenium`: 동적 웹 페이지(Steam Category 및 무한 스크롤) 탐색 및 데이터 추출
* `Requests`: Steam Web API 통신 및 정적 데이터 파싱

**Data Processing & Analysis**
* `Pandas`: 수집된 대규모 리뷰 데이터 프레임화, 결측치 처리 및 정제 작업
* `re` (Python Built-in): 정규표현식을 활용한 리뷰 텍스트 내 노이즈(특수기호, 이모지 등) 제거 및 전처리

**Data Storage**
* `JSONL`: 대규모 텍스트 데이터의 메모리 효율적 처리를 위한 직렬화 및 저장 포맷

## 📝 코드 변경 및 업데이트 내역 (Changelog)

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
.
├── main.py                       # 크롤러 실행 메인 엔진
├── lib/
│   ├── SteamAppIDExtractor.py     # 셀레니움 기반 게임 ID 추출 모듈
│   └── SteamMassiveReviewCrawler.py # API 기반 리뷰 수집 모듈
│ 
└── 1st Crawling Data             # 1차 데이터 및 체크포인트 저장폴더
│   ├── steam_pure_reviews_1.jsonl # 1차 수집된 원본 리뷰 데이터셋
│   └── completed_appids_1.txt    # 1차 수집 완료된 게임 목록 (중복 방지용)
│ 
└── 2nd Crawling Data/            # 2차 수정 데이터 및 체크포인트 저장 폴더
│   ├── completed_appids_2.txt       # 2차 수집 완료된 게임 목록 (중복 방지용)
│   └── steam_raw_reviews_2.jsonl    # 2차 수집된 원본 리뷰 데이터셋
│ 
└── 3rd Crawling Data/            # 3차 수정 데이터 및 체크포인트 저장 폴더
│   ├── completed_appids_3.txt       # 3차 수집 완료된 게임 목록 (중복 방지용)
│   └── steam_raw_reviews_3.jsonl    # 3차 수집된 원본 리뷰 데이터셋
│ 
└── data/                           # 4차 수정 데이터 및 체크포인트 저장 폴더 (현재)
│   ├── completed_appids.txt       # 4차 수집 완료된 게임 목록 (중복 방지용)
│   └── steam_raw_reviews.jsonl    # 4차 수집된 원본 리뷰 데이터셋


