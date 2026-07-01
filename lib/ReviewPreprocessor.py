import pandas as pd
import re
import os
from konlpy.tag import Okt

class ReviewPreprocessor:
    """수집된 JSONL 리뷰 데이터를 정제 → 토큰화 → CSV 저장하는 클래스"""
    def __init__(self, input_file, output_file="./data/steam_processed_reviews.csv"):
        self.input_file = input_file
        self.output_file = output_file
        self._okt = None

    @property
    def okt(self):
        if self._okt is None:
            print("  - KoNLPy Okt 초기화 중...")
            self._okt = Okt()
        return self._okt

    def clean_text(self, text):
        """특수문자 제거, 공백 정규화"""
        if not isinstance(text, str):
            return ""
        text = re.sub(r'[\r\n\xa0]', ' ', text)
        text = re.sub(r'[^가-힣ㄱ-ㅎㅏ-ㅣa-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def tokenize(self, text):
        """Okt 형태소 분석 → 명사/동사/형용사/부사만 추출하여 공백 결합"""
        if not text:
            return ""
        keep_pos = {'Noun', 'Verb', 'Adjective', 'Adverb'}
        tokens = [word for word, pos in self.okt.pos(text, norm=True, stem=True)
                  if pos in keep_pos and len(word) > 1]
        return ' '.join(tokens)

    def run_pipeline(self):
        if not os.path.exists(self.input_file):
            print(f"[오류] 전처리할 데이터가 없습니다: {self.input_file}")
            return None

        print(f"\n[{self.__class__.__name__}] 데이터 전처리를 시작합니다...")

        # 1. JSONL 파일 읽기
        df = pd.read_json(self.input_file, lines=True)
        print(f"  - 원본 데이터 로드 완료: 총 {len(df)}건")

        # 2. 라벨링 (추천:1, 비추천:0)
        df['label'] = df['평점'].map({'추천': 1, '비추천': 0})

        # 3. 텍스트 정제
        df['clean_review'] = df['리뷰 내용'].apply(self.clean_text)

        # 4. 결측치 및 빈 문자열 제거
        df['clean_review'] = df['clean_review'].replace('', pd.NA)
        df.dropna(subset=['clean_review'], inplace=True)

        # 5. 중복 리뷰 제거
        df.drop_duplicates(subset=['게임 이름', '작성자', 'clean_review'], keep='first', inplace=True)
        print(f"  - 정제 완료: {len(df)}건 (결측치/중복 제거)")

        # 6. 형태소 토큰화 (핵심 품사만 추출)
        print(f"  - 형태소 토큰화 중... (시간이 걸릴 수 있습니다)")
        df['tokenized_review'] = df['clean_review'].apply(self.tokenize)

        # 토큰화 후 빈 결과 제거
        df['tokenized_review'] = df['tokenized_review'].replace('', pd.NA)
        df.dropna(subset=['tokenized_review'], inplace=True)

        # 7. 결과 저장
        df.to_csv(self.output_file, index=False, encoding='utf-8-sig')

        print(f"  - 토큰화 완료: 최종 {len(df)}건")
        print(f"[{self.__class__.__name__}] 학습용 데이터가 저장되었습니다: {self.output_file}\n")

        return df