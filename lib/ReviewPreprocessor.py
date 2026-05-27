import pandas as pd
import re
import os

class ReviewPreprocessor:
    """수집된 JSONL 리뷰 데이터를 Pandas로 전처리하여 CSV로 저장하는 클래스"""
    def __init__(self, input_file, output_file="./data/steam_processed_reviews.csv"):
        self.input_file = input_file
        self.output_file = output_file

    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
        
        # \r, \n 등 제어문자를 공백으로 치환
        text = re.sub(r'[\r\n\xa0]', ' ', text)
        # 한글, 영문, 공백만 남기고 모두 제거 (숫자, 특수기호 제거)
        text = re.sub(r'[^가-힣ㄱ-ㅎㅏ-ㅣa-zA-Z\s]', '', text)
        # 다중 공백을 하나의 공백으로 치환
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

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

        # 3. 텍스트 정제 적용
        df['clean_review'] = df['리뷰 내용'].apply(self.clean_text)

        # 4. 결측치 및 빈 문자열 처리
        df['clean_review'] = df['clean_review'].replace('', pd.NA)
        df.dropna(subset=['clean_review'], inplace=True)

        # 5. 중복 리뷰 제거
        df.drop_duplicates(subset=['게임 이름', '작성자', 'clean_review'], keep='first', inplace=True)
        
        # 6. 결과 저장 (BOM이 포함된 UTF-8로 저장하여 엑셀에서 한글 안 깨지게 처리)
        df.to_csv(self.output_file, index=False, encoding='utf-8-sig')
        
        print(f"  - 전처리 완료: 최종 {len(df)}건 (결측치/중복 제거됨)")
        print(f"[{self.__class__.__name__}] 학습용 데이터가 저장되었습니다: {self.output_file}\n")
        
        return df