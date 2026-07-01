import json
import glob
import os
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_FILE = os.path.join(BASE_DIR, "data", "steam_raw_reviews.jsonl")
# KcELECTRA 모델이 본문 감성을 미리 분석한 캐시 (predict_all_sentiments.py 산출물)
SCORED_FILE = os.path.join(BASE_DIR, "data", "steam_raw_reviews_scored.jsonl")
CATEGORY_MAP_FILE = os.path.join(BASE_DIR, "data", "game_category_map.json")

CATEGORY_KR = {
    'action_fps': 'FPS',
    'action_tps': 'TPS',
    'hack_and_slash': '핵앤슬래시',
    'arcade_rhythm': '리듬/아케이드',
    'action_run_jump': '런앤점프',
    'shmup': '슈팅게임',
    'fighting_martial_arts': '격투게임',
    '기타': '기타',
}


def load_category_map():
    """게임 이름 → 카테고리 매핑 (build_category_map.py 생성 파일 사용)"""
    if not os.path.exists(CATEGORY_MAP_FILE):
        return {}
    with open(CATEGORY_MAP_FILE, encoding='utf-8') as f:
        return json.load(f)


def load_reviews_with_category():
    """리뷰 전체를 카테고리 정보 + 리뷰 본문과 함께 DataFrame으로 반환.

    모델 분석 캐시(SCORED_FILE)가 있으면 'label'을 KcELECTRA가 분석한 본문 감성으로 사용.
    (없으면 원본 평점 추천/비추천을 사용 — 폴백)
    """
    game_category = load_category_map()
    use_scored = os.path.exists(SCORED_FILE)
    src = SCORED_FILE if use_scored else DATA_FILE

    rows = []
    with open(src, encoding='utf-8') as f:
        for line in f:
            try:
                d = json.loads(line)
                if use_scored:
                    game = d.get('game', '')
                    label = d.get('model_label', 0)        # 모델이 분석한 본문 감성
                    rating = d.get('rating', 0)            # 원본 평점 (비교용)
                    review = d.get('review', '')
                else:
                    game = d.get('게임 이름', '')
                    label = 1 if d.get('평점') == '추천' else 0
                    rating = label
                    review = d.get('리뷰 내용', '')
                category = game_category.get(game, '기타')
                rows.append({
                    'game': game,
                    'label': label,
                    'rating': rating,
                    'category': category,
                    'review': review,
                })
            except:
                pass
    return pd.DataFrame(rows)


def get_sample_reviews(df, category, sentiment=None, n=15):
    """특정 카테고리(+선택적 긍/부정)의 리뷰 샘플을 반환.

    sentiment: 1=긍정, 0=부정, None=전체
    """
    sub = df[df['category'] == category]
    if sentiment is not None:
        sub = sub[sub['label'] == sentiment]
    if sub.empty:
        return sub
    # 너무 짧은 리뷰는 제외하고 다양하게 보이도록 샘플링
    sub = sub[sub['review'].str.len() >= 5]
    if sub.empty:
        return sub
    take = min(n, len(sub))
    return sub.sample(take, random_state=None)[['game', 'label', 'review']]


def get_reviews_by_game(df, game, n=20):
    """특정 게임의 리뷰 샘플 반환 (리뷰 본문 + 평점 + 모델 판단)."""
    sub = df[df['game'] == game]
    sub = sub[sub['review'].str.len() >= 5]
    if sub.empty:
        return sub
    take = min(n, len(sub))
    cols = ['review', 'rating', 'label'] if 'rating' in sub.columns else ['review', 'label']
    return sub.sample(take, random_state=None)[cols]


def get_category_stats(df):
    """카테고리별 긍정/부정 집계"""
    stats = (
        df.groupby('category')
        .agg(total=('label', 'count'), positive=('label', 'sum'))
        .reset_index()
    )
    stats['negative'] = stats['total'] - stats['positive']
    stats['positive_rate'] = (stats['positive'] / stats['total'] * 100).round(1)
    stats['category_kr'] = stats['category'].map(lambda x: CATEGORY_KR.get(x, x))
    return stats.sort_values('total', ascending=False).reset_index(drop=True)


def get_game_stats(df, category=None):
    """게임별 긍정/부정 집계 (카테고리 필터 가능)"""
    if category and category != '전체':
        df = df[df['category'] == category]
    stats = (
        df.groupby(['game', 'category'])
        .agg(total=('label', 'count'), positive=('label', 'sum'))
        .reset_index()
    )
    stats['negative'] = stats['total'] - stats['positive']
    stats['positive_rate'] = (stats['positive'] / stats['total'] * 100).round(1)
    stats['category_kr'] = stats['category'].map(lambda x: CATEGORY_KR.get(x, x))
    return stats.sort_values('total', ascending=False).reset_index(drop=True)
