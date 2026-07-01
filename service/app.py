import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from service.analytics import load_reviews_with_category, get_category_stats, get_game_stats, get_sample_reviews, get_reviews_by_game, CATEGORY_KR
from service.live_stats import load_all_appids, fetch_top_games

from service.SentimentAnalyzer import SentimentAnalyzer as _Analyzer

st.set_page_config(
    page_title="Steam 리뷰 감성 분석",
    page_icon="🎮",
    layout="wide",
)

# ── 공통 캐시 ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _load_df():
    return load_reviews_with_category()

@st.cache_resource(show_spinner=False)
def _get_filter():
    return _Analyzer()

# ── 사이드바 네비게이션 ────────────────────────────────────
st.sidebar.title("🎮 Steam 리뷰 분석")
page = st.sidebar.radio(
    "메뉴",
    ["📊 카테고리 통계", "🔥 실시간 인기 게임", "✍️ 리뷰 감성 검사"],
)

# ══════════════════════════════════════════════════════════
# 탭 1 : 카테고리 통계
# ══════════════════════════════════════════════════════════
if page == "📊 카테고리 통계":
    st.title("📊 장르별 리뷰 긍정 / 부정 비율")
    st.caption("긍정/부정은 평점이 아니라 **KcELECTRA 모델이 리뷰 본문을 분석한 결과** 기준입니다.")

    with st.spinner("데이터 로딩 중..."):
        df = _load_df()

    cat_stats = get_category_stats(df)

    # 평점 vs 모델 감성 불일치율 (모델 분석 캐시가 있을 때만)
    if 'rating' in df.columns and not (df['rating'] == df['label']).all():
        disagree = int((df['rating'] != df['label']).sum())
        st.info(
            f"💡 전체 {len(df):,}건 중 **{disagree:,}건({disagree/len(df)*100:.1f}%)** 은 "
            f"평점(추천/비추천)과 본문 감성이 다릅니다."
        )

    # ── 상단 지표 (컴팩트) ─────────────────────────────────
    total_reviews = int(df.shape[0])
    total_pos = int(df['label'].sum())
    total_games = df['game'].nunique()

    def _stat(label, value):
        return (
            f"<div style='line-height:1.25'>"
            f"<span style='font-size:0.85rem;color:#8b97a3'>{label}</span><br>"
            f"<span style='font-size:1.55rem;font-weight:600'>{value}</span>"
            f"</div>"
        )

    s1, s2, s3, s4 = st.columns(4)
    s1.markdown(_stat("총 리뷰", f"{total_reviews:,}건"), unsafe_allow_html=True)
    s2.markdown(_stat("총 게임", f"{total_games:,}개"), unsafe_allow_html=True)
    s3.markdown(_stat("긍정 리뷰", f"{total_pos:,}건"), unsafe_allow_html=True)
    s4.markdown(_stat("전체 긍정률", f"{total_pos/total_reviews*100:.1f}%"), unsafe_allow_html=True)

    st.divider()

    # ── 장르 선택 필터 ─────────────────────────────────────
    # '기타'(미매핑 게임)가 압도적으로 많아 기본값에서 제외하면 다른 장르가 잘 보입니다.
    all_cats_kr = cat_stats['category_kr'].tolist()
    default_cats = [c for c in all_cats_kr if c != '기타'] or all_cats_kr

    fcol1, fcol2 = st.columns([4, 1])
    with fcol1:
        selected_cats_kr = st.multiselect(
            "표시할 장르 선택",
            options=all_cats_kr,
            default=default_cats,
            help="보고 싶은 장르만 골라서 비교할 수 있습니다. '기타'는 미분류 게임이라 기본 제외됩니다.",
        )
    with fcol2:
        st.write("")
        st.write("")
        if st.button("전체 선택", use_container_width=True):
            selected_cats_kr = all_cats_kr

    if not selected_cats_kr:
        st.info("장르를 1개 이상 선택하세요.")
        st.stop()

    cat_view = cat_stats[cat_stats['category_kr'].isin(selected_cats_kr)].reset_index(drop=True)

    # ── 차트 ──────────────────────────────────────────────
    col_left, col_right = st.columns([3, 2])

    with col_left:
        view_mode = st.radio(
            "보기 방식",
            ["로그 스케일", "실제 개수", "긍정/부정 비율(%)"],
            horizontal=True,
            help="FPS처럼 리뷰가 많은 장르 때문에 작은 장르가 안 보일 때 로그 스케일이나 비율로 보세요.",
        )

        if view_mode == "긍정/부정 비율(%)":
            bar_data = cat_view.copy()
            bar_data['positive'] = (bar_data['positive'] / bar_data['total'] * 100).round(1)
            bar_data['negative'] = (bar_data['negative'] / bar_data['total'] * 100).round(1)
            barmode, y_label = 'stack', '비율(%)'
        else:
            bar_data, barmode, y_label = cat_view, 'group', '리뷰 수'

        fig_bar = px.bar(
            bar_data,
            x='category_kr',
            y=['positive', 'negative'],
            title="장르별 리뷰 (긍정 vs 부정) — 막대를 클릭하면 리뷰를 볼 수 있어요",
            labels={'value': y_label, 'variable': '종류', 'category_kr': '장르'},
            color_discrete_map={'positive': '#4CAF50', 'negative': '#F44336'},
            barmode=barmode,
        )
        if view_mode == "로그 스케일":
            fig_bar.update_yaxes(type='log')
        fig_bar.update_layout(legend_title_text='')
        # on_select="rerun" → 막대 클릭 시 선택 정보를 받아 리뷰 표시
        bar_event = st.plotly_chart(
            fig_bar, use_container_width=True,
            on_select="rerun", key="cat_bar",
        )

    with col_right:
        fig_pie = px.pie(
            cat_view,
            names='category_kr',
            values='total',
            title="장르별 리뷰 분포",
            hole=0.4,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # ── 막대 클릭 → 해당 장르/감성의 실제 리뷰 표시 ──────────
    points = (bar_event.get("selection", {}) or {}).get("points", []) if bar_event else []
    if points:
        p = points[0]
        sel_cat_kr = p.get("x")
        # 클릭한 막대가 긍정(positive) 트레이스인지 부정인지 판별
        legend = (p.get("legendgroup") or "").lower()
        curve = p.get("curve_number")
        if "positive" in legend:
            sel_sent, sent_label = 1, "긍정"
        elif "negative" in legend:
            sel_sent, sent_label = 0, "부정"
        elif curve in (0, 1):
            sel_sent, sent_label = (1, "긍정") if curve == 0 else (0, "부정")
        else:
            sel_sent, sent_label = None, "전체"

        # category_kr → category(영문 키) 역변환
        kr_to_key = {v: k for k, v in CATEGORY_KR.items()}
        sel_cat = kr_to_key.get(sel_cat_kr, sel_cat_kr)

        st.divider()
        icon = "🟢" if sel_sent == 1 else ("🔴" if sel_sent == 0 else "📋")
        st.subheader(f"{icon} '{sel_cat_kr}' 장르의 {sent_label} 리뷰 샘플")

        n_show = st.slider("표시할 리뷰 수", 5, 50, 15, key="sample_n")
        samples = get_sample_reviews(df, sel_cat, sentiment=sel_sent, n=n_show)
        if samples.empty:
            st.info("표시할 리뷰가 없습니다.")
        else:
            show = samples.copy()
            show['평점'] = show['label'].map({1: '👍 추천', 0: '👎 비추천'})
            show = show[['game', '평점', 'review']]
            show.columns = ['게임', '평점', '리뷰 내용']
            st.dataframe(show, use_container_width=True, hide_index=True)
    else:
        st.caption("💡 위 막대 그래프에서 장르(긍정/부정) 막대를 클릭하면 해당 리뷰가 여기에 표시됩니다.")

    # ── 긍정률 랭킹 표 ────────────────────────────────────
    st.subheader("장르별 긍정률 순위")
    display = cat_view[['category_kr', 'total', 'positive', 'negative', 'positive_rate']].copy()
    display.columns = ['장르', '총 리뷰', '긍정', '부정', '긍정률(%)']
    st.dataframe(display, use_container_width=True, hide_index=True)

    st.divider()

    # ── 게임별 상세 ────────────────────────────────────────
    st.subheader("🎯 게임별 상세 통계")
    cat_options = ['전체'] + sorted(df['category'].unique().tolist())
    selected_cat = st.selectbox(
        "장르 선택",
        cat_options,
        format_func=lambda x: CATEGORY_KR.get(x, x) if x != '전체' else '전체',
    )
    top_n = st.slider("상위 N개 게임", 5, 50, 20)

    game_stats = get_game_stats(df, selected_cat).head(top_n)
    cat_label = '전체' if selected_cat == '전체' else CATEGORY_KR.get(selected_cat, selected_cat)

    fig_h = px.bar(
        game_stats,
        x='positive_rate',
        y='game',
        orientation='h',
        color='positive_rate',
        color_continuous_scale='RdYlGn',
        range_color=[0, 100],
        title=f"{cat_label} — 게임별 긍정률 TOP {top_n}",
        labels={'positive_rate': '긍정률(%)', 'game': '게임'},
    )
    fig_h.update_layout(height=max(400, top_n * 28), yaxis={'categoryorder': 'total ascending'})
    fig_h.update_layout(title=f"{cat_label} — 게임별 긍정률 TOP {top_n} (막대를 클릭하면 리뷰 표시)")
    game_event = st.plotly_chart(
        fig_h, use_container_width=True,
        on_select="rerun", key="game_bar",
    )

    # ── 게임 막대 클릭 → 해당 게임의 리뷰(본문/평점/모델판단) ───
    g_points = (game_event.get("selection", {}) or {}).get("points", []) if game_event else []
    if g_points:
        sel_game = g_points[0].get("y")  # 가로 막대 → 게임명은 y축
        st.divider()
        st.subheader(f"🎮 '{sel_game}' 리뷰")
        st.caption("평점은 작성자가 누른 추천/비추천, '모델 판단'은 KcELECTRA가 본문을 분석한 결과입니다.")
        g_n = st.slider("표시할 리뷰 수", 5, 50, 20, key="game_sample_n")
        greviews = get_reviews_by_game(df, sel_game, n=g_n)
        if greviews.empty:
            st.info("표시할 리뷰가 없습니다.")
        else:
            gshow = greviews.copy()
            if 'rating' in gshow.columns:
                gshow['평점'] = gshow['rating'].map({1: '👍 추천', 0: '👎 비추천'})
            gshow['모델 판단'] = gshow['label'].map({1: '🟢 긍정', 0: '🔴 부정'})
            ordered = (['review', '평점', '모델 판단'] if 'rating' in greviews.columns
                       else ['review', '모델 판단'])
            gshow = gshow[ordered].rename(columns={'review': '리뷰 내용'})
            st.dataframe(gshow, use_container_width=True, hide_index=True)
            # 평점과 모델 판단이 다른 리뷰 강조
            if 'rating' in greviews.columns:
                diff = int((greviews['rating'] != greviews['label']).sum())
                if diff:
                    st.caption(f"↳ 이 중 {diff}건은 평점과 모델 판단이 다릅니다.")
    else:
        st.caption("💡 위 게임별 막대를 클릭하면 해당 게임의 리뷰(본문·평점·모델 판단)가 표시됩니다.")


# ══════════════════════════════════════════════════════════
# 탭 2 : 실시간 인기 게임
# ══════════════════════════════════════════════════════════
elif page == "🔥 실시간 인기 게임":
    st.title("🔥 실시간 동시접속자 TOP 게임의 리뷰 감성")
    st.info(
        "Steam API로 실시간 동시접속자 수를 조회한 뒤, "
        "수집된 리뷰 데이터의 긍정률과 함께 보여줍니다."
    )

    with st.spinner("리뷰 데이터 로딩 중..."):
        df = _load_df()
    game_stats = get_game_stats(df)

    all_appids = load_all_appids()
    all_cats = list(all_appids.keys())

    selected_cats = st.multiselect(
        "조회할 장르 선택",
        options=all_cats,
        default=all_cats,
        format_func=lambda x: CATEGORY_KR.get(x, x),
    )
    top_n = st.number_input("상위 게임 수", min_value=5, max_value=50, value=10, step=5)

    if st.button("🔄 실시간 데이터 조회", type="primary"):
        if not selected_cats:
            st.warning("장르를 1개 이상 선택하세요.")
        else:
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            def _cb(i, total, title):
                progress_bar.progress((i + 1) / total)
                status_text.text(f"조회 중 ({i+1}/{total}): {title}")

            with st.spinner("Steam 서버에서 데이터를 가져오는 중..."):
                results = fetch_top_games(selected_cats, top_n=int(top_n), progress_callback=_cb)

            progress_bar.empty()
            status_text.empty()

            if not results:
                st.error("조회된 게임이 없습니다. 네트워크 연결을 확인하세요.")
            else:
                result_df = pd.DataFrame(results)

                # 리뷰 감성 데이터 병합
                merged = result_df.merge(
                    game_stats[['game', 'total', 'positive_rate']],
                    left_on='title', right_on='game', how='left',
                )
                merged['positive_rate'] = merged['positive_rate'].fillna(-1)

                st.subheader(f"🏆 동시접속자 TOP {len(results)}")

                for rank, row in enumerate(merged.itertuples(), 1):
                    with st.container():
                        col1, col2, col3, col4 = st.columns([4, 2, 1, 2])
                        with col1:
                            st.markdown(f"**#{rank} {row.title}**")
                        with col2:
                            st.metric("동시접속자", f"{int(row.player_count):,}명")
                        with col3:
                            st.markdown(f"`{row.category_kr}`")
                        with col4:
                            if row.positive_rate >= 0:
                                icon = "🟢" if row.positive_rate >= 70 else ("🟡" if row.positive_rate >= 50 else "🔴")
                                st.markdown(f"{icon} 긍정률 **{row.positive_rate:.1f}%**")
                            else:
                                st.markdown("📊 리뷰 데이터 없음")
                    st.divider()

                # 산점도: 접속자 vs 긍정률
                plot_df = merged[merged['positive_rate'] >= 0].copy()
                if not plot_df.empty:
                    fig_scatter = px.scatter(
                        plot_df,
                        x='player_count',
                        y='positive_rate',
                        text='title',
                        color='category_kr',
                        size='total',
                        title="동시접속자 수 vs 리뷰 긍정률",
                        labels={'player_count': '동시접속자 수', 'positive_rate': '긍정률(%)', 'category_kr': '장르'},
                    )
                    fig_scatter.update_traces(textposition='top center')
                    st.plotly_chart(fig_scatter, use_container_width=True)


# ══════════════════════════════════════════════════════════
# 탭 3 : 리뷰 감성 검사
# ══════════════════════════════════════════════════════════
elif page == "✍️ 리뷰 감성 검사":
    st.title("✍️ 리뷰 감성 분석기")
    st.write("작성한 리뷰가 긍정적인지 부정적인지 AI가 분석합니다.")

    try:
        review_filter = _get_filter()
        model_ready = True
        st.caption("모델: KcELECTRA (beomi/kcelectra-base-v2022 파인튜닝)")
    except FileNotFoundError as e:
        st.error(f"⚠️ {e}")
        model_ready = False

    if model_ready:
        review_text = st.text_area(
            "리뷰를 입력하세요",
            height=160,
            placeholder="게임 리뷰를 자유롭게 작성해보세요...",
        )
        threshold = st.slider("부정 감지 임계값 (%)", 50, 95, 70, help="이 값 이상이면 부정 리뷰로 판정합니다.")

        if st.button("🔍 분석하기", type="primary"):
            if not review_text.strip():
                st.warning("리뷰 내용을 입력해주세요.")
            else:
                result = review_filter.analyze(review_text)
                pos_prob = result['pos_prob']
                neg_prob = result['neg_prob']
                is_neg, _ = review_filter.is_too_negative(review_text, threshold / 100)

                # 지표 카드
                c1, c2 = st.columns(2)
                c1.metric("긍정 확률", f"{pos_prob*100:.1f}%")
                c2.metric("부정 확률", f"{neg_prob*100:.1f}%")

                # 게이지 차트
                bar_color = "#4CAF50" if pos_prob >= 0.5 else "#F44336"
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=pos_prob * 100,
                    number={'suffix': '%'},
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "감성 지수 (높을수록 긍정)"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': bar_color},
                        'steps': [
                            {'range': [0, 30], 'color': '#ffcdd2'},
                            {'range': [30, 70], 'color': '#fff9c4'},
                            {'range': [70, 100], 'color': '#c8e6c9'},
                        ],
                        'threshold': {
                            'line': {'color': 'red', 'width': 3},
                            'thickness': 0.75,
                            'value': 100 - threshold,
                        },
                    },
                ))
                st.plotly_chart(fig_gauge, use_container_width=True)

                # 판정 결과
                if is_neg:
                    st.error(
                        f"🚫 **이 리뷰는 부정적인 내용이 많아 등록이 제한됩니다.**\n\n"
                        f"부정 점수: {neg_prob*100:.1f}%  (기준: {threshold}%)"
                    )
                    st.info("💡 더 건설적인 피드백으로 수정하시면 게시할 수 있습니다.")
                else:
                    st.success(f"✅ **리뷰가 정상적으로 등록 가능합니다!** (감성 판정: {result['label']})")
