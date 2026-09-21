"""이안류 조난자 긴급 탐지 알림 시스템 — 표준 운영 콘솔 버전.

    streamlit run app.py
"""

from __future__ import annotations

import streamlit as st
from views import video_analysis

st.set_page_config(
    page_title="이안류 조난자 AI 관제 콘솔",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ⭐️ 검증된 최적의 관제 파라미터로 고정 (설정 UI 제거)
SURVEILLANCE_CONFIG = {
    "frame_skip": 2,         # 연산 부하 절감 및 쾌적한 재생 속도 유지
    "max_frames": 400,       # 시연 영상 분석 분량
    "show_foot": True,       # 수면 접촉점(침범 판정) 기준점 표시
    "swimmer_conf": 0.05,    # 파도 속 수영객 검출 최적값
    "rip_conf": 0.25,        # 이안류 구역 판별 최적값
    "rip_model_path": "best_yolo11m_integrated_v2_first.pt",
    "swimmer_model_path": "best_swimmer_yolo11m_b8.pt",
    "dev_mode": False,       # 내부 디버그 플래그 비활성화
}

_HEADER_CSS = """
<style>
header[data-testid="stHeader"] { background: transparent; }
div[data-testid="stMainBlockContainer"] { padding-top: 4.7rem; padding-bottom: 1rem; }
div[data-testid="stSidebarUserContent"] { padding-top: 0.0rem; }
section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] { gap: .5rem; }
section[data-testid="stSidebar"] hr { margin: .6rem 0; }
div[data-testid="stMainBlockContainer"] div[data-testid="stVerticalBlock"] { gap: 0; }
div[data-testid="stMainBlockContainer"] h4 { margin: 0 0 .25rem 0; padding: 0; }
div[data-testid="stImage"] img { width: 100%; height: auto; display: block; border-radius: 0; }
div[data-testid="stHorizontalBlock"] { align-items: flex-start; }
</style>
"""


def main() -> None:
    st.markdown(_HEADER_CSS, unsafe_allow_html=True)
    st.session_state.setdefault("uploaded_video", None)
    st.session_state.setdefault("run_analysis", False)

    uploaded, start_clicked = sidebar()

    # 업로드가 없으면 기본 테스트 영상으로 분석 뷰를 표시
    video_analysis.render(uploaded, SURVEILLANCE_CONFIG, start_clicked)


def sidebar() -> tuple[any, bool]:
    st.sidebar.markdown(
        "<div style='font-size:.82rem;opacity:.6;margin-bottom:2px'>YOLO11m + SAHI 듀얼 엔진 가동</div>",
        unsafe_allow_html=True,
    )
    st.sidebar.caption("⚡ 실시간 정밀 파이프라인 (Conf 0.05 / Frame Skip 2)")

    uploaded = st.sidebar.file_uploader(
        "테스트 영상 업로드",
        type=["mp4", "avi", "mov", "mkv"],
        help="영상을 올리면 실시간 AI 조난 분석 준비가 완료됩니다.",
    )
    st.session_state["uploaded_video"] = uploaded

    # 좌측 테스트 영상 업로드 영역 최하단에 시작 버튼 배치
    start_clicked = st.sidebar.button(
        "🚀 실시간 관제 시작",
        type="primary",
        use_container_width=True,
    )

    st.sidebar.markdown(
        "<div style='font-size:.82rem;line-height:1.6;color:#888'>"
        "• <b>이안류 판별:</b> YOLO11m (Conf 0.25)<br>"
        "• <b>수영객 탐지:</b> SAHI 640px (Conf 0.05)<br>"
        "• <b>분석 간격:</b> 2 프레임 (가속 파이프라인)<br>"
        "• <b>판정 기준:</b> 수면 접점 Point-in-Polygon"
        "</div>",
        unsafe_allow_html=True,
    )

    return uploaded, start_clicked


if __name__ == "__main__":
    main()