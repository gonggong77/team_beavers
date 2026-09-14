"""이안류 조난자 긴급 탐지 알림 시스템.

    streamlit run app.py

기본 화면은 가상 해변 A~E의 상시 관제 화면이다.
사이드바에서 영상을 올리면 화면 전반이 분석 결과로 바뀌고,
좌측 상단 제목을 누르면 관제 화면으로 돌아온다.
"""

from __future__ import annotations

import streamlit as st

from core.detector import (
    DEFAULT_CONF,
    DEFAULT_IMGSZ,
    DEFAULT_IOU,
    build_detector,
    resolve_weights,
)
from core.model_probe import probe
from core.store import EventStore
from views import dashboard, test_run

st.set_page_config(
    page_title="이안류 조난자 탐지",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",   # 접힌 채로 시작하지 않게 고정
)

# 모델의 names 로 클래스를 자동 매핑하지 못했을 때 쓸 기본값
FALLBACK_RIP_IDS = [0]
FALLBACK_PERSON_IDS = [1]

_HEADER_CSS = """
<style>
/* 상단 툴바는 내용 위에 겹쳐 뜬다. 배경만 투명하게 한다.
   header 나 toolbar 를 숨기면 Streamlit 버전에 따라 사이드바를 다시 펼치는
   버튼까지 같이 사라져서 복구가 불가능해진다. 절대 숨기지 말 것. */
header[data-testid="stHeader"] { background: transparent; }

/* 사이드바 펼치기 버튼은 어떤 경우에도 보이게 강제한다 */
div[data-testid="stSidebarCollapsedControl"],
div[data-testid="collapsedControl"] {
    display: flex !important; visibility: visible !important; opacity: 1 !important;
    z-index: 999999 !important;
}

/* 메인 화면이 스크롤 없이 한 눈에 들어오도록 여백을 줄인다.
   툴바가 겹치지 않을 만큼만 남긴다. */
div[data-testid="stMainBlockContainer"] { padding-top: 2.4rem; padding-bottom: 1rem; }

/* 사이드바 항목이 세로로 늘어지지 않게 간격을 좁힌다 */
section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] { gap: .4rem; }
section[data-testid="stSidebar"] hr { margin: .5rem 0; }
div[data-testid="stMainBlockContainer"] div[data-testid="stVerticalBlock"] { gap: .55rem; }
div[data-testid="stMainBlockContainer"] h4 { margin: 0 0 .25rem 0; padding: 0; }

/* 영상은 열 너비를 꽉 채운다. 높이를 제한하면 가로가 같이 줄어 작아진다 */
div[data-testid="stImage"] img { width: 100%; height: auto; display: block; }

/* 상시 관제로 돌아가는 링크 버튼 */
div[data-testid="stButton"] button[kind="tertiary"] {
    padding: 0; margin: 0; color: inherit; border: none; background: none;
    font-size: .9rem; font-weight: 600;
}
div[data-testid="stButton"] button[kind="tertiary"]:hover { opacity: .65; }
</style>
"""


@st.cache_resource(show_spinner="모델을 불러오는 중입니다.")
def load_detector(weights_key: str, conf: float, iou: float, imgsz: int):
    """모델은 세션마다 다시 올리면 안 되므로 cache_resource로 전역 공유한다."""
    return build_detector(weights_key or None, conf=conf, iou=iou, imgsz=imgsz)


@st.cache_resource
def get_store() -> EventStore:
    return EventStore()


def main() -> None:
    st.markdown(_HEADER_CSS, unsafe_allow_html=True)
    st.session_state.setdefault("uploaded_video", None)

    settings, detector = sidebar()

    uploaded = st.session_state.get("uploaded_video")
    if uploaded is not None:
        header()

    if uploaded is None:
        dashboard.render(
            mode=settings["dashboard_mode"],
            detector=detector,
            weights_key=settings["weights_key"],
            rip_ids=settings["rip_ids"],
            person_ids=settings["person_ids"],
        )
    else:
        test_run.render(uploaded, detector, get_store(), settings)


def header() -> None:
    """테스트 영상을 보는 중일 때만 뜨는 복귀 링크.

    상시 관제 화면에서는 제목줄을 아예 그리지 않는다.
    메인 콘텐츠가 스크롤 없이 다 보여야 하기 때문이다.
    """
    if st.button("← 상시 관제 화면으로", key="home_link", type="tertiary"):
        st.session_state["uploaded_video"] = None
        st.rerun()


def sidebar() -> tuple[dict, object]:
    # 프로젝트명은 사이드바에 둔다. 메인 화면의 세로 공간을 쓰지 않기 위해서다.
    st.sidebar.markdown(
        "<div style='font-size:1.02rem;font-weight:700;line-height:1.35'>"
        "🌊 이안류 조난자 긴급 탐지 알림</div>"
        "<div style='font-size:.78rem;opacity:.6;margin-bottom:10px'>팀 비버즈</div>",
        unsafe_allow_html=True,
    )

    # 모델 경로와 클래스 매핑은 화면에서 고르지 않는다.
    # models/ 폴더나 .env 의 RIP_MODEL_PATH 에서 자동으로 찾고,
    # 클래스는 모델 파일의 names 를 읽어 이름으로 자동 매핑한다.
    found = resolve_weights()
    weights_key = str(found) if found else ""
    detector, is_real = load_detector(weights_key, DEFAULT_CONF, DEFAULT_IOU, DEFAULT_IMGSZ)
    info = probe(detector)

    rip_ids = info.rip_ids or FALLBACK_RIP_IDS
    person_ids = info.person_ids or FALLBACK_PERSON_IDS

    # 자주 쓰는 것만 밖에 두고 나머지는 접어 둔다. 사이드바가 스크롤되면 조작이 번거롭다.
    uploaded = st.sidebar.file_uploader(
        "테스트 영상",
        type=["mp4", "avi", "mov", "mkv"],
        help="영상을 올리면 관제 화면 대신 분석 결과가 표시됩니다.",
    )
    st.session_state["uploaded_video"] = uploaded

    dashboard_mode = st.sidebar.selectbox(
        "관제 화면 표시 방식",
        ["scenario", "hybrid", "real"],
        format_func=lambda m: {
            "scenario": "시나리오 (모델 없음)",
            "hybrid": "사람만 실제 탐지",
            "real": "전체 실제 추론",
        }[m],
        help=(
            "시나리오: 전부 시연용 구성. "
            "사람만 실제 탐지: 사람은 모델이 찾고 이안류 구역만 구성. "
            "전체 실제 추론: 이안류까지 모델 결과."
        ),
    )
    if dashboard_mode != "scenario" and not is_real:
        st.sidebar.caption("모델 파일이 없어 시나리오 모드로 표시됩니다.")

    with st.sidebar.expander("분석 설정"):
        frame_skip = st.slider(
            "프레임 간격", 1, 30, 5,
            help="N프레임마다 한 번 추론합니다. 값을 키우면 빨라지고 놓치는 순간이 늘어납니다.",
        )
        max_frames = st.number_input("최대 추론 횟수", 10, 5000, 300, step=10)

    with st.sidebar.expander("알림 설정"):
        notifier_kind = st.selectbox(
            "발송 방식", ["console", "telegram"],
            format_func={"console": "화면 기록만", "telegram": "텔레그램"}.get,
        )
        min_consecutive = st.slider("연속 긴급 판정 횟수", 1, 10, 3)
        cooldown_sec = st.slider("재발송 금지 시간(초)", 0, 120, 30, 5)

    settings = {
        "dashboard_mode": dashboard_mode,
        "weights_key": weights_key,
        "rip_ids": rip_ids,
        "person_ids": person_ids,
        "frame_skip": frame_skip,
        "max_frames": int(max_frames),
        "min_consecutive": min_consecutive,
        "cooldown_sec": float(cooldown_sec),
        "notifier_kind": notifier_kind,
    }
    return settings, detector


if __name__ == "__main__":
    main()
