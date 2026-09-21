"""이안류 조난자 긴급 탐지 알림 시스템 — 화면(UI) 전용 버전.

    streamlit run app.py

원본 앱에서 모델 추론과 알림 발송을 걷어내고 화면 코드만 남긴 것이다.
  - 모델(ultralytics / torch)을 부르지 않는다 → 가중치 파일도, GPU도 필요 없다
  - 어디로도 메시지를 보내지 않는다 (FCM / 콘솔 알림 모두 없음)
  - DB(SQLite)를 쓰지 않는다 → 이벤트는 브라우저 세션에만 남는다

기본 화면은 가상 해변 A~E의 상시 관제 화면이다.
사이드바에서 영상을 올리면 화면 전반이 재생 화면으로 바뀌고,
좌측 상단 링크를 누르면 관제 화면으로 돌아온다.
"""

from __future__ import annotations

import streamlit as st

from views import dashboard, test_run

st.set_page_config(
    page_title="이안류 조난자 탐지 (UI)",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",   # 접힌 채로 시작하지 않게 고정
)

UI_MODE_LABEL = {"user": "사용자 모드", "dev": "개발자 모드"}

# 개발자 모드에서만 조절하는 값들. 사용자 모드에서도 이 값이 그대로 쓰이므로,
# 모드를 오갈 때 바꾼 값이 초기화되지 않도록 session_state 에 담아 둔다.
# 전부 '화면이 어떻게 보이는가' 에만 영향을 준다.
DEV_DEFAULTS = {
    # 영상에서 N프레임마다 한 장씩 화면에 띄운다. 키우면 빨리 넘어간다.
    "frame_skip": 2,
    # 한 번 재생에서 띄울 최대 장수. 긴 영상이 끝없이 도는 것을 막는다.
    "max_frames": 200,
    # 구역 침범 판정에 쓰는 기준점(발끝)을 화면에 점으로 찍는다.
    "show_foot": False,
}

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


def main() -> None:
    st.markdown(_HEADER_CSS, unsafe_allow_html=True)
    st.session_state.setdefault("uploaded_video", None)
    st.session_state.setdefault("ui_mode", "user")
    st.session_state.setdefault("dev_settings", dict(DEV_DEFAULTS))
    # 코드를 고치고 새로고침했을 때 브라우저 세션에 남은 옛 설정 dict 에는
    # 새로 생긴 키가 없다. 빠진 것만 기본값으로 채운다 (KeyError 방지).
    for key, value in DEV_DEFAULTS.items():
        st.session_state["dev_settings"].setdefault(key, value)

    settings = sidebar()
    header()

    uploaded = st.session_state.get("uploaded_video")
    if uploaded is None:
        dashboard.render()
    else:
        test_run.render(uploaded, settings)


def header() -> None:
    """테스트 영상을 보는 중일 때만 뜨는 복귀 링크.

    상시 관제 화면에서는 이 줄을 아예 그리지 않는다.
    메인 콘텐츠가 스크롤 없이 다 보여야 하기 때문이다.
    """
    if st.session_state.get("uploaded_video") is None:
        return
    if st.button("← 상시 관제 화면으로", key="home_link", type="tertiary"):
        st.session_state["uploaded_video"] = None
        st.rerun()


def _settings_popover() -> None:
    """웹 전체 설정. 사이드바 맨 위에 둔다."""
    with st.sidebar.popover("⚙️ 설정", width="stretch"):
        st.markdown("**화면 모드**")
        st.radio(
            "화면 모드",
            list(UI_MODE_LABEL),
            key="ui_mode",
            format_func=UI_MODE_LABEL.get,
            label_visibility="collapsed",
            help=(
                "사용자 모드: 영상을 올려 화면을 확인하는 데 필요한 것만 보입니다. "
                "개발자 모드: 재생 속도와 표시 옵션이 사이드바에 나타납니다."
            ),
        )
        if st.session_state["ui_mode"] == "dev":
            if st.button("설정값 기본으로 되돌리기", width="stretch"):
                st.session_state["dev_settings"] = dict(DEV_DEFAULTS)
                st.rerun()
        else:
            st.caption("개발자 모드로 바꾸면 표시 설정이 사이드바에 나타납니다.")


def sidebar() -> dict:
    dev = st.session_state["ui_mode"] == "dev"
    cfg = st.session_state["dev_settings"]

    # 프로젝트명은 사이드바에 둔다. 메인 화면의 세로 공간을 쓰지 않기 위해서다.
    st.sidebar.markdown(
        "<div style='font-size:1.02rem;font-weight:700;line-height:1.35'>"
        "🌊 이안류 조난자 긴급 탐지 알림</div>"
        "<div style='font-size:.78rem;opacity:.6;margin-bottom:10px'>팀 비버즈 · 화면(UI) 버전</div>",
        unsafe_allow_html=True,
    )

    _settings_popover()

    # 모델도 알림도 없는 버전임을 화면에 항상 띄운다.
    # 이걸 보지 못하면 표시된 구역·사람을 실제 탐지 결과로 오해할 수 있다.
    st.sidebar.caption("모델 추론 · 알림 발송 없음 — 화면 확인용 버전입니다.")

    uploaded = st.sidebar.file_uploader(
        "테스트 영상",
        type=["mp4", "avi", "mov", "mkv"],
        help="영상을 올리면 관제 화면 대신 재생 화면이 표시됩니다.",
    )
    st.session_state["uploaded_video"] = uploaded

    if dev:
        with st.sidebar.expander("표시 설정", expanded=True):
            cfg["frame_skip"] = st.slider(
                "프레임 간격", 1, 30, cfg["frame_skip"],
                help="영상에서 N프레임마다 한 장씩 화면에 띄웁니다. 키우면 빨리 넘어갑니다.",
            )
            cfg["max_frames"] = int(
                st.number_input("최대 표시 장수", 10, 2000, cfg["max_frames"], step=10,
                                help="한 번 재생에서 띄울 최대 장수입니다.")
            )
            cfg["show_foot"] = st.checkbox(
                "발끝 기준점 표시", value=bool(cfg["show_foot"]),
                help="구역 침범 판정에 쓰는 점(박스 아래쪽 중앙)을 화면에 찍습니다.",
            )

    return {
        "frame_skip": cfg["frame_skip"],
        "max_frames": cfg["max_frames"],
        "show_foot": cfg["show_foot"],
        "dev_mode": dev,
    }


if __name__ == "__main__":
    main()
