"""이안류 조난자 긴급 탐지 알림 시스템.

    streamlit run app.py

기본 화면은 가상 해변 A~E의 상시 관제 화면이다.
사이드바에서 영상을 올리면 화면 전반이 분석 결과로 바뀌고,
좌측 상단 제목을 누르면 관제 화면으로 돌아온다.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from core.detector import (
    DEFAULT_IMGSZ,
    DEFAULT_IOU,
    DEFAULT_PERSON_CONF,
    DEFAULT_RIP_CONF,
    build_detector,
    resolve_person_weights,
    resolve_rip_weights,
)
from core.model_probe import probe
from core.scenario import MODE_LABEL
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

UI_MODE_LABEL = {"user": "사용자 모드", "dev": "개발자 모드"}

# 개발자 모드에서만 조절하는 값들. 사용자 모드에서도 이 값이 그대로 쓰이므로,
# 모드를 오갈 때 튜닝한 값이 초기화되지 않도록 session_state 에 담아 둔다.
DEV_DEFAULTS = {
    "rip_conf": DEFAULT_RIP_CONF,
    "rip_iou": DEFAULT_IOU,
    "rip_imgsz": DEFAULT_IMGSZ,
    "person_conf": DEFAULT_PERSON_CONF,
    "person_iou": DEFAULT_IOU,
    "person_imgsz": DEFAULT_IMGSZ,
    "frame_skip": 5,
    "max_frames": 300,
    "min_consecutive": 1,
    "cooldown_sec": 5.0,
    "dashboard_mode": "auto",
}

IMGSZ_CHOICES = [320, 416, 512, 640, 768, 960, 1280]

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
def load_detector(
    rip_weights_key: str,
    person_weights_key: str,
    rip_conf: float,
    rip_iou: float,
    rip_imgsz: int,
    person_conf: float,
    person_iou: float,
    person_imgsz: int,
):
    """모델은 세션마다 다시 올리면 안 되므로 cache_resource로 전역 공유한다.

    임계값도 캐시 키에 들어간다. 개발자 모드에서 conf 를 바꾸면 그 조합으로 새로 올라간다.
    """
    return build_detector(
        rip_weights_key or None,
        person_weights_key or None,
        rip_conf=rip_conf,
        rip_iou=rip_iou,
        rip_imgsz=rip_imgsz,
        person_conf=person_conf,
        person_iou=person_iou,
        person_imgsz=person_imgsz,
    )


@st.cache_resource
def get_store() -> EventStore:
    return EventStore()


def main() -> None:
    st.markdown(_HEADER_CSS, unsafe_allow_html=True)
    st.session_state.setdefault("uploaded_video", None)
    st.session_state.setdefault("ui_mode", "user")
    st.session_state.setdefault("dev_settings", dict(DEV_DEFAULTS))

    settings, detector = sidebar()
    header()

    uploaded = st.session_state.get("uploaded_video")
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
                "사용자 모드: 영상을 올려 결과를 확인하는 데 필요한 것만 보입니다. "
                "개발자 모드: 모델 임계값, 분석 성능, 판정 기준까지 사이드바에서 조절합니다."
            ),
        )
        if st.session_state["ui_mode"] == "dev":
            st.caption("모델 임계값은 아래 **모델 설정** 에서 조절합니다.")
            if st.button("설정값 기본으로 되돌리기", width="stretch"):
                st.session_state["dev_settings"] = dict(DEV_DEFAULTS)
                st.rerun()
        else:
            st.caption("개발자 모드로 바꾸면 모델·분석·판정 설정이 사이드바에 나타납니다.")


def _auto_dashboard_mode(rip_is_real: bool, person_is_real: bool) -> str:
    """가진 모델로 가장 정직하게 보여줄 수 있는 관제 화면 모드.

    사람 모델이 없는데 real 로 두면 가짜 사람이 진짜 이안류 구역에 들어가
    실제와 구분되지 않는 긴급 알림이 상시 화면에서 계속 발생한다.
    """
    if rip_is_real and person_is_real:
        return "real"
    if person_is_real:
        return "hybrid"
    return "scenario"


def sidebar() -> tuple[dict, object]:
    dev = st.session_state["ui_mode"] == "dev"
    cfg = st.session_state["dev_settings"]

    # 프로젝트명은 사이드바에 둔다. 메인 화면의 세로 공간을 쓰지 않기 위해서다.
    st.sidebar.markdown(
        "<div style='font-size:1.02rem;font-weight:700;line-height:1.35'>"
        "🌊 이안류 조난자 긴급 탐지 알림</div>"
        "<div style='font-size:.78rem;opacity:.6;margin-bottom:10px'>팀 비버즈</div>",
        unsafe_allow_html=True,
    )

    _settings_popover()

    # 모델 경로는 화면에서 고르지 않는다.
    # 이안류 모델은 models/rip/(또는 .env 의 RIP_MODEL_PATH), 사람 모델은
    # models/person/(또는 .env 의 PERSON_MODEL_PATH) 에서 자동으로 찾는다.
    rip_found = resolve_rip_weights()
    person_found = resolve_person_weights()
    rip_weights_key = str(rip_found) if rip_found else ""
    person_weights_key = str(person_found) if person_found else ""
    detector, rip_is_real, person_is_real = load_detector(
        rip_weights_key,
        person_weights_key,
        cfg["rip_conf"],
        cfg["rip_iou"],
        cfg["rip_imgsz"],
        cfg["person_conf"],
        cfg["person_iou"],
        cfg["person_imgsz"],
    )
    info = probe(detector)

    # 데모용 가짜 사람이 실제 이안류 구역 안에 들어가면 진짜와 똑같은 긴급 알림이 나간다.
    # 어느 쪽이 실제 추론인지 모드와 무관하게 항상 눈에 보여야 한다.
    st.sidebar.caption(
        f"모델 · 이안류 {'실제 추론' if rip_is_real else '데모'}"
        f" / 사람 {'실제 추론' if person_is_real else '데모'}"
    )

    uploaded = st.sidebar.file_uploader(
        "테스트 영상",
        type=["mp4", "avi", "mov", "mkv"],
        help="영상을 올리면 관제 화면 대신 분석 결과가 표시됩니다.",
    )
    st.session_state["uploaded_video"] = uploaded

    notifier_kind = st.sidebar.selectbox(
        "알림 발송 방식", ["fcm", "console"],
        format_func={"console": "화면 기록만", "fcm": "안드로이드 앱(FCM)"}.get,
    )

    auto_mode = _auto_dashboard_mode(rip_is_real, person_is_real)
    dashboard_mode = auto_mode

    if dev:
        _model_settings(cfg, detector, rip_found, person_found, rip_is_real, person_is_real)

        with st.sidebar.expander("분석 설정"):
            cfg["frame_skip"] = st.slider(
                "프레임 간격", 1, 30, cfg["frame_skip"],
                help="N프레임마다 한 번 추론합니다. 값을 키우면 빨라지고 놓치는 순간이 늘어납니다.",
            )
            cfg["max_frames"] = int(
                st.number_input("최대 추론 횟수", 10, 5000, cfg["max_frames"], step=10)
            )

        with st.sidebar.expander("판정·발송 기준"):
            cfg["min_consecutive"] = st.slider("연속 긴급 판정 횟수", 1, 10, cfg["min_consecutive"])
            cfg["cooldown_sec"] = float(
                st.slider("재발송 금지 시간(초)", 0, 120, int(cfg["cooldown_sec"]), 5)
            )

        with st.sidebar.expander("관제 화면 표시 방식"):
            choices = ["auto", "scenario", "hybrid", "real"]
            cfg["dashboard_mode"] = st.radio(
                "표시 방식",
                choices,
                index=choices.index(cfg["dashboard_mode"]),
                format_func=lambda m: {
                    "auto": f"자동 ({MODE_LABEL[auto_mode]})",
                    "scenario": "시나리오 (전부 시연용)",
                    "hybrid": "사람만 실제 탐지",
                    "real": "전체 실제 추론",
                }[m],
                label_visibility="collapsed",
            )
            dashboard_mode = auto_mode if cfg["dashboard_mode"] == "auto" else cfg["dashboard_mode"]

            ready = {
                "scenario": True,
                "hybrid": person_is_real,
                "real": rip_is_real and person_is_real,
            }[dashboard_mode]
            if not ready:
                st.caption("필요한 모델이 없어 시나리오 모드로 표시됩니다.")

    settings = {
        "dashboard_mode": dashboard_mode,
        # 임계값이 바뀌면 관제 화면의 사전 추론 결과도 다시 만들어야 한다.
        # 이 키가 그대로면 conf 를 바꿔도 예전 결과가 그대로 보인다.
        "weights_key": (
            f"{rip_weights_key}@{cfg['rip_conf']}/{cfg['rip_iou']}/{cfg['rip_imgsz']}"
            f"|{person_weights_key}@{cfg['person_conf']}/{cfg['person_iou']}/{cfg['person_imgsz']}"
        ),
        "rip_ids": info.rip_ids or FALLBACK_RIP_IDS,
        "person_ids": info.person_ids or FALLBACK_PERSON_IDS,
        "frame_skip": cfg["frame_skip"],
        "max_frames": cfg["max_frames"],
        "min_consecutive": cfg["min_consecutive"],
        "cooldown_sec": cfg["cooldown_sec"],
        "notifier_kind": notifier_kind,
        "dev_mode": dev,
    }
    return settings, detector


def _model_settings(cfg: dict, detector, rip_found, person_found,
                    rip_is_real: bool, person_is_real: bool) -> None:
    """모델 개발자용 설정. 두 모델의 임계값을 각각 잡는다.

    이안류(detect, 넓은 구역)와 원거리 CCTV의 작은 사람은 적정 임계값이 서로 다르다.
    한 값을 공유하면 한쪽은 반드시 놓치거나 과검출한다.
    """
    with st.sidebar.expander("모델 설정", expanded=True):
        # 라벨에 모델 이름을 넣어 두 묶음의 위젯을 구분한다.
        # 라벨과 값이 모두 같으면 Streamlit 이 같은 위젯으로 보고 DuplicateWidgetID 로 죽는다.
        for prefix, short, found, is_real in (
            ("rip", "이안류", rip_found, rip_is_real),
            ("person", "사람", person_found, person_is_real),
        ):
            st.markdown(f"**{short} 모델**")
            if is_real:
                st.caption(f"`{Path(found).name}` · 사용 클래스 {_class_ids(detector, prefix)}")
            else:
                st.caption("가중치 없음 — 데모(Fake)로 동작 중이라 아래 값은 적용되지 않습니다.")

            # number_input 을 쓴다: 슬라이더(0.05 스텝)로는 이안류 기본값인
            # 0.003 같은 미세한 값을 표현/입력할 수 없어 0.05로 뭉개져 버린다.
            cfg[f"{prefix}_conf"] = st.number_input(
                f"conf · {short}", min_value=0.0, max_value=0.95,
                value=float(cfg[f"{prefix}_conf"]), step=0.001, format="%.3f",
                help="신뢰도 임계값. 낮출수록 더 많이 잡습니다. "
                     "원거리 CCTV의 작은 사람이나 이안류처럼 미세한 값이 필요하면 "
                     "0.003 같은 값도 직접 입력할 수 있습니다.",
            )
            cfg[f"{prefix}_iou"] = st.slider(
                f"iou · {short}", 0.1, 0.95, float(cfg[f"{prefix}_iou"]), 0.05,
                help="겹친 탐지를 지우는 기준. 낮출수록 중복 박스를 많이 지웁니다.",
            )
            cfg[f"{prefix}_imgsz"] = st.selectbox(
                f"imgsz · {short}", IMGSZ_CHOICES,
                index=IMGSZ_CHOICES.index(cfg[f"{prefix}_imgsz"]),
                help="추론 해상도. 학습 때 쓴 값과 맞추세요. "
                     "키우면 작은 객체에 유리하지만 그만큼 느려집니다.",
            )
            st.divider()

        st.caption("값을 바꾸면 그 조합으로 모델을 다시 올립니다. 잠시 걸릴 수 있습니다.")


def _class_ids(detector, prefix: str) -> list[int]:
    return list(getattr(detector, f"{prefix}_class_ids", []))


if __name__ == "__main__":
    main()
