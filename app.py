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
    DEFAULT_PERSON_IMGSZ,
    DEFAULT_PERSON_IOU,
    DEFAULT_RIP_CONF,
    build_detector,
    device_label,
    resolve_device,
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
    "person_iou": DEFAULT_PERSON_IOU,
    "person_imgsz": DEFAULT_PERSON_IMGSZ,
    "person_track": True,
    # TTA 는 추론 시간이 2~3배로 늘어 시연 중 화면이 눈에 띄게 느려진다. 기본 off.
    "person_augment": False,
    "smooth_window": 5,
    # 트래킹은 프레임이 연속일수록 안정적이라 기본 간격을 좁게 둔다.
    "frame_skip": 2,
    # frame_skip 과 함께 봐야 한다. 둘을 곱한 값이 분석이 닿는 프레임 수다.
    # 750 x 2 = 1500 프레임 = 30fps 기준 50초. frame_skip 을 키우면 이 값도
    # 같이 줄여야 분석 시간이 폭증하지 않는다.
    "max_frames": 750,
    "min_consecutive": 1,
    "cooldown_sec": 5.0,
    "dashboard_mode": "auto",
}

IMGSZ_CHOICES = [320, 416, 512, 640, 768, 960, 1024, 1280]

# 관제 화면의 기본 표시 방식. 모델이 둘 다 있어도 시나리오로 띄운다.
#
# 이 화면은 시스템이 어떻게 보이는지 설명하는 배경이고, 실제 탐지 성능은
# 영상 업로드 쪽에서 보여준다. 실제 추론으로 띄우면 구역마다 24장을 미리
# 돌려야 해서 첫 진입에 1분 넘게 걸린다 (사람 모델이 imgsz 1024 라 한 장에
# 0.26초). 게다가 그 24장은 영상 전체에서 몇 초씩 건너뛰며 뽑은 것이라
# 추적도 의미가 없다. 시나리오 모드는 추론을 아예 하지 않아 10초면 뜬다.
#
# 개발자 모드의 '관제 화면 표시 방식' 에서 hybrid/real 로 바꿀 수 있다.
# 관제 화면에서도 모델 결과를 확인하고 싶을 때만 쓴다 — 로딩이 길어진다.
DEFAULT_DASHBOARD_MODE = "scenario"

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
    person_track: bool,
    person_augment: bool,
):
    """모델은 세션마다 다시 올리면 안 되므로 cache_resource로 전역 공유한다.

    임계값도 캐시 키에 들어간다. 개발자 모드에서 conf 를 바꾸면 그 조합으로 새로 올라간다.
    트래킹/TTA 도 마찬가지다. 빠뜨리면 사이드바에서 값을 바꿔도
    캐시된 옛 detector 가 그대로 나온다.
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
        person_track=person_track,
        person_augment=person_augment,
    )


@st.cache_resource
def get_store() -> EventStore:
    return EventStore()


def main() -> None:
    st.markdown(_HEADER_CSS, unsafe_allow_html=True)
    st.session_state.setdefault("uploaded_video", None)
    st.session_state.setdefault("ui_mode", "user")
    st.session_state.setdefault("dev_settings", dict(DEV_DEFAULTS))
    # 앱을 고치고 새로고침했을 때 브라우저 세션에 남은 옛 설정 dict 에는
    # 새로 생긴 키가 없다. 빠진 것만 기본값으로 채운다 (KeyError 방지).
    for key, value in DEV_DEFAULTS.items():
        st.session_state["dev_settings"].setdefault(key, value)

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
        cfg["person_track"],
        cfg["person_augment"],
    )
    info = probe(detector)

    # 데모용 가짜 사람이 실제 이안류 구역 안에 들어가면 진짜와 똑같은 긴급 알림이 나간다.
    # 어느 쪽이 실제 추론인지 모드와 무관하게 항상 눈에 보여야 한다.
    # 연산 장치도 같이 띄운다. torch 가 CPU 전용 빌드로 깔려 있으면 GPU가 꽂혀 있어도
    # 조용히 CPU 로 떨어져 분석이 10배 이상 느려지는데, 화면에 안 보이면 시연 중에
    # 원인을 알 길이 없다.
    st.sidebar.caption(
        f"모델 · 이안류 {'실제 추론' if rip_is_real else '데모'}"
        f" / 사람 {'실제 추론' if person_is_real else '데모'}"
        f" · {device_label()}"
    )
    if (rip_is_real or person_is_real) and not resolve_device().startswith("cuda"):
        st.sidebar.warning(
            "GPU를 쓰지 않아 분석이 10배 이상 느립니다. "
            "requirements.txt 의 torch 설치 안내를 확인하세요."
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

    auto_mode = DEFAULT_DASHBOARD_MODE
    dashboard_mode = auto_mode

    if dev:
        _model_settings(cfg, detector, rip_found, person_found, rip_is_real, person_is_real)

        with st.sidebar.expander("분석 설정"):
            cfg["frame_skip"] = st.slider(
                "프레임 간격", 1, 30, cfg["frame_skip"],
                help="N프레임마다 한 번 추론합니다. 값을 키우면 빨라지고 놓치는 순간이 늘어납니다.",
            )
            if cfg["person_track"] and cfg["frame_skip"] > 4:
                st.caption(
                    "간격이 넓으면 추적 번호가 자주 바뀝니다. "
                    "속도가 필요하면 이 값보다 사람 모델의 imgsz 를 먼저 낮추세요."
                )
            cfg["max_frames"] = int(
                st.number_input("최대 추론 횟수", 10, 5000, cfg["max_frames"], step=10)
            )

        with st.sidebar.expander("판정·발송 기준"):
            # 침범 기준점은 발끝(수면 접점)으로 고정한다. 원근이 있는 CCTV 화면에서
            # 박스 중심은 실제 사람이 서 있는 위치보다 늘 위쪽을 가리켜, 구역 경계에서
            # 판정이 어긋난다. 모델팀 추론 스크립트도 같은 기준을 쓴다.
            st.caption("구역 침범 기준점 · 발끝(수면 접점) 고정")
            cfg["smooth_window"] = st.slider(
                "추적 스무딩 창", 1, 15, cfg["smooth_window"],
                help="사람별로 최근 N회 판정을 모아 다수결로 확정합니다. "
                     "박스가 구역 경계에서 떨리는 것을 걸러냅니다. 1이면 사용하지 않습니다.",
            )
            if cfg["smooth_window"] > 1 and not cfg["person_track"]:
                st.caption("추적이 꺼져 있어 스무딩이 적용되지 않습니다.")
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
                    "scenario": "시나리오 (전부 시연용, 즉시 표시)",
                    "hybrid": "사람만 실제 탐지 (로딩 김)",
                    "real": "전체 실제 추론 (로딩 김)",
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
            elif dashboard_mode != "scenario":
                # 구역마다 24장을 미리 추론한다. 시연 직전에 무심코 켜면
                # 화면이 1분 넘게 안 뜬다.
                st.caption(
                    "구역 5곳을 미리 추론하느라 첫 표시까지 1분 이상 걸립니다. "
                    "모델 성능 확인은 영상 업로드 쪽이 빠릅니다."
                )

    settings = {
        "dashboard_mode": dashboard_mode,
        # 임계값이 바뀌면 관제 화면의 사전 추론 결과도 다시 만들어야 한다.
        # 이 키가 그대로면 conf 를 바꿔도 예전 결과가 그대로 보인다.
        "weights_key": (
            f"{rip_weights_key}@{cfg['rip_conf']}/{cfg['rip_iou']}/{cfg['rip_imgsz']}"
            f"|{person_weights_key}@{cfg['person_conf']}/{cfg['person_iou']}/{cfg['person_imgsz']}"
            f"/track={cfg['person_track']}/tta={cfg['person_augment']}"
        ),
        "rip_ids": info.rip_ids or FALLBACK_RIP_IDS,
        "person_ids": info.person_ids or FALLBACK_PERSON_IDS,
        "frame_skip": cfg["frame_skip"],
        "max_frames": cfg["max_frames"],
        "min_consecutive": cfg["min_consecutive"],
        "cooldown_sec": cfg["cooldown_sec"],
        "smooth_window": cfg["smooth_window"],
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

            # 트래킹과 TTA 는 사람 모델에만 붙인다. 이안류 구역은 개체를 세는
            # 대상이 아니라 번호를 이어줄 이유가 없고, 두 기법 모두 느려진다.
            if prefix == "person":
                cfg["person_track"] = st.checkbox(
                    "추적 사용 (ByteTrack)", value=bool(cfg["person_track"]),
                    help="같은 사람에게 프레임 내내 같은 번호를 붙입니다. "
                         "번호가 있어야 '추적 스무딩'이 동작합니다.",
                )
                cfg["person_augment"] = st.checkbox(
                    "TTA 사용 (augment)", value=bool(cfg["person_augment"]),
                    help="여러 배율로 추론해 합칩니다. 작은 사람을 더 잡지만 "
                         "추론 시간이 2~3배로 늘어 시연이 느려집니다.",
                )
            st.divider()

        st.caption("값을 바꾸면 그 조합으로 모델을 다시 올립니다. 잠시 걸릴 수 있습니다.")


def _class_ids(detector, prefix: str) -> list[int]:
    return list(getattr(detector, f"{prefix}_class_ids", []))


if __name__ == "__main__":
    main()
