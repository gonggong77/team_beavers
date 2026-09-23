"""비디오 정밀 관제 분석 화면 (재시작 시 캡처 갤러리 및 우측 경보 카드 완전 초기화 적용)."""

from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from alert.store import EventStore
from core.pipeline import run_pipeline
from core.render import (
    bottom_panel_html,
    idle_bottom_panel_html,
    idle_top_bar_html,
    top_bar_html,
)
from core.schemas import RISK_LABEL

ACTION_GUIDE = {
    "emergency": "즉시 확인 후 안내 방송과 현장 출동을 검토하세요.",
    "warn": "물놀이객이 구역에 진입하는지 우선 관찰하세요.",
    "watch": "정상 관측 중입니다.",
}

TRIGGER_LABEL = {
    "auto": "자동",
    "escalation": "인원 증가",
}

MAX_GALLERY = 12

ZONE_LABEL = "가상 관측 구역 A-1"
CAMERA_ID = "CCTV-A01"

# 좌측 사이드바 상태별 [가운데, 오른쪽] 컬럼 비율. 원하는 값으로 직접 수정.
SIDEBAR_EXPANDED_RATIO = (2.3, 1)
SIDEBAR_COLLAPSED_RATIO = (3, 1)

# 좌측 사이드바 상태별 우측 캡처 갤러리 높이(px). 원하는 값으로 직접 수정.
GALLERY_HEIGHT_EXPANDED = 560
GALLERY_HEIGHT_COLLAPSED = 700


def save_upload(uploaded) -> Path:
    tmp_dir = Path(tempfile.mkdtemp())
    path = tmp_dir / uploaded.name
    path.write_bytes(uploaded.getbuffer())
    return path


DEFAULT_VIDEO_PATH = (
    Path(__file__).resolve().parents[1] / "test_video" / "rip_current_testv1.mp4"
)


def _get_video_path(uploaded) -> Path:
    if uploaded is None:
        return DEFAULT_VIDEO_PATH
    key = f"{uploaded.name}:{uploaded.size}"
    if st.session_state.get("tr_video_key") != key:
        st.session_state["tr_video_key"] = key
        st.session_state["tr_video_path"] = save_upload(uploaded)
    return st.session_state["tr_video_path"]


@st.cache_data(show_spinner=False)
def _first_frame_rgb(video_path: Path):
    """대기 화면용 첫 프레임. 분석중 화면과 같은 st.image 요소로 띄우기 위함."""
    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    cap.release()
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ok else None


def render(uploaded, settings: dict, start_clicked: bool = False) -> None:
    video_path = _get_video_path(uploaded)
    source_name = uploaded.name if uploaded is not None else video_path.name
    
    # ⭐️ [실시간 관제 시작] 버튼을 누르면 이전 분석 데이터 즉시 초기화
    if start_clicked:
        st.session_state["run_analysis"] = True
        st.session_state["snapshots"] = []  # 캡처 갤러리 초기화
        st.session_state["events"] = []     # 이벤트 로그 초기화
        st.session_state["last_frame"] = None
        _collapse_sidebar()

    st.session_state.setdefault("events", [])
    st.session_state.setdefault("snapshots", [])

    is_running = st.session_state.get("run_analysis", False)
    last_frame = st.session_state.get("last_frame")

    _apply_collapsed_sidebar_ratio()

    viewer, panel = st.columns(list(SIDEBAR_EXPANDED_RATIO), gap="medium")

    with panel:
        st.markdown("<div style='height:50px'></div>", unsafe_allow_html=True)
        if st.button("📋 실시간 로그 확인하기", use_container_width=True, disabled=is_running):
            show_log_dialog()
        st.markdown(
            "<div style='font-size:1.15rem;font-weight:700;margin-top:20px;margin-bottom:20px'>"
            "🚨 긴급 경보 발생 순간 캡처</div>",
            unsafe_allow_html=True,
        )
        gallery_slot = st.empty()
    _gallery(gallery_slot, interactive=not is_running)

    with viewer:
        st.markdown(
            "<div style='font-size:2.5rem;font-weight:700;line-height:1.3;margin-top:-30px;margin-bottom:20px'>"
            "🌊 이안류 조난자 AI 관제 시스템</div>",
            unsafe_allow_html=True,
        )

    if is_running:
        _play(viewer, gallery_slot, video_path, source_name, settings)
    elif last_frame is not None:
        with viewer:
            st.markdown(idle_top_bar_html("분석 완료"), unsafe_allow_html=True)
            st.image(last_frame, use_container_width=True)
            st.markdown(
                idle_bottom_panel_html(
                    ZONE_LABEL, CAMERA_ID, "분석 완료",
                    "우측 캡처를 확인 처리하면 목록에서 사라집니다.",
                ),
                unsafe_allow_html=True,
            )
    else:
        with viewer:
            st.markdown(idle_top_bar_html("분석 대기중입니다"), unsafe_allow_html=True)
            st.markdown(
                "<div style='aspect-ratio:16/9;min-height:340px;background:#000;color:#fff;"
                "display:flex;align-items:center;justify-content:center;font-size:1.1rem'>"
                "CCTV 를 준비중입니다</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                idle_bottom_panel_html(
                    ZONE_LABEL, CAMERA_ID, "분석 대기중입니다",
                    "실시간 관제 시작 버튼을 누르면 분석이 시작됩니다.",
                ),
                unsafe_allow_html=True,
            )


def _apply_collapsed_sidebar_ratio() -> None:
    """좌측 사이드바가 접혀 있는 동안 가운데/오른쪽 컬럼 비율과 갤러리 높이를 덮어쓴다.

    st.columns 로 준 비율/갤러리 높이는 사이드바를 펼친 상태 기준이다. Streamlit 은
    접힘 상태를 서버로 알려주지 않아 Python 조건문으로 분기할 수 없으므로, 사이드바에
    붙는 aria-expanded 속성을 :has() 로 잡아 CSS 로만 값을 바꾼다.
    """
    left, right = SIDEBAR_COLLAPSED_RATIO
    st.markdown(
        f"""<style>
        body:has(section[data-testid="stSidebar"][aria-expanded="false"])
        div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(1) {{
            flex: {left} 1 0% !important; width: auto !important;
        }}
        body:has(section[data-testid="stSidebar"][aria-expanded="false"])
        div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(2) {{
            flex: {right} 1 0% !important; width: auto !important;
        }}
        body:has(section[data-testid="stSidebar"][aria-expanded="false"])
        div[data-testid="stVerticalBlock"][style*="{GALLERY_HEIGHT_EXPANDED}px"] {{
            height: {GALLERY_HEIGHT_COLLAPSED}px !important;
        }}
        </style>""",
        unsafe_allow_html=True,
    )


def _collapse_sidebar() -> None:
    """분석 시작 시 좌측 사이드바를 한 번 접는다.

    set_page_config(initial_sidebar_state) 는 브라우저 localStorage 값에 밀리므로
    사용자가 직접 누르는 것과 같은 경로인 접기 버튼 클릭을 주입한다.
    nonce 는 iframe 이 매번 다시 마운트되어 스크립트가 재실행되게 한다.
    """
    nonce = st.session_state.get("collapse_nonce", 0) + 1
    st.session_state["collapse_nonce"] = nonce
    components.html(
        f"""<script>
        /* {nonce} */
        const doc = window.parent.document;
        const sidebar = doc.querySelector('[data-testid="stSidebar"]');
        if (sidebar && sidebar.getAttribute('aria-expanded') === 'true') {{
            doc.querySelector('[data-testid="stSidebarCollapseButton"] button')?.click();
        }}
        </script>""",
        height=0,
    )



def _play(viewer, gallery_slot, video_path: Path, source_name: str, settings: dict) -> None:
    with viewer:
        top_slot = st.empty()
        frame_slot = st.empty()
        bottom_slot = st.empty()
        top_slot.markdown(idle_top_bar_html("CCTV 를 준비중입니다"), unsafe_allow_html=True)
        # 대기 화면과 같은 첫 프레임을 유지해야 시작 순간에 화면이 깜빡이지 않는다.
        first_frame = _first_frame_rgb(video_path)
        if first_frame is None:
            frame_slot.markdown(
                "<div style='aspect-ratio:16/9;min-height:340px;background:#000;color:#fff;"
                "display:flex;align-items:center;justify-content:center;font-size:1.1rem'>"
                "CCTV 를 준비중입니다</div>",
                unsafe_allow_html=True,
            )
        else:
            frame_slot.image(first_frame, use_container_width=True)
        bottom_slot.markdown(
            idle_bottom_panel_html(
                ZONE_LABEL, CAMERA_ID, "CCTV 를 준비중입니다",
                "잠시만 기다려 주세요. 곧 분석이 시작됩니다.",
            ),
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
        progress = st.progress(0.0)

    alerts = 0
    last_rgb = None

    for step in run_pipeline(
        video_path=video_path,
        frame_skip=settings["frame_skip"],
        max_frames=settings["max_frames"],
        show_foot=settings.get("show_foot", False),
        source_name=source_name,
        rip_model_path=settings.get("rip_model_path", "best_yolo11m_integrated_v2_first.pt"),
        swimmer_model_path=settings.get("swimmer_model_path", "best_swimmer_yolo11m_b8.pt"),
        swimmer_conf=settings.get("swimmer_conf", 0.10),
    ):
        rgb = cv2.cvtColor(step.annotated_bgr, cv2.COLOR_BGR2RGB)
        last_rgb = rgb

        top_slot.markdown(
            top_bar_html(step.risk, step.result, datetime.now().strftime("%H:%M:%S")),
            unsafe_allow_html=True,
        )
        frame_slot.image(rgb, use_container_width=True)
        bottom_slot.markdown(
            bottom_panel_html(step.risk, ZONE_LABEL, CAMERA_ID, ACTION_GUIDE[step.risk]),
            unsafe_allow_html=True,
        )

        progress.progress(
            min(max(step.progress, 0.0), 1.0),
            text=f"{step.result.timestamp_sec:.1f}초 지점 분석 중 (추론 {step.result.infer_ms:.0f}ms)",
        )

        if step.event:
            alerts += 1
            event = step.event
            st.session_state["events"].insert(0, {**event.to_dict(), "handled": False})
            st.session_state["snapshots"].insert(
                0,
                {
                    "event_id": event.event_id,
                    "image": rgb,
                    "caption": f"구역 내 {event.persons_in_rip}명 감지 · {event.occurred_at}",
                },
            )
            del st.session_state["snapshots"][MAX_GALLERY:]
            _gallery(gallery_slot)

    progress.progress(1.0, text="영상 분석 완료")
    st.markdown(
        "<style>div[data-testid='stProgressBarTrack'] > div"
        "{background-color:#22c55e !important;}</style>",
        unsafe_allow_html=True,
    )

    st.session_state["last_frame"] = last_rgb
    st.session_state["run_analysis"] = False
    st.rerun()


def _gallery(slot, interactive: bool = False) -> None:
    """우측 캡처 목록 (스크롤 영역).

    분석 루프 중에는 위젯을 만들지 않는다. 위젯 key 중복 검사가 스크립트 런 단위라
    루프가 갤러리를 다시 그릴 때 중복 키 오류가 나고, 클릭 시 분석도 중단되기 때문이다.
    """
    snapshots = st.session_state.get("snapshots", [])
    with slot.container(height=GALLERY_HEIGHT_EXPANDED):
        if not snapshots:
            st.caption("아직 감지된 긴급 경보가 없습니다.")
            return
        for snapshot in snapshots:
            st.image(snapshot["image"], use_container_width=True)
            st.caption(snapshot["caption"])
            if interactive:
                st.checkbox(
                    "✅ 확인 완료",
                    key=f"ack_{snapshot['event_id']}",
                    on_change=_ack,
                    args=(snapshot["event_id"],),
                )


def _ack(event_id: str) -> None:
    """관리자 확인 처리: DB에 기록하고 우측 목록에서 제거한다."""
    EventStore().mark_handled(event_id)
    st.session_state["snapshots"] = [
        s for s in st.session_state.get("snapshots", []) if s["event_id"] != event_id
    ]
    for e in st.session_state.get("events", []):
        if e["event_id"] == event_id:
            e["handled"] = True


@st.dialog("📋 실시간 탐지 이벤트 로그", width="large")
def show_log_dialog() -> None:
    _event_log()


def _event_log(dev: bool = False) -> None:
    events = st.session_state.get("events", [])
    if not events:
        st.caption("아직 감지된 긴급 이벤트가 없습니다.")
        return

    frame = pd.DataFrame(events)[
        ["occurred_at", "risk_level", "persons_in_rip", "total_persons",
         "rip_count", "timestamp_sec", "video_source", "trigger_kind", "notify_status", "handled"]
    ].rename(columns={
        "occurred_at": "발생 시각", "risk_level": "위험 등급", "persons_in_rip": "구역 내 인원",
        "total_persons": "전체 인원", "rip_count": "이안류 수", "timestamp_sec": "영상 위치(초)",
        "video_source": "분석 영상", "trigger_kind": "경보 사유", "notify_status": "발송 상태",
        "handled": "확인 처리",
    })
    frame["경보 사유"] = frame["경보 사유"].map(TRIGGER_LABEL).fillna(frame["경보 사유"])
    frame["위험 등급"] = frame["위험 등급"].map(RISK_LABEL).fillna(frame["위험 등급"])
    frame["확인 처리"] = frame["확인 처리"].map({True: "✅ 처리됨", False: "대기중"})
    st.dataframe(frame, use_container_width=True, hide_index=True)

    if not dev:
        return
    if st.button("이벤트 기록 초기화", use_container_width=True):
        st.session_state["events"] = []
        st.session_state["snapshots"] = []
        st.session_state["run_analysis"] = False
        st.rerun()