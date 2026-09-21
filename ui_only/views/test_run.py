"""테스트 영상 화면 (UI 전용).

관제 화면과 같은 레이아웃을 쓴다.
큰 화면 자리에 업로드한 영상이 재생되고, 우측에 상황 정보가 뜬다.

원본 앱은 이 자리에서 실제 추론을 돌리고 알림을 발송한다.
UI 버전은 core.demo_feed 가 만든 시연용 결과를 흘려보낼 뿐,
모델도 부르지 않고 어디로도 메시지를 보내지 않는다.

이벤트 기록과 경보 이미지는 st.session_state 에만 쌓인다 (DB 없음).
새로고침하면 사라진다.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

from core.demo_feed import play
from core.render import risk_bar_html, status_panel_html
from core.schemas import RISK_LABEL, SAFETY_NOTE

ACTION_GUIDE = {
    "emergency": "즉시 확인 후 안내 방송과 현장 출동을 검토하세요.",
    "warn": "물놀이객이 구역에 진입하는지 우선 관찰하세요.",
    "watch": "정상 관측 중입니다.",
}

TRIGGER_LABEL = {
    "auto": "자동",
    "escalation": "인원 증가",
}

MAX_GALLERY = 12   # 갤러리에 남겨둘 경보 이미지 수


def save_upload(uploaded) -> Path:
    # NamedTemporaryFile 은 원본 파일명을 tmpXXXXXXXX 같은 랜덤 이름으로 지워버린다.
    # 화면이 파일명을 '영상 이름' 으로 그대로 쓰기 때문에, 업로드 이름을 유지한 채
    # 저장하도록 업로드마다 임시 디렉터리를 하나씩 판다.
    tmp_dir = Path(tempfile.mkdtemp())
    path = tmp_dir / uploaded.name
    path.write_bytes(uploaded.getbuffer())
    return path


def _get_video_path(uploaded) -> Path:
    """업로드 파일이 바뀌지 않았으면 임시파일을 새로 쓰지 않는다.

    매 리런마다 저장하면 파일이 계속 쌓이고, 경로도 매번 바뀌어
    st.video 가 재생 위치를 잃는다.
    """
    key = f"{uploaded.name}:{uploaded.size}"
    if st.session_state.get("tr_video_key") != key:
        st.session_state["tr_video_key"] = key
        st.session_state["tr_video_path"] = save_upload(uploaded)
    return st.session_state["tr_video_path"]


def render(uploaded, settings: dict) -> None:
    video_path = _get_video_path(uploaded)
    st.session_state.setdefault("events", [])      # 이벤트 기록 (dict 목록)
    st.session_state.setdefault("snapshots", [])   # 경보 이미지 (RGB 배열, 캡션)

    viewer, panel = st.columns([2.7, 1], gap="medium")

    with viewer:
        st.markdown(
            f"#### 테스트 영상 재생 "
            f"<span style='font-size:.8rem;opacity:.6'>{uploaded.name}</span>",
            unsafe_allow_html=True,
        )
    with panel:
        st.markdown("#### 상황 정보")

    with viewer:
        start = st.button("재생 시작", type="primary", width="stretch")
        if not start:
            st.video(str(video_path))
    with panel:
        if not start:
            st.info("재생 시작을 누르면 이 자리에 상황 정보가 표시됩니다.")
            st.caption("좌측 상단 '← 상시 관제 화면으로' 를 누르면 관제 화면으로 돌아갑니다.")

    if start:
        _play(viewer, panel, video_path, uploaded.name, settings)

    _gallery()
    _event_log(settings.get("dev_mode", False))


def _play(viewer, panel, video_path: Path, source_name: str, settings: dict) -> None:
    dev = settings.get("dev_mode", False)

    with viewer:
        frame_slot = st.empty()
        bar_slot = st.empty()
        progress = st.progress(0.0, text="화면을 준비하고 있습니다.")
    with panel:
        panel_slot = st.empty()
        alert_slot = st.container()

    alerts = 0
    for step in play(
        video_path,
        frame_skip=settings["frame_skip"],
        max_frames=settings["max_frames"],
        show_foot=settings.get("show_foot", False),
        source_name=source_name,
    ):
        rgb = cv2.cvtColor(step.annotated_bgr, cv2.COLOR_BGR2RGB)
        frame_slot.image(rgb, width="stretch")

        # 프레임 번호와 처리 시간은 개발용 정보라 개발자 모드에서만 띄운다.
        info_rows = [("영상 위치", f"{step.result.timestamp_sec:.1f}초")]
        if dev:
            info_rows += [
                ("프레임", str(step.result.frame_idx)),
                ("처리 시간", f"{step.result.infer_ms:.0f} ms"),
            ]

        bar_slot.markdown(risk_bar_html(step.risk), unsafe_allow_html=True)
        panel_slot.markdown(
            status_panel_html(
                risk=step.risk,
                title="테스트 영상",
                info_rows=info_rows,
                metrics=[
                    ("구역 내 인원", f"{step.result.persons_in_rip}명", True),
                    ("화면 전체 인원", f"{step.result.total_persons}명", False),
                    ("의심 구역", f"{step.result.rip_count}개", False),
                ],
                guide=ACTION_GUIDE[step.risk],
                footer=f"경보 {alerts}건",
            ),
            unsafe_allow_html=True,
        )

        progress.progress(
            min(max(step.progress, 0.0), 1.0),
            text=f"{step.result.timestamp_sec:.1f}초 지점 재생 중",
        )

        if step.event:
            alerts += 1
            event = step.event
            reason = TRIGGER_LABEL.get(event.trigger_kind, event.trigger_kind)
            st.session_state["events"].insert(0, event.to_dict())
            st.session_state["snapshots"].insert(
                0,
                (rgb, f"구역 내 {event.persons_in_rip}명 · {event.occurred_at} · 발송 사유: {reason}"),
            )
            del st.session_state["snapshots"][MAX_GALLERY:]
            with alert_slot:
                st.error(
                    f"긴급 경보({reason}) · {event.occurred_at} · "
                    f"영상 {event.timestamp_sec:.1f}초 · 구역 내 {event.persons_in_rip}명"
                )

    progress.progress(1.0, text="재생 완료")
    with viewer:
        st.success(f"재생을 마쳤습니다. 화면에 표시된 긴급 경보 {alerts}건")
        st.caption(SAFETY_NOTE)


def _gallery() -> None:
    """경보가 뜬 순간의 화면을 모아 보여준다.

    원본 앱에서는 실제로 앱에 발송된 이미지가 이 자리에 온다.
    """
    snapshots = st.session_state.get("snapshots", [])
    if not snapshots:
        return

    st.divider()
    st.subheader("경보 발생 화면")

    columns = st.columns(3)
    for idx, (image, caption) in enumerate(snapshots):
        with columns[idx % 3]:
            st.image(image, width="stretch")
            st.caption(caption)


def _event_log(dev: bool = False) -> None:
    st.divider()
    st.subheader("알림 이벤트 기록")

    events = st.session_state.get("events", [])
    if not events:
        st.caption("아직 기록된 이벤트가 없습니다.")
        return

    frame = pd.DataFrame(events)[
        ["occurred_at", "risk_level", "persons_in_rip", "total_persons",
         "rip_count", "timestamp_sec", "video_source", "trigger_kind", "notify_status"]
    ].rename(columns={
        "occurred_at": "발생 시각", "risk_level": "등급", "persons_in_rip": "구역 내 인원",
        "total_persons": "전체 인원", "rip_count": "구역 수", "timestamp_sec": "영상 위치(초)",
        "video_source": "영상", "trigger_kind": "발송 사유", "notify_status": "발송 상태",
    })
    frame["발송 사유"] = frame["발송 사유"].map(TRIGGER_LABEL).fillna(frame["발송 사유"])
    frame["등급"] = frame["등급"].map(RISK_LABEL).fillna(frame["등급"])
    st.dataframe(frame, width="stretch", hide_index=True)
    st.caption("발송 기능이 빠져 있어 '발송 상태' 는 항상 ui-demo 입니다.")

    # 기록 삭제는 되돌릴 수 없어 사용자 모드에서 감춘다 (원본 앱과 같은 규칙).
    if not dev:
        return
    if st.button("기록 비우기", width="stretch"):
        st.session_state["events"] = []
        st.session_state["snapshots"] = []
        st.rerun()
