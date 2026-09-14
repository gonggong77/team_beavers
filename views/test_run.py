"""테스트 영상 분석 화면.

관제 화면과 같은 레이아웃을 쓴다.
큰 화면 자리에 업로드한 영상의 분석 결과가 재생되고, 우측에 상황 정보가 뜬다.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

from core.notifier import build_notifier
from core.pipeline import PipelineConfig, run
from core.render import risk_bar_html, status_panel_html
from core.schemas import RISK_LABEL, SAFETY_NOTE
from core.store import EventStore

ACTION_GUIDE = {
    "emergency": "즉시 확인 후 안내 방송과 현장 출동을 검토하세요.",
    "warn": "물놀이객이 구역에 진입하는지 우선 관찰하세요.",
    "watch": "정상 관측 중입니다.",
}


def save_upload(uploaded) -> Path:
    suffix = Path(uploaded.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getbuffer())
        return Path(tmp.name)


def render(uploaded, detector, store: EventStore, settings: dict) -> None:
    video_path = save_upload(uploaded)

    viewer, panel = st.columns([2.7, 1], gap="medium")

    with viewer:
        st.markdown(
            f"#### 테스트 영상 분석 "
            f"<span style='font-size:.8rem;opacity:.6'>{uploaded.name}</span>",
            unsafe_allow_html=True,
        )
    with panel:
        st.markdown("#### 상황 정보")

    if not settings["rip_ids"] or not settings["person_ids"]:
        with viewer:
            st.error("사이드바에서 이안류 클래스와 사람 클래스를 각각 하나 이상 지정해 주세요.")
        return

    with viewer:
        start = st.button("분석 시작", type="primary", width="stretch")
        if not start:
            st.video(str(video_path))
    with panel:
        if not start:
            st.info("분석 시작을 누르면 이 자리에 상황 정보가 표시됩니다.")
            st.caption("제목을 누르면 상시 관제 화면으로 돌아갑니다.")

    if start:
        _analyze(viewer, panel, video_path, detector, store, settings)

    _event_log(store)


def _analyze(viewer, panel, video_path: Path, detector, store: EventStore, settings: dict) -> None:
    notifier = build_notifier(settings["notifier_kind"])
    if settings["notifier_kind"] == "telegram" and not getattr(notifier, "configured", False):
        with panel:
            st.warning("텔레그램 토큰이 설정되지 않아 발송이 실패합니다. .env를 확인해 주세요.")

    config = PipelineConfig(
        rip_ids=settings["rip_ids"],
        person_ids=settings["person_ids"],
        frame_skip=settings["frame_skip"],
        max_frames=settings["max_frames"],
        min_consecutive=settings["min_consecutive"],
        cooldown_sec=settings["cooldown_sec"],
    )

    with viewer:
        frame_slot = st.empty()
        bar_slot = st.empty()
        progress = st.progress(0.0, text="분석을 준비하고 있습니다.")
    with panel:
        panel_slot = st.empty()
        alert_slot = st.container()

    alerts = 0
    for step in run(video_path, detector, config, store, notifier):
        frame_slot.image(cv2.cvtColor(step.annotated_bgr, cv2.COLOR_BGR2RGB), width="stretch")

        bar_slot.markdown(risk_bar_html(step.risk), unsafe_allow_html=True)
        panel_slot.markdown(
            status_panel_html(
                risk=step.risk,
                title="테스트 영상",
                info_rows=[
                    ("영상 위치", f"{step.result.timestamp_sec:.1f}초"),
                    ("프레임", str(step.result.frame_idx)),
                    ("추론 시간", f"{step.result.infer_ms:.0f} ms"),
                ],
                metrics=[
                    ("구역 내 인원", f"{step.result.persons_in_rip}명", True),
                    ("화면 전체 인원", f"{step.result.total_persons}명", False),
                    ("의심 구역", f"{step.result.rip_count}개", False),
                ],
                guide=ACTION_GUIDE[step.risk],
                footer=f"발송 알림 {alerts}건",
            ),
            unsafe_allow_html=True,
        )

        progress.progress(
            min(max(step.progress, 0.0), 1.0),
            text=f"{step.result.timestamp_sec:.1f}초 지점 분석 중",
        )

        if step.event:
            alerts += 1
            with alert_slot:
                st.error(
                    f"긴급 알림 발송 · {step.event.occurred_at} · "
                    f"영상 {step.event.timestamp_sec:.1f}초 · 구역 내 {step.event.persons_in_rip}명"
                )

    progress.progress(1.0, text="분석 완료")
    with viewer:
        st.success(f"분석을 마쳤습니다. 발송된 긴급 알림 {alerts}건")
        st.caption(SAFETY_NOTE)


def _event_log(store: EventStore) -> None:
    st.divider()
    st.subheader("알림 이벤트 기록")

    events = store.list_events()
    if not events:
        st.caption("아직 기록된 이벤트가 없습니다.")
        return

    frame = pd.DataFrame(events)[
        ["occurred_at", "risk_level", "persons_in_rip", "total_persons",
         "rip_count", "timestamp_sec", "video_source", "notify_status"]
    ].rename(columns={
        "occurred_at": "발생 시각", "risk_level": "등급", "persons_in_rip": "구역 내 인원",
        "total_persons": "전체 인원", "rip_count": "구역 수", "timestamp_sec": "영상 위치(초)",
        "video_source": "영상", "notify_status": "발송 상태",
    })
    frame["등급"] = frame["등급"].map(RISK_LABEL).fillna(frame["등급"])
    st.dataframe(frame, width="stretch", hide_index=True)

    left, right = st.columns(2)
    left.download_button(
        "이벤트 내려받기 (JSONL)", store.to_jsonl(),
        file_name="alert_events.jsonl", mime="application/x-ndjson", width="stretch",
    )
    if right.button("기록 비우기", width="stretch"):
        store.clear()
        st.rerun()
