"""테스트 영상 분석 화면.

관제 화면과 같은 레이아웃을 쓴다.
큰 화면 자리에 업로드한 영상의 분석 결과가 재생되고, 우측에 상황 정보가 뜬다.
"""

from __future__ import annotations

import os
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

TRIGGER_LABEL = {
    "auto": "자동",
    "escalation": "인원 증가",
}


def save_upload(uploaded) -> Path:
    # NamedTemporaryFile 은 원본 파일명을 tmpXXXXXXXX 같은 랜덤 이름으로 지워버린다.
    # pipeline.py 가 video_path 의 파일명을 그대로 "영상 이름"으로 쓰기 때문에,
    # 업로드 이름을 유지한 채 저장하도록 업로드마다 임시 디렉터리를 하나씩 판다.
    tmp_dir = Path(tempfile.mkdtemp())
    path = tmp_dir / uploaded.name
    path.write_bytes(uploaded.getbuffer())
    return path


def _get_video_path(uploaded) -> Path:
    """업로드 파일이 바뀌지 않았으면 임시파일을 새로 쓰지 않는다.

    save_upload 를 매 리런마다 부르면 파일이 계속 쌓이고, video_path 도
    매번 바뀌어 st.video 가 재생 위치를 잃는다.
    """
    key = f"{uploaded.name}:{uploaded.size}"
    if st.session_state.get("tr_video_key") != key:
        st.session_state["tr_video_key"] = key
        st.session_state["tr_video_path"] = save_upload(uploaded)
    return st.session_state["tr_video_path"]


def render(uploaded, detector, store: EventStore, settings: dict) -> None:
    video_path = _get_video_path(uploaded)

    viewer, panel = st.columns([2.7, 1], gap="medium")

    with viewer:
        st.markdown(
            f"#### 테스트 영상 분석 "
            f"<span style='font-size:.8rem;opacity:.6'>{uploaded.name}</span>",
            unsafe_allow_html=True,
        )
    with panel:
        st.markdown("#### 상황 정보")

    with viewer:
        start = st.button("분석 시작", type="primary", width="stretch")
        if not start:
            st.video(str(video_path))
    with panel:
        if not start:
            st.info("분석 시작을 누르면 이 자리에 상황 정보가 표시됩니다.")
            st.caption("좌측 상단 '← 상시 관제 화면으로' 를 누르면 관제 화면으로 돌아갑니다.")

    if start:
        _analyze(viewer, panel, video_path, detector, store, settings)

    _gallery(store)
    _event_log(store, settings.get("dev_mode", False))


def _analyze(viewer, panel, video_path: Path, detector, store: EventStore, settings: dict) -> None:
    notifier = build_notifier(settings["notifier_kind"])
    if not getattr(notifier, "configured", True):
        with panel:
            st.warning(notifier.setup_hint)

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

    dev = settings.get("dev_mode", False)

    alerts = 0
    for step in run(video_path, detector, config, store, notifier):
        frame_slot.image(cv2.cvtColor(step.annotated_bgr, cv2.COLOR_BGR2RGB), width="stretch")

        # 프레임 번호와 추론 시간은 튜닝용 정보라 개발자 모드에서만 띄운다.
        info_rows = [("영상 위치", f"{step.result.timestamp_sec:.1f}초")]
        if dev:
            info_rows += [
                ("프레임", str(step.result.frame_idx)),
                ("추론 시간", f"{step.result.infer_ms:.0f} ms"),
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
            reason = TRIGGER_LABEL.get(step.event.trigger_kind, step.event.trigger_kind)
            with alert_slot:
                st.error(
                    f"긴급 알림 발송({reason}) · {step.event.occurred_at} · "
                    f"영상 {step.event.timestamp_sec:.1f}초 · 구역 내 {step.event.persons_in_rip}명 "
                    "· 이미지 업로드로 잠시 멈췄습니다"
                )

    progress.progress(1.0, text="분석 완료")
    with viewer:
        st.success(f"분석을 마쳤습니다. 발송된 긴급 알림 {alerts}건")
        st.caption(SAFETY_NOTE)


def _gallery(store: EventStore) -> None:
    """앱으로 실제 보낸 이미지를 화면에 보여준다.

    폰에 뜬 사진과 같은 것인지 이 자리에서 바로 대조할 수 있어야 한다.
    snapshot_path 가 core.pipeline._prune_snapshots 로 정리돼 파일이
    사라진 이벤트는 조용히 건너뛴다 (오류로 취급하지 않는다).
    """
    events = [
        e for e in store.list_events()
        if e.get("snapshot_path") and os.path.exists(e["snapshot_path"])
    ][:12]
    if not events:
        return

    st.divider()
    st.subheader("발송된 경보 이미지")

    columns = st.columns(3)
    for idx, event in enumerate(events):
        with columns[idx % 3]:
            st.image(event["snapshot_path"], width="stretch")
            reason = TRIGGER_LABEL.get(event.get("trigger_kind", "auto"), event.get("trigger_kind", "auto"))
            st.caption(
                f"구역 내 {event['persons_in_rip']}명 · {event['occurred_at']} · 발송 사유: {reason}"
            )
            if event.get("image_url"):
                st.code(event["image_url"], language=None)
            else:
                st.caption("URL 없음 (콘솔/텔레그램 알림이거나 업로드 실패)")


def _event_log(store: EventStore, dev: bool = False) -> None:
    st.divider()
    st.subheader("알림 이벤트 기록")

    events = store.list_events()
    if not events:
        st.caption("아직 기록된 이벤트가 없습니다.")
        return

    frame = pd.DataFrame(events)[
        ["occurred_at", "risk_level", "persons_in_rip", "total_persons",
         "rip_count", "timestamp_sec", "video_source", "trigger_kind", "notify_status", "image_url"]
    ].rename(columns={
        "occurred_at": "발생 시각", "risk_level": "등급", "persons_in_rip": "구역 내 인원",
        "total_persons": "전체 인원", "rip_count": "구역 수", "timestamp_sec": "영상 위치(초)",
        "video_source": "영상", "trigger_kind": "발송 사유", "notify_status": "발송 상태",
        "image_url": "이미지 URL",
    })
    frame["발송 사유"] = frame["발송 사유"].map(TRIGGER_LABEL).fillna(frame["발송 사유"])
    frame["등급"] = frame["등급"].map(RISK_LABEL).fillna(frame["등급"])
    st.dataframe(frame, width="stretch", hide_index=True)

    # 내려받기와 기록 삭제는 개발·검증용이다. 특히 삭제는 되돌릴 수 없어 사용자 모드에서 감춘다.
    if not dev:
        return

    left, right = st.columns(2)
    left.download_button(
        "이벤트 내려받기 (JSONL)", store.to_jsonl(),
        file_name="alert_events.jsonl", mime="application/x-ndjson", width="stretch",
    )
    if right.button("기록 비우기", width="stretch"):
        store.clear()
        st.rerun()
