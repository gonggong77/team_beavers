"""상시 관제 화면.

좌측: 큰 CCTV 화면 + 영상 바로 아래 전체 폭 등급 표시줄
우측: 상황 정보 패널 하나. 영상과 같은 높이로 맞춰진다.

구역 A~E는 10초 간격으로 자동 순환한다.
사이드바에서 테스트 영상을 올리면 같은 자리에 분석 결과가 대신 재생된다.
"""

from __future__ import annotations

import time
from datetime import datetime

import cv2
import streamlit as st

from core.render import draw_overlay, risk_bar_html, status_panel_html
from core.scenario import build_main_streams, rotating_index
from core.schemas import SAFETY_NOTE

REFRESH_SEC = 0.7

ACTION_GUIDE = {
    "emergency": "즉시 확인 후 안내 방송과 현장 출동을 검토하세요.",
    "warn": "물놀이객이 구역에 진입하는지 우선 관찰하세요.",
    "watch": "정상 관측 중입니다.",
}

MODE_NOTE = {
    "scenario": "이안류 구역과 사람 표시는 시연을 위해 구성된 값입니다. 실제 추론 결과가 아닙니다.",
    "hybrid": "사람은 실제 탐지 결과이며, 이안류 의심 구역은 시연을 위해 구성된 값입니다.",
    "real": "이안류 구역과 사람 모두 실제 모델 추론 결과입니다.",
}


@st.cache_resource(show_spinner="관제 화면을 준비하고 있습니다. 실제 추론 모드는 시간이 걸립니다.")
def _streams(mode: str, weights_key: str, rip_ids: tuple, person_ids: tuple, _detector):
    """모드나 모델이 바뀌면 자동으로 다시 만든다.

    _detector 는 앞에 밑줄이 있어 캐시 키에서 제외된다.
    대신 weights_key 가 모델 식별자 역할을 한다.
    """
    return build_main_streams(mode, _detector, rip_ids, person_ids)


@st.cache_resource
def _started_at() -> float:
    """앱 기동 시각. 재생 위치와 구역 순환의 기준점이다."""
    return time.time()


def render(mode: str = "scenario", detector=None, weights_key: str = "",
           rip_ids=(), person_ids=()) -> None:
    streams = _streams(mode, weights_key, tuple(rip_ids), tuple(person_ids), detector)
    actual = streams[0].mode  # 모델이 없으면 scenario로 강등된다

    if actual != mode:
        st.info("모델이 없어 시나리오 모드로 표시합니다. 사이드바에서 모델 경로를 지정해 주세요.")

    viewer, panel = st.columns([2.7, 1], gap="medium")
    _live_area(viewer, panel, streams)

    st.divider()
    st.caption(
        f"{MODE_NOTE[actual]} {SAFETY_NOTE} "
        "화면의 A~E는 가상 관측구역이며 실제 해수욕장 정보가 아닙니다."
    )


@st.fragment(run_every=REFRESH_SEC)
def _live_area(viewer, panel, streams) -> None:
    """이 함수만 주기적으로 다시 실행된다."""
    now = datetime.now()
    elapsed = time.time() - _started_at()   # 경과 초. Unix 시각을 그대로 쓰면 안 된다
    tick = int(elapsed / REFRESH_SEC)
    clock = now.strftime("%H:%M:%S")

    stream = streams[rotating_index(elapsed, len(streams))]
    beach = stream.beach

    frame, result, risk = stream.at(tick)
    annotated = draw_overlay(frame, result, risk, show_conf=True, time_label=clock)

    with viewer:
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), width="stretch")
        st.markdown(risk_bar_html(risk), unsafe_allow_html=True)

    with panel:
        st.markdown(
            status_panel_html(
                risk=risk,
                title=beach.name,
                info_rows=[
                    ("확인 시점", clock),
                    ("관측 구역", beach.zone),
                    ("카메라", beach.camera_id),
                ],
                metrics=[
                    ("구역 내 인원", f"{result.persons_in_rip}명", True),
                    ("화면 전체 인원", f"{result.total_persons}명", False),
                    ("의심 구역", f"{result.rip_count}개", False),
                ],
                guide=ACTION_GUIDE[risk],
                footer=f"최근 갱신 {clock} · {REFRESH_SEC}초 주기",
            ),
            unsafe_allow_html=True,
        )
