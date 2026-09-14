"""상시 관제 화면.

좌측: 가상 해변 A~E의 5분할 CCTV 벽
우측: 긴급 상황 카드와 전체 구역 상태 요약

새로고침은 st.fragment(run_every=...)로 이 영역만 돌린다.
페이지 전체를 rerun하면 사이드바 입력까지 다시 그려져서 화면이 깜빡인다.
"""

from __future__ import annotations

import time
from datetime import datetime

import cv2
import streamlit as st

from core.render import draw_overlay
from core.scenario import BEACHES, MODE_LABEL, build_streams
from core.schemas import RISK_LABEL, SAFETY_NOTE

REFRESH_SEC = 0.7

RISK_HEX = {"watch": "#2e9e5b", "warn": "#e08a00", "emergency": "#d6202a"}

ACTION_GUIDE = "해당 구역 즉시 확인 후 안내 방송과 현장 출동을 검토하세요."

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
    return build_streams(mode, _detector, rip_ids, person_ids)


def render(mode: str = "scenario", detector=None, weights_key: str = "",
           rip_ids=(), person_ids=()) -> None:
    streams = _streams(mode, weights_key, tuple(rip_ids), tuple(person_ids), detector)
    actual = streams[BEACHES[0].code].mode  # 모델이 없으면 scenario로 강등된다

    if actual != mode:
        st.info("모델이 없어 시나리오 모드로 표시합니다. 사이드바에서 모델 경로를 지정해 주세요.")

    missing = [code for code, s in streams.items() if s.beach.code and not _has_video(code)]
    if missing:
        st.info(
            f"영상이 없는 구역({', '.join(missing)})은 합성 배경으로 표시됩니다. "
            "`data/cctv/` 폴더에 `A.mp4` ~ `E.mp4` 를 넣으면 실제 영상으로 바뀝니다."
        )

    wall, panel = st.columns([3, 1.15], gap="medium")
    with wall:
        st.markdown(f"#### 다중 구역 관제  <span style='font-size:.8rem;opacity:.6'>"
                    f"{MODE_LABEL[actual]}</span>", unsafe_allow_html=True)
    with panel:
        st.markdown("#### 상황 정보")

    _live_area(wall, panel, streams)

    st.divider()
    st.caption(f"{MODE_NOTE[actual]} {SAFETY_NOTE} 화면의 A~E는 가상 관측구역이며 실제 해수욕장 정보가 아닙니다.")


def _has_video(code: str) -> bool:
    from core.scenario import resolve_sources
    return resolve_sources().get(code) is not None


@st.cache_resource
def _started_at() -> float:
    """앱 기동 시각. 애니메이션 시간축의 기준점이다."""
    return time.time()


@st.fragment(run_every=REFRESH_SEC)
def _live_area(wall, panel, streams) -> None:
    """이 함수만 주기적으로 다시 실행된다."""
    now = datetime.now()
    elapsed = time.time() - _started_at()   # 경과 초. Unix 시각을 그대로 쓰면 안 된다
    tick = int(elapsed / REFRESH_SEC)
    clock = now.strftime("%H:%M:%S")

    states, panels = [], {}
    for beach in BEACHES:
        frame, result, risk = streams[beach.code].at(tick)
        panels[beach.code] = draw_overlay(frame, result, risk, show_conf=False, time_label=clock)
        states.append((beach, risk, result))

    with wall:
        _draw_wall(states, panels)
    with panel:
        _draw_status(states, now)


def _draw_wall(states, panels) -> None:
    slots = list(st.columns(3, gap="small")) + list(st.columns(3, gap="small"))

    for slot, (beach, risk, result) in zip(slots, states):
        with slot:
            st.image(cv2.cvtColor(panels[beach.code], cv2.COLOR_BGR2RGB), width="stretch")
            color = RISK_HEX[risk]
            alarm = " 🚨" if risk == "emergency" else ""
            st.markdown(
                f"<div style='border-left:5px solid {color};padding:2px 0 2px 8px;margin-top:-6px'>"
                f"<span style='font-weight:700'>{beach.name}</span>"
                f"<span style='color:{color};font-weight:700'> · {RISK_LABEL[risk]}{alarm}</span><br>"
                f"<span style='font-size:0.78rem;opacity:.7'>{beach.camera_id} · "
                f"구역 내 {result.persons_in_rip}명 / 전체 {result.total_persons}명</span></div>",
                unsafe_allow_html=True,
            )

    with slots[5]:
        st.markdown(
            "<div style='font-size:0.82rem;line-height:1.9;padding-top:8px'><b>위험 등급</b><br>"
            f"<span style='color:{RISK_HEX['watch']}'>●</span> 관찰 · 이안류 의심 구역 없음<br>"
            f"<span style='color:{RISK_HEX['warn']}'>●</span> 경고 · 구역은 있으나 인원 없음<br>"
            f"<span style='color:{RISK_HEX['emergency']}'>●</span> 긴급 · 구역 안에 인원 감지</div>",
            unsafe_allow_html=True,
        )


def _draw_status(states, now) -> None:
    emergencies = [(b, r, res) for b, r, res in states if r == "emergency"]

    if emergencies:
        for beach, _risk, result in emergencies:
            st.markdown(
                f"<div style='background:{RISK_HEX['emergency']};color:#fff;"
                f"padding:14px 16px;border-radius:10px'>"
                f"<div style='font-size:0.8rem;letter-spacing:.08em;opacity:.9'>긴급</div>"
                f"<div style='font-size:1.5rem;font-weight:800;margin:2px 0 10px'>{beach.name}</div>"
                f"<div style='font-size:0.92rem;line-height:1.8'>"
                f"발생 시점 · {now.strftime('%H:%M:%S')}<br>"
                f"관측 구역 · {beach.zone}<br>"
                f"카메라 · {beach.camera_id}<br>"
                f"구역 내 인원 · <b>{result.persons_in_rip}명</b><br>"
                f"화면 전체 인원 · {result.total_persons}명<br>"
                f"의심 구역 · {result.rip_count}개</div></div>",
                unsafe_allow_html=True,
            )
            st.warning(ACTION_GUIDE)
    else:
        st.success("현재 긴급 상황이 없습니다.")

    st.markdown("###### 전체 구역 상태")
    for beach, risk, _result in states:
        color = RISK_HEX[risk]
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;align-items:center;"
            f"padding:6px 10px;margin-bottom:4px;border-radius:6px;background:rgba(128,128,128,.10)'>"
            f"<span><span style='color:{color}'>●</span> <b>{beach.name}</b> "
            f"<span style='font-size:.78rem;opacity:.65'>{beach.zone}</span></span>"
            f"<span style='color:{color};font-weight:700'>{RISK_LABEL[risk]}</span></div>",
            unsafe_allow_html=True,
        )

    st.caption(f"최근 갱신 {now.strftime('%H:%M:%S')} · {REFRESH_SEC}초 주기")
