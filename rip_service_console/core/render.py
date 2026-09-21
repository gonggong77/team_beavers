"""오버레이 렌더링 (동영상 프레임 하단 위험 상태 일체형 렌더러 및 이전 호환 함수 포함)."""

from __future__ import annotations

import cv2
import numpy as np

from core.schemas import RISK_COLOR_BGR, RISK_LABEL, FrameResult, RiskLevel

RIP_FILL_ALPHA = 0.28
PERSON_SAFE_BGR = (90, 220, 90)
PERSON_RISK_BGR = (0, 0, 255)

RISK_HEX = {"watch": "#2e9e5b", "warn": "#e08a00", "emergency": "#d6202a"}

RISK_DESC = {
    "watch": "이안류 의심 구역 없음",
    "warn": "구역은 있으나 인원 없음",
    "emergency": "구역 안에 인원 감지",
}


def draw_overlay(
    frame_bgr: np.ndarray,
    result: FrameResult,
    risk: RiskLevel,
    show_conf: bool = True,
    time_label: str | None = None,
    show_foot: bool = False,
) -> np.ndarray:
    canvas = frame_bgr.copy()
    color = RISK_COLOR_BGR[risk]

    # 1. 이안류 구역: 반투명 채움 + 외곽선
    if result.rips:
        layer = canvas.copy()
        for rip in result.rips:
            pts = np.asarray(rip.polygon, dtype=np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(layer, [pts], color)
        canvas = cv2.addWeighted(layer, RIP_FILL_ALPHA, canvas, 1 - RIP_FILL_ALPHA, 0)
        for rip in result.rips:
            pts = np.asarray(rip.polygon, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=2)
            if show_conf:
                x, y = pts[0][0]
                cv2.putText(
                    canvas,
                    f"rip {rip.conf:.2f}",
                    (int(x), max(int(y) - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA,
                )

    # 2. 사람 박스: 구역 안이면 빨강, 밖이면 초록
    for person in result.persons:
        x1, y1, x2, y2 = (int(v) for v in person.xyxy)
        box_color = PERSON_RISK_BGR if person.in_rip else PERSON_SAFE_BGR
        thickness = 3 if person.in_rip else 1
        cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, thickness)
        if show_conf:
            tag = "" if person.track_id is None else f"#{person.track_id} "
            cv2.putText(
                canvas,
                f"{tag}{person.conf:.2f}",
                (x1, max(y1 - 4, 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                box_color,
                1,
                cv2.LINE_AA,
            )
        if show_foot:
            fx, fy = person.bottom_center
            cv2.circle(canvas, (int(fx), int(fy)), 4, box_color, -1)

    # 상/하단 한글 상태 표시는 Streamlit HTML 바로 렌더링한다.
    return canvas


def top_bar_html(risk: RiskLevel, result: FrameResult, clock: str) -> str:
    """영상 위에 붙는 한글 상태 요약 바."""
    color = RISK_HEX[risk]
    return (
        f"<div style='background:{color};color:#fff;border-radius:5px 5px 0 0;"
        f"padding:13px 18px 28px;font-size:1rem;font-weight:800'>"
        f"{RISK_LABEL[risk]} | 구역 내 인원 {result.persons_in_rip}"
        f" | 화면 전체 인원 {result.total_persons}"
        f" | 이안류 의심 구역 {result.rip_count} | {clock}</div>"
    )


IDLE_HEX = "#f0f2f6"
IDLE_TEXT_HEX = "#31333f"


def idle_top_bar_html(text: str) -> str:
    """대기/준비 상태에서 쓰는 중립색 상단 바 (분석중 상단 바와 동일 구조)."""
    return (
        f"<div style='background:{IDLE_HEX};color:{IDLE_TEXT_HEX};border-radius:5px 5px 0 0;"
        f"padding:13px 18px 28px;font-size:1rem;font-weight:800'>{text}</div>"
    )


def idle_bottom_panel_html(zone: str, camera: str, text: str, guide: str) -> str:
    """대기/준비 상태에서 쓰는 중립색 하단 패널 (분석중 하단 패널과 동일 구조)."""
    return (
        f"<div style='background:{IDLE_HEX};color:{IDLE_TEXT_HEX};border-radius:0 0 5px 5px;"
        f"padding:16px;text-align:center;margin-bottom:12px'>"
        f"<div style='font-size:.95rem;font-weight:700;opacity:.95'>{zone} | {camera}</div>"
        f"<div style='font-size:2.6rem;font-weight:900;line-height:1.2;margin:2px 0 6px'>"
        f"{text}</div>"
        f"<div style='font-size:.95rem;font-weight:600;opacity:.95'>{guide}</div></div>"
    )


def bottom_panel_html(risk: RiskLevel, zone: str, camera: str, guide: str) -> str:
    """영상 아래에 붙는 한글 위험 등급 패널."""
    color = RISK_HEX[risk]
    return (
        f"<div style='background:{color};color:#fff;border-radius:0 0 5px 5px;"
        f"padding:16px;text-align:center;margin-bottom:12px'>"
        f"<div style='font-size:.95rem;font-weight:700;opacity:.95'>{zone} | {camera}</div>"
        f"<div style='font-size:2.6rem;font-weight:900;line-height:1.2;margin:2px 0 6px'>"
        f"{RISK_LABEL[risk]}</div>"
        f"<div style='font-size:.95rem;font-weight:600;opacity:.95'>{guide}</div></div>"
    )


def risk_bar_html(risk: RiskLevel) -> str:
    """기존 dashboard.py 호환용 전체 폭 등급 표시줄 HTML 함수."""
    color = RISK_HEX[risk]
    icon = "🚨 " if risk == "emergency" else ""
    return (
        f"<div style='background:{color};color:#fff;border-radius:14px;"
        f"padding:18px 16px;text-align:center;margin-top:8px;overflow:hidden;"
        f"box-shadow:0 4px 18px {color}55'>"
        f"<div style='font-size:clamp(2.2rem,5vw,3.6rem);font-weight:900;"
        f"line-height:1.05;letter-spacing:.06em'>{icon}{RISK_LABEL[risk]}</div>"
        f"<div style='font-size:clamp(.9rem,1.4vw,1.1rem);margin-top:8px;opacity:.92'>"
        f"{RISK_DESC[risk]}</div></div>"
    )


def status_panel_html(
    risk: RiskLevel,
    title: str,
    info_rows: list[tuple[str, str]],
    metrics: list[tuple[str, str, bool]],
    guide: str,
    footer: str,
) -> str:
    """우측 상황 정보 패널 HTML."""
    color = RISK_HEX[risk]

    info = "".join(
        f"<div style='display:flex;gap:6px'>"
        f"<span style='opacity:.85'>{label}</span><span>·</span>"
        f"<span style='font-weight:600'>{value}</span></div>"
        for label, value in info_rows
    )

    card = (
        f"<div style='flex:2.4 1 0;background:{color};color:#fff;border-radius:12px;"
        f"padding:14px 16px;overflow:hidden;display:flex;flex-direction:column;"
        f"justify-content:center;min-height:0'>"
        f"<div style='font-size:.8rem;letter-spacing:.08em;opacity:.9'>{RISK_LABEL[risk]}</div>"
        f"<div style='font-size:1.45rem;font-weight:800;margin:2px 0 8px'>{title}</div>"
        f"<div style='font-size:.88rem;line-height:1.65'>{info}</div></div>"
    )

    boxes = "".join(
        f"<div style='flex:1 1 0;min-height:0;display:flex;align-items:center;"
        f"justify-content:space-between;padding:0 14px;border-radius:10px;"
        f"background:rgba(128,128,128,.10)'>"
        f"<span style='font-size:.84rem;opacity:.75'>{label}</span>"
        f"<span style='font-size:{'1.5rem' if strong else '1.25rem'};"
        f"font-weight:{'800' if strong else '600'}'>{value}</span></div>"
        for label, value, strong in metrics
    )

    guide_box = (
        f"<div style='flex:1 1 0;min-height:0;display:flex;align-items:center;"
        f"padding:0 14px;border-radius:10px;background:{color}1f;color:{color};"
        f"font-size:.88rem;font-weight:600'>{guide}</div>"
    )

    footer_box = (
        f"<div style='flex:0 0 auto;font-size:.76rem;opacity:.55;padding-top:2px'>{footer}</div>"
    )

    head = (
        "<div style='flex:0 0 auto;font-size:1.15rem;font-weight:700;"
        "padding-bottom:2px'>상황 정보</div>"
    )

    return (
        "<div style='display:flex;flex-direction:column;gap:8px;"
        "aspect-ratio:1/1.51875;min-height:430px'>"
        + head + card + boxes + guide_box + footer_box +
        "</div>"
    )