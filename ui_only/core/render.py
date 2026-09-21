"""오버레이 렌더링.

Ultralytics의 results.plot()을 쓰지 않는 이유:
  1) 모델 출력 형식이 바뀌면 화면이 같이 깨진다
  2) 위험 등급별 색과 한글 문구를 우리 마음대로 못 넣는다
직접 그리면 FakeDetector와 실제 모델이 완전히 같은 화면을 만든다.
"""

from __future__ import annotations

import cv2
import numpy as np

from core.schemas import RISK_COLOR_BGR, RISK_LABEL, FrameResult, RiskLevel

RIP_FILL_ALPHA = 0.28
PERSON_SAFE_BGR = (90, 220, 90)
PERSON_RISK_BGR = (0, 0, 255)


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

    # 이안류 구역: 반투명 채움 + 외곽선
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
                cv2.putText(canvas, f"rip {rip.conf:.2f}", (int(x), max(int(y) - 6, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # 사람 박스: 구역 안이면 빨강, 밖이면 초록
    for person in result.persons:
        x1, y1, x2, y2 = (int(v) for v in person.xyxy)
        box_color = PERSON_RISK_BGR if person.in_rip else PERSON_SAFE_BGR
        thickness = 3 if person.in_rip else 1
        cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, thickness)
        if show_conf:
            # 트랙 번호가 있으면 앞에 붙인다. 같은 번호가 유지되는지 눈으로 확인하는 용도다.
            tag = "" if person.track_id is None else f"#{person.track_id} "
            cv2.putText(canvas, f"{tag}{person.conf:.2f}", (x1, max(y1 - 4, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, box_color, 1, cv2.LINE_AA)
        if show_foot:
            # 침범 판정에 실제로 쓰이는 점. 박스가 구역에 걸쳤을 때
            # 왜 그런 판정이 나왔는지 이 점 하나로 설명된다.
            fx, fy = person.bottom_center
            cv2.circle(canvas, (int(fx), int(fy)), 3, box_color, -1)

    return _draw_banner(canvas, result, risk, time_label)


def _draw_banner(canvas: np.ndarray, result: FrameResult, risk: RiskLevel,
                 time_label: str | None = None) -> np.ndarray:
    """상단 상태 띠. 한글 폰트 의존을 피하려고 영문 키워드를 쓴다.

    한글로 띄우고 싶으면 Pillow + NanumGothic 으로 교체하면 되지만,
    폰트 파일이 배포 환경에 없으면 깨지므로 MVP에서는 영문으로 둔다.
    한글 등급 문구는 Streamlit 화면 쪽에서 별도로 크게 표시한다.
    """
    h, w = canvas.shape[:2]
    band = max(int(h * 0.07), 34)
    color = RISK_COLOR_BGR[risk]

    cv2.rectangle(canvas, (0, 0), (w, band), color, thickness=-1)
    stamp = time_label if time_label is not None else f"t={result.timestamp_sec:.1f}s"
    text = (
        f"{risk.upper()}  |  in-zone {result.persons_in_rip}"
        f"  |  total {result.total_persons}"
        f"  |  zones {result.rip_count}"
        f"  |  {stamp}"
    )
    cv2.putText(canvas, text, (12, int(band * 0.68)),
                cv2.FONT_HERSHEY_SIMPLEX, min(w / 1100, 0.9), (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def risk_badge_markdown(risk: RiskLevel) -> str:
    """Streamlit 본문에 띄울 한글 등급 배지."""
    emoji = {"watch": "🟢", "warn": "🟠", "emergency": "🔴"}[risk]
    return f"{emoji} **{RISK_LABEL[risk]}**"


# --------------------------------------------------------------------------
# 화면용 HTML 조각
# --------------------------------------------------------------------------

RISK_HEX = {"watch": "#2e9e5b", "warn": "#e08a00", "emergency": "#d6202a"}

RISK_DESC = {
    "watch": "이안류 의심 구역 없음",
    "warn": "구역은 있으나 인원 없음",
    "emergency": "구역 안에 인원 감지",
}


def risk_bar_html(risk: RiskLevel) -> str:
    """영상 바로 아래에 깔 전체 폭 등급 표시줄.

    현재 등급 하나만 화면 너비 전체에 크게 띄운다.
    멀리서도 색과 글자만 보고 즉시 판단할 수 있어야 한다.
    """
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


def status_panel_html(risk: RiskLevel, title: str, info_rows: list[tuple[str, str]],
                      metrics: list[tuple[str, str, bool]], guide: str, footer: str) -> str:
    """우측 상황 정보 패널 전체를 한 덩어리로 만든다.

    영상과 높이를 맞추는 것이 핵심이다.
    영상은 16:9이고 좌우 열 비율이 2.7:1이므로, 패널 높이는 항상 패널 너비의
    2.7 x 9/16 = 1.51875 배가 된다. aspect-ratio 로 그 비율을 고정하면
    창 크기가 바뀌어도 영상 아래끝과 패널 아래끝이 함께 움직인다.

    요소를 따로 그리지 않고 하나의 flex 열로 묶어야 간격이 일정하고 겹치지 않는다.
    """
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
