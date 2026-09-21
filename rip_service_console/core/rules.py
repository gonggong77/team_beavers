"""화면 표시에 필요한 최소 판정.

원본 core/rules.py 에서 '화면이 무슨 색을 칠할지' 를 정하는 부분만 옮겼다.
알림 스팸 방지(AlertGate), 트랙 스무딩(TrackSmoother) 같은 발송·추론 쪽 규칙은
이 UI 버전에 들어 있지 않다.

    python -m core.rules   # 자체 점검
"""

from __future__ import annotations

import cv2
import numpy as np

from core.schemas import FrameResult, PersonBox, RipRegion, RiskLevel

# 사람이 구역 안에 있다고 볼 기준점
#   "center" : 바운딩 박스 중심
#   "bottom" : 박스 아래쪽 중앙(발끝/수면 접점). 원본과 같은 기본값
InsideRule = str


def point_in_polygon(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
    if len(polygon) < 3:
        return False
    contour = np.asarray(polygon, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), False) >= 0


def mark_persons_in_rip(
    persons: list[PersonBox],
    rips: list[RipRegion],
    rule: InsideRule = "bottom",
) -> list[PersonBox]:
    """각 사람의 in_rip 플래그를 채워서 그대로 돌려준다."""
    for person in persons:
        point = person.bottom_center if rule == "bottom" else person.center
        person.in_rip = any(point_in_polygon(point, rip.polygon) for rip in rips)
    return persons


def grade(result: FrameResult) -> RiskLevel:
    """위험 등급 판정.

    관찰 : 이안류 의심 구역이 없음
    경고 : 의심 구역은 있으나 구역 안에 사람이 없음
    긴급 : 의심 구역 안에 사람이 있음
    """
    if result.rip_count == 0:
        return "watch"
    if result.persons_in_rip == 0:
        return "warn"
    return "emergency"


def _self_check() -> None:
    square = [(10.0, 10.0), (90.0, 10.0), (90.0, 90.0), (10.0, 90.0)]
    rips = [RipRegion(polygon=square, conf=0.9)]
    inside = PersonBox(xyxy=(40, 40, 60, 60), conf=0.8)
    outside = PersonBox(xyxy=(200, 200, 220, 220), conf=0.8)

    mark_persons_in_rip([inside, outside], rips)
    assert inside.in_rip is True
    assert outside.in_rip is False

    empty = FrameResult(frame_idx=0, timestamp_sec=0.0, width=640, height=480)
    assert grade(empty) == "watch"

    warn = FrameResult(1, 0.1, 640, 480, rips=rips, persons=[outside])
    assert grade(warn) == "warn"

    emergency = FrameResult(2, 0.2, 640, 480, rips=rips, persons=[inside])
    assert grade(emergency) == "emergency"

    print("rules self-check passed")


if __name__ == "__main__":
    _self_check()
