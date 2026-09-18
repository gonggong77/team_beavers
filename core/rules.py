"""위험 등급 판정과 알림 발송 게이트.

Streamlit에 의존하지 않는다. 알림 파트가 UI 없이 단독으로 테스트할 수 있어야 한다.

    python -m core.rules   # 자체 점검 실행
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from core.schemas import FrameResult, PersonBox, RipRegion, RiskLevel

# 사람이 이안류 구역 안에 있다고 볼 기준점
#   "center"  : 바운딩 박스 중심. 수영 중인 사람에 적합 (기본값)
#   "bottom"  : 박스 아래쪽 중앙. 서 있는 사람의 발 위치에 적합
InsideRule = str


def point_in_polygon(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
    if len(polygon) < 3:
        return False
    contour = np.asarray(polygon, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), False) >= 0


def mark_persons_in_rip(
    persons: list[PersonBox],
    rips: list[RipRegion],
    rule: InsideRule = "center",
) -> list[PersonBox]:
    """각 사람의 in_rip 플래그를 채워서 그대로 돌려준다."""
    for person in persons:
        point = person.bottom_center if rule == "bottom" else person.center
        person.in_rip = any(point_in_polygon(point, rip.polygon) for rip in rips)
    return persons


def grade(result: FrameResult) -> RiskLevel:
    """FR-03 위험 등급 판정.

    관찰 : 이안류 의심 구역이 없음
    경고 : 의심 구역은 있으나 구역 안에 사람이 없음
    긴급 : 의심 구역 안에 사람이 있음
    """
    if result.rip_count == 0:
        return "watch"
    if result.persons_in_rip == 0:
        return "warn"
    return "emergency"


@dataclass
class AlertGate:
    """알림 스팸 방지.

    30fps 영상에서 긴급 판정이 200프레임 연속 나오면 메시지도 200통 간다.
    아래 두 조건을 모두 통과해야 발송한다.

    min_consecutive : 긴급 판정이 N회 연속일 때만 발송 (순간 오탐 제거)
    cooldown_sec    : 마지막 발송 후 N초간 재발송 금지 (영상 시간 기준)
    """

    min_consecutive: int = 1
    cooldown_sec: float = 20.0

    _streak: int = 0
    _last_sent_at: float | None = None
    _last_sent_persons: int = 0

    def should_alert(self, risk: RiskLevel, timestamp_sec: float) -> bool:
        if risk != "emergency":
            self._streak = 0
            return False

        self._streak += 1
        if self._streak < self.min_consecutive:
            return False

        if self._last_sent_at is not None and (timestamp_sec - self._last_sent_at) < self.cooldown_sec:
            return False

        self._last_sent_at = timestamp_sec
        return True

    def check(self, risk: RiskLevel, timestamp_sec: float, persons_in_rip: int) -> str | None:
        """should_alert 에 '인원 증가 시 쿨다운 무시' 규칙을 얹은 판정.

        구역 내 인원이 계속 커져야만 재발송되므로 (3명→5명→7명 식) 별도
        쿨다운 없이도 자연히 제한된다. escalation 분기는 streak 조건 때문에
        should_alert 가 이미 한 번 True 를 낸 뒤에만 열려, 첫 경보를 앞지르지 못한다.
        """
        if self.should_alert(risk, timestamp_sec):
            self._last_sent_persons = persons_in_rip
            return "auto"

        if (
            risk == "emergency"
            and self._streak >= self.min_consecutive
            and persons_in_rip > self._last_sent_persons
        ):
            self._last_sent_at = timestamp_sec
            self._last_sent_persons = persons_in_rip
            return "escalation"

        return None

    def reset(self) -> None:
        self._streak = 0
        self._last_sent_at = None
        self._last_sent_persons = 0


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

    warn = FrameResult(frame_idx=1, timestamp_sec=0.1, width=640, height=480, rips=rips, persons=[outside])
    assert grade(warn) == "warn"

    emergency = FrameResult(frame_idx=2, timestamp_sec=0.2, width=640, height=480, rips=rips, persons=[inside])
    assert grade(emergency) == "emergency"

    gate = AlertGate(min_consecutive=3, cooldown_sec=10.0)
    assert gate.should_alert("emergency", 0.0) is False   # 1회
    assert gate.should_alert("emergency", 0.1) is False   # 2회
    assert gate.should_alert("emergency", 0.2) is True    # 3회 -> 발송
    assert gate.should_alert("emergency", 0.3) is False   # 쿨다운
    assert gate.should_alert("emergency", 11.0) is True   # 쿨다운 해제
    assert gate.should_alert("warn", 12.0) is False       # 긴급 아님 -> 연속 카운트 초기화

    escalation_gate = AlertGate(min_consecutive=3, cooldown_sec=10.0)
    assert escalation_gate.check("emergency", 0.0, 3) is None       # 1회
    assert escalation_gate.check("emergency", 0.1, 3) is None       # 2회
    assert escalation_gate.check("emergency", 0.2, 3) == "auto"     # 3회 -> 자동 발송
    assert escalation_gate.check("emergency", 0.3, 3) is None       # 쿨다운 중, 인원 동일
    assert escalation_gate.check("emergency", 0.4, 2) is None       # 쿨다운 중, 인원 감소
    assert escalation_gate.check("emergency", 0.5, 5) == "escalation"  # 쿨다운 중이어도 인원 증가 -> 즉시 발송
    assert escalation_gate.check("emergency", 0.6, 5) is None       # 같은 인원수로는 재발송 안 함

    print("rules self-check passed")


if __name__ == "__main__":
    _self_check()
