"""알림 스팸 방지 게이트.

원본 core/rules.py 의 AlertGate 를 그대로 옮긴 것이다. 탐지·화면 쪽 규칙
(구역 침범 판정, 등급 판정, 트랙 스무딩)은 이 폴더에 들어 있지 않다.

    python -m core.gate   # 자체 점검
"""

from __future__ import annotations

from dataclasses import dataclass

from core.schemas import RiskLevel


@dataclass
class AlertGate:
    """발송 횟수를 제한한다.

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

        돌려주는 값이 곧 AlertEvent.trigger_kind 다 ("auto" / "escalation" / None).

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
    gate = AlertGate(min_consecutive=3, cooldown_sec=10.0)
    assert gate.should_alert("emergency", 0.0) is False   # 1회
    assert gate.should_alert("emergency", 0.1) is False   # 2회
    assert gate.should_alert("emergency", 0.2) is True    # 3회 -> 발송
    assert gate.should_alert("emergency", 0.3) is False   # 쿨다운
    assert gate.should_alert("emergency", 11.0) is True   # 쿨다운 해제
    assert gate.should_alert("warn", 12.0) is False       # 긴급 아님 -> 연속 카운트 초기화

    escalation = AlertGate(min_consecutive=3, cooldown_sec=10.0)
    assert escalation.check("emergency", 0.0, 3) is None       # 1회
    assert escalation.check("emergency", 0.1, 3) is None       # 2회
    assert escalation.check("emergency", 0.2, 3) == "auto"     # 3회 -> 자동 발송
    assert escalation.check("emergency", 0.3, 3) is None       # 쿨다운 중, 인원 동일
    assert escalation.check("emergency", 0.4, 2) is None       # 쿨다운 중, 인원 감소
    assert escalation.check("emergency", 0.5, 5) == "escalation"  # 인원 증가 -> 즉시 발송
    assert escalation.check("emergency", 0.6, 5) is None       # 같은 인원수로는 재발송 안 함

    print("gate self-check passed")


if __name__ == "__main__":
    _self_check()
