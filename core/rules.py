"""위험 등급 판정과 알림 발송 게이트.

Streamlit에 의존하지 않는다. 알림 파트가 UI 없이 단독으로 테스트할 수 있어야 한다.

    python -m core.rules   # 자체 점검 실행
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from core.schemas import FrameResult, PersonBox, RipRegion, RiskLevel

# 사람이 이안류 구역 안에 있다고 볼 기준점
#   "center"  : 바운딩 박스 중심. 수영 중인 사람에 적합
#   "bottom"  : 박스 아래쪽 중앙(발끝/수면 접점). 모델팀 추론 스크립트와 같은 기준 (기본값)
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
class TrackSmoother:
    """트랙 ID별로 최근 판정을 모아 다수결로 in_rip 을 확정한다.

    AlertGate 와 역할이 다르다. AlertGate 는 '장면 전체'가 몇 번 연속 긴급인지를 보고,
    이쪽은 '사람 한 명'이 정말 구역 안에 있는지를 본다. 사람 박스가 구역 경계에서
    한두 프레임 떨리는 경우는 AlertGate 만으로 걸러지지 않는다 — 판정이 계속
    긴급으로 유지되기 때문이다.

    window=1 이면 아무 것도 하지 않으므로 트래킹을 끈 것과 같은 결과가 나온다.
    track_id 가 없는 박스(트래킹 off, 또는 추적 시작 직후)는 그대로 통과시킨다.
    """

    window: int = 5
    min_votes: int = 3
    # 연속 미탐지가 이만큼 이어지면 그 트랙의 기록을 버린다.
    # ByteTrack 의 track_buffer 와 같은 값이라, 트래커가 트랙을 포기하는 시점과
    # 기록이 사라지는 시점이 맞는다.
    forget_after: int = 30

    _history: dict[int, deque[bool]] = field(default_factory=dict)
    _missing: dict[int, int] = field(default_factory=dict)

    def apply(self, persons: list[PersonBox]) -> list[PersonBox]:
        """각 사람의 in_rip 을 평활화해서 그대로 돌려준다."""
        if self.window <= 1:
            return persons

        seen: set[int] = set()
        for person in persons:
            if person.track_id is None:
                continue
            seen.add(person.track_id)
            self._missing.pop(person.track_id, None)
            votes = self._history.setdefault(person.track_id, deque(maxlen=self.window))
            votes.append(person.in_rip)
            # 기준표는 창 크기로 한 번 자른다. len(votes) 로 자르면 트랙이 막 생긴
            # 첫 프레임에서 기준이 1표가 되어 스무딩이 통째로 무력해진다.
            person.in_rip = sum(votes) >= min(self.min_votes, self.window)

        # 이번에 안 보인 트랙의 기록은 지우지 않고 그대로 얼려 둔다.
        # 여기서 바로 버리면 한 프레임 놓칠 때마다 표가 초기화돼, 깜빡임을 견디려고
        # 만든 스무딩이 정작 깜빡임에서 무너진다. 우리 사람 모델은 conf 0.10 으로
        # 돌아 탐지 점수가 0.2 언저리라 한두 프레임 누락이 흔하다.
        # 안 보이는 사람은 어차피 persons 목록에 없어 인원 집계에 들어가지 않으므로,
        # 기록을 남겨둔다고 해서 사라진 사람이 계속 긴급으로 잡히지는 않는다.
        for track_id in list(self._history):
            if track_id in seen:
                continue
            self._missing[track_id] = self._missing.get(track_id, 0) + 1
            if self._missing[track_id] >= self.forget_after:
                del self._history[track_id]
                del self._missing[track_id]
        return persons

    def reset(self) -> None:
        self._history.clear()
        self._missing.clear()


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

    # 발끝 기준 판정: 박스 중심은 구역 밖이지만 발끝은 안쪽인 경우
    straddler = PersonBox(xyxy=(40, 60, 60, 95), conf=0.8)   # center y=77.5(안), bottom y=95(밖)
    mark_persons_in_rip([straddler], rips, rule="center")
    assert straddler.in_rip is True
    mark_persons_in_rip([straddler], rips, rule="bottom")
    assert straddler.in_rip is False

    # 트랙 스무딩: 한두 프레임 흔들려도 결과가 뒤집히지 않아야 한다
    smoother = TrackSmoother(window=5, min_votes=3)

    def _vote(flag: bool) -> bool:
        box = PersonBox(xyxy=(40, 40, 60, 60), conf=0.8, in_rip=flag, track_id=1)
        return smoother.apply([box])[0].in_rip

    assert _vote(True) is False     # 1표 — 기준(3표) 미달
    assert _vote(True) is False     # 2표
    assert _vote(True) is True      # 3표 -> 확정
    assert _vote(False) is True     # 순간 누락, 창 안에 3표가 남아 유지
    assert _vote(False) is True     # [T,T,T,F,F] — 아직 3표
    assert _vote(False) is False    # 창이 밀려 2표로 떨어져 해제

    # 한 프레임 통째로 놓쳐도 (그 사람이 목록에서 빠져도) 쌓아둔 표는 남아야 한다.
    # 스무딩을 넣은 이유가 바로 이 깜빡임이다.
    flicker = TrackSmoother(window=5, min_votes=3)

    def _seen(flag: bool) -> bool:
        box = PersonBox(xyxy=(40, 40, 60, 60), conf=0.2, in_rip=flag, track_id=7)
        return flicker.apply([box])[0].in_rip

    assert _seen(True) is False
    assert _seen(True) is False
    assert _seen(True) is True      # 3표 -> 확정
    flicker.apply([])               # 미탐지 한 프레임 (목록에 아예 없음)
    assert _seen(True) is True      # 곧바로 복귀해야 한다 (예전엔 여기서 False 였다)

    # 오래 안 보이면 기록을 버린다
    forgetful = TrackSmoother(window=5, min_votes=2, forget_after=3)
    box = PersonBox(xyxy=(40, 40, 60, 60), conf=0.2, in_rip=True, track_id=8)
    forgetful.apply([box])
    forgetful.apply([PersonBox(xyxy=(40, 40, 60, 60), conf=0.2, in_rip=True, track_id=8)])
    assert 8 in forgetful._history
    for _ in range(3):
        forgetful.apply([])
    assert 8 not in forgetful._history

    # 트랙 번호가 없는 박스는 손대지 않는다 (트래킹 off 일 때의 기존 동작 보존)
    untracked = PersonBox(xyxy=(40, 40, 60, 60), conf=0.8, in_rip=True)
    assert smoother.apply([untracked])[0].in_rip is True

    # window=1 이면 스무딩 자체를 건너뛴다
    passthrough = TrackSmoother(window=1, min_votes=3)
    once = PersonBox(xyxy=(40, 40, 60, 60), conf=0.8, in_rip=True, track_id=9)
    assert passthrough.apply([once])[0].in_rip is True

    print("rules self-check passed")


if __name__ == "__main__":
    _self_check()
