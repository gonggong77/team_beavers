"""팀 공통 데이터 계약.

이 파일은 웹(Streamlit), 모델, 알림 세 파트가 모두 참조하는 유일한 계약서다.
여기 정의된 필드를 바꿀 때는 반드시 팀 전체에 공유한다.

- FrameResult : 화면 렌더링용. numpy 배열을 포함할 수 있어 직렬화하지 않는다.
- AlertEvent  : 알림/로그용. JSON 직렬화가 항상 가능해야 한다.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Literal

# FR-03 위험 등급 3단계
RiskLevel = Literal["watch", "warn", "emergency"]

RISK_LABEL: dict[str, str] = {
    "watch": "관찰",
    "warn": "경고",
    "emergency": "긴급",
}

# BGR 순서 (OpenCV 기준)
RISK_COLOR_BGR: dict[str, tuple[int, int, int]] = {
    "watch": (150, 150, 150),
    "warn": (0, 170, 255),
    "emergency": (0, 0, 255),
}

RISK_ORDER: dict[str, int] = {"watch": 0, "warn": 1, "emergency": 2}

# 모든 출력에 따라붙는 안전 문구 (기획서 NFR-04)
SAFETY_NOTE = "AI 참고용 탐지 결과입니다. 현장 안전요원의 확인이 필요합니다."


@dataclass
class PersonBox:
    """탐지된 사람 한 명."""

    xyxy: tuple[float, float, float, float]  # 원본 프레임 픽셀 좌표
    conf: float
    in_rip: bool = False

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def bottom_center(self) -> tuple[float, float]:
        x1, _, x2, y2 = self.xyxy
        return ((x1 + x2) / 2.0, y2)


@dataclass
class RipRegion:
    """이안류 의심 구역 한 개."""

    polygon: list[tuple[float, float]]  # 원본 프레임 픽셀 좌표
    conf: float


@dataclass
class FrameResult:
    """한 프레임의 추론 결과. 화면 렌더링 전용."""

    frame_idx: int
    timestamp_sec: float
    width: int
    height: int
    rips: list[RipRegion] = field(default_factory=list)
    persons: list[PersonBox] = field(default_factory=list)
    infer_ms: float = 0.0

    @property
    def persons_in_rip(self) -> int:
        return sum(1 for p in self.persons if p.in_rip)

    @property
    def total_persons(self) -> int:
        return len(self.persons)

    @property
    def rip_count(self) -> int:
        return len(self.rips)


@dataclass
class AlertEvent:
    """알림 파트로 넘기는 표준 이벤트. JSON 직렬화 가능해야 한다."""

    risk_level: RiskLevel
    persons_in_rip: int
    total_persons: int
    rip_count: int
    frame_idx: int
    timestamp_sec: float
    video_source: str
    snapshot_path: str = ""
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    occurred_at: str = field(default_factory=lambda: datetime.now().astimezone().isoformat(timespec="seconds"))
    note: str = SAFETY_NOTE

    def to_dict(self) -> dict:
        return asdict(self)

    def summary_text(self) -> str:
        """텔레그램 메시지 기본 문안. 알림 파트에서 자유롭게 교체해도 된다."""
        return (
            f"[{RISK_LABEL[self.risk_level]}] 이안류 구역 내 인원 {self.persons_in_rip}명 감지\n"
            f"발생 시각: {self.occurred_at}\n"
            f"영상 위치: {self.timestamp_sec:.1f}초 (프레임 {self.frame_idx})\n"
            f"화면 전체 인원: {self.total_persons}명 / 의심 구역 {self.rip_count}개\n"
            f"영상: {self.video_source}\n"
            f"\n{self.note}"
        )
