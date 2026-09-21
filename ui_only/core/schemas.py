"""화면이 그리는 데이터의 모양.

원본 프로젝트(core/schemas.py)의 팀 공통 계약에서 화면에 필요한 것만 옮겼다.
필드 이름과 의미는 원본과 같으므로, 나중에 실제 모델·알림과 합칠 때
이 dataclass 를 그대로 받아 쓰면 화면 코드는 고칠 게 없다.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Literal

# 위험 등급 3단계
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

# 모든 출력에 따라붙는 안전 문구
SAFETY_NOTE = "AI 참고용 탐지 결과입니다. 현장 안전요원의 확인이 필요합니다."


@dataclass
class PersonBox:
    """화면에 그릴 사람 한 명."""

    xyxy: tuple[float, float, float, float]  # 원본 프레임 픽셀 좌표
    conf: float
    in_rip: bool = False
    track_id: int | None = None

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
    """한 프레임분 화면 데이터."""

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
    """이벤트 기록 표에 한 줄로 들어가는 데이터.

    이 UI 버전에서는 실제로 어디에도 발송하지 않는다.
    화면 표와 갤러리를 채우는 용도로만 만든다.
    """

    risk_level: RiskLevel
    persons_in_rip: int
    total_persons: int
    rip_count: int
    frame_idx: int
    timestamp_sec: float
    video_source: str
    trigger_kind: str = "auto"      # "auto" = 연속 긴급 판정, "escalation" = 인원 증가
    notify_status: str = "ui-demo"  # 발송 기능이 빠져 있으므로 항상 이 값
    image_url: str = ""
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    occurred_at: str = field(
        default_factory=lambda: datetime.now().astimezone().isoformat(timespec="seconds")
    )
    note: str = SAFETY_NOTE

    def to_dict(self) -> dict:
        return asdict(self)
