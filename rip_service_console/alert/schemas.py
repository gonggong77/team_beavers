"""알림 파트가 받는 데이터의 모양 (팀 공통 계약).

원본 프로젝트 core/schemas.py 에서 알림에 필요한 부분만 옮겼다.
필드 이름과 의미는 원본과 같으므로, 웹 본체와 합칠 때 이 AlertEvent 를
그대로 주고받으면 발송 코드는 고칠 게 없다.

화면 렌더링 전용 타입(FrameResult / PersonBox / RipRegion)은 여기 없다.
발송에는 쓰이지 않기 때문이다.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Literal

# 위험 등급 3단계. 안드로이드 앱이 읽는 risk_level 값과 같아야 한다.
RiskLevel = Literal["watch", "warn", "emergency"]

RISK_LABEL: dict[str, str] = {
    "watch": "관찰",
    "warn": "경고",
    "emergency": "긴급",
}

RISK_ORDER: dict[str, int] = {"watch": 0, "warn": 1, "emergency": 2}

# 모든 출력에 따라붙는 안전 문구 (기획서 NFR-04)
SAFETY_NOTE = "AI 참고용 탐지 결과입니다. 현장 안전요원의 확인이 필요합니다."


@dataclass
class AlertEvent:
    """알림 파트로 넘어오는 표준 이벤트. JSON 직렬화가 항상 가능해야 한다."""

    risk_level: RiskLevel
    persons_in_rip: int
    total_persons: int
    rip_count: int
    frame_idx: int
    timestamp_sec: float
    video_source: str
    snapshot_path: str = ""
    # "auto" = 긴급 연속 판정, "escalation" = 구역 내 인원이 직전 발송보다 늘어남.
    # 필드명을 trigger 로 두면 안 된다 — EventStore 가 키를 그대로 컬럼명에 쓰는데
    # TRIGGER 는 SQLite 예약어라 INSERT 구문이 깨진다.
    trigger_kind: str = "auto"
    # 발송 시 앱에 전달한 스냅샷 URL. 폰이 받은 것과 같은 값이다.
    image_url: str = ""
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    occurred_at: str = field(
        default_factory=lambda: datetime.now().astimezone().isoformat(timespec="seconds")
    )
    note: str = SAFETY_NOTE

    def to_dict(self) -> dict:
        return asdict(self)

    def summary_text(self) -> str:
        """기본 메시지 문안. 알림 파트에서 자유롭게 교체해도 된다."""
        return (
            f"[{RISK_LABEL[self.risk_level]}] 이안류 구역 내 인원 {self.persons_in_rip}명 감지\n"
            f"발생 시각: {self.occurred_at}\n"
            f"영상 위치: {self.timestamp_sec:.1f}초 (프레임 {self.frame_idx})\n"
            f"화면 전체 인원: {self.total_persons}명 / 의심 구역 {self.rip_count}개\n"
            f"영상: {self.video_source}\n"
            f"\n{self.note}"
        )
