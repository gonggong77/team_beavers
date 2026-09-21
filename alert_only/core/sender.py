"""이벤트 하나를 '기록 → 발송 → 결과 반영' 순서로 처리한다.

원본 core/pipeline.py 안에 흩어져 있던 발송 부분만 떼어낸 것이다.
탐지 파이프라인은 프레임마다 AlertEvent 를 만들고, 그 뒤 처리는 전부 여기와 같다.

    event = AlertEvent(...)          # 탐지 쪽이 만든다
    result = dispatch(event, notifier, store)

Streamlit 화면도, CLI 도, 나중에 합칠 파이프라인도 이 함수 하나만 부르면 된다.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from core.fcm_alert import SNAPSHOT_DIR
from core.notifier import Notifier, SendResult
from core.schemas import AlertEvent
from core.store import EventStore

SNAPSHOT_KEEP = 50   # static/snapshots/ 에 최근 몇 장까지 남길지


def save_snapshot(image_bytes: bytes, event: AlertEvent, keep: int = SNAPSHOT_KEEP) -> Path:
    """경보 이미지를 static/snapshots/ 아래에 저장하고 event.snapshot_path 를 채운다.

    파일명을 YYYYmmdd_HHMMSS_ 로 시작하게 만들어, 이름순 정렬이 곧 시간순이 되게 한다
    (파일시스템 mtime 에 기대지 않는다).
    """
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = SNAPSHOT_DIR / f"{stamp}_{event.event_id}.jpg"
    path.write_bytes(image_bytes)
    event.snapshot_path = str(path)
    prune_snapshots(keep)
    return path


def prune_snapshots(keep: int = SNAPSHOT_KEEP) -> None:
    """최근 keep장만 남기고 지운다.

    정리된 파일을 가리키는 DB 행은 남지만, 화면이 파일 존재 여부를 확인하므로
    조용히 빠질 뿐이다.
    """
    if not SNAPSHOT_DIR.exists():
        return
    for stale in sorted(SNAPSHOT_DIR.glob("*.jpg"), reverse=True)[keep:]:
        stale.unlink(missing_ok=True)


def dispatch(event: AlertEvent, notifier: Notifier | None = None,
             store: EventStore | None = None) -> SendResult:
    """이벤트를 기록하고 발송한다. 예외는 notifier 안에서 흡수된다.

    발송보다 기록이 먼저다. 발송이 실패해도 '이런 이벤트가 있었다' 는 남아야
    나중에 재전송하거나 원인을 찾을 수 있다.
    """
    if store:
        store.add(event)

    if notifier is None:
        return SendResult(ok=False, detail="notifier 가 없어 발송하지 않았습니다.")

    sent = notifier.send(event)
    event.image_url = sent.image_url
    if store:
        store.update_notify(
            event.event_id,
            "sent" if sent.ok else "failed",
            sent.detail,
            sent.image_url,
        )
    return sent
