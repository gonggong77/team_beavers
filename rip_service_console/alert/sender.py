"""이벤트 하나를 '기록 → 발송 → 결과 반영' 순서로 처리한다 (백그라운드 비동기 워커 적용)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from alert.fcm_alert import SNAPSHOT_DIR
from alert.notifier import Notifier, SendResult
from alert.store import EventStore
from core.schemas import AlertEvent

SNAPSHOT_KEEP = 50

# ⭐️ 네트워크/디스크 지연이 영상 추론 루프를 멈추지 않도록 전용 백그라운드 스레드 풀 할당
_ALERT_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="AlertDispatcher")


def save_snapshot(image_bytes: bytes, event: AlertEvent, keep: int = SNAPSHOT_KEEP) -> Path:
    """경보 이미지를 static/snapshots/ 아래에 저장하고 event.snapshot_path 를 채운다."""
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = SNAPSHOT_DIR / f"{stamp}_{event.event_id}.jpg"
    path.write_bytes(image_bytes)
    event.snapshot_path = str(path)
    prune_snapshots(keep)
    return path


def prune_snapshots(keep: int = SNAPSHOT_KEEP) -> None:
    """최근 keep장만 남기고 오래된 스냅샷 파일을 정리한다."""
    if not SNAPSHOT_DIR.exists():
        return
    for stale in sorted(SNAPSHOT_DIR.glob("*.jpg"), reverse=True)[keep:]:
        stale.unlink(missing_ok=True)


def _do_dispatch_worker(
    event: AlertEvent,
    notifier: Notifier | None = None,
    store: EventStore | None = None,
) -> SendResult:
    """실제 DB 기록과 FCM 통신을 수행하는 백그라운드 워커 함수."""
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


def dispatch(
    event: AlertEvent,
    notifier: Notifier | None = None,
    store: EventStore | None = None,
    sync: bool = False,
) -> SendResult:
    """이벤트를 기록하고 발송한다.

    - sync=False (기본): 백그라운드 스레드 풀에 넘겨 0ms 딜레이로 즉시 리턴
    - sync=True: CLI 테스트 스크립트 등에서 즉각 결과 확인 시 사용
    """
    if sync:
        return _do_dispatch_worker(event, notifier, store)

    # ⭐️ 추론 파이프라인에서는 백그라운드로 넘겨 영상 멈춤(딜레이) 원천 차단
    _ALERT_EXECUTOR.submit(_do_dispatch_worker, event, notifier, store)
    return SendResult(ok=True, detail="background-dispatched")