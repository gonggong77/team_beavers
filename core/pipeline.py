"""영상 처리 파이프라인.

Streamlit에 의존하지 않는다. CLI로도 똑같이 돌릴 수 있어야
알림 담당자와 모델 담당자가 UI 없이 검증할 수 있다.

    python -m core.pipeline data/demo.mp4 --skip 5
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator, Sequence

import cv2
import numpy as np

from core.detector import Detector, build_detector
from core.notifier import Notifier, build_notifier
from core.render import draw_overlay
from core.rules import AlertGate, grade
from core.schemas import AlertEvent, FrameResult, RiskLevel
from core.store import EventStore

SNAPSHOT_DIR = Path(__file__).resolve().parent.parent / "data" / "snapshots"


@dataclass
class PipelineConfig:
    rip_ids: Sequence[int] = (0,)
    person_ids: Sequence[int] = (1,)
    frame_skip: int = 5           # N프레임마다 1회 추론
    max_frames: int | None = None
    min_consecutive: int = 3
    cooldown_sec: float = 30.0
    save_snapshot: bool = True


@dataclass
class Step:
    """한 번의 추론 결과 묶음. UI는 이걸 받아 화면만 그린다."""

    result: FrameResult
    risk: RiskLevel
    annotated_bgr: np.ndarray
    event: AlertEvent | None = None
    progress: float = 0.0


def iter_video(path: str | Path, frame_skip: int = 1, max_frames: int | None = None):
    """(frame_bgr, frame_idx, timestamp_sec, total_frames)를 순회한다."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"영상을 열 수 없습니다: {path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    idx, emitted = 0, 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if idx % max(frame_skip, 1) == 0:
                yield frame, idx, idx / fps, total
                emitted += 1
                if max_frames and emitted >= max_frames:
                    break
            idx += 1
    finally:
        capture.release()


def run(
    video_path: str | Path,
    detector: Detector,
    config: PipelineConfig,
    store: EventStore | None = None,
    notifier: Notifier | None = None,
) -> Iterator[Step]:
    """프레임마다 Step을 yield한다. 제너레이터라 UI가 진행 상황을 실시간으로 그릴 수 있다."""
    gate = AlertGate(min_consecutive=config.min_consecutive, cooldown_sec=config.cooldown_sec)
    source_name = Path(video_path).name
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

    for frame, frame_idx, timestamp_sec, total in iter_video(
        video_path, config.frame_skip, config.max_frames
    ):
        result = detector.predict(frame, frame_idx, timestamp_sec, config.rip_ids, config.person_ids)
        risk = grade(result)
        annotated = draw_overlay(frame, result, risk)

        event = None
        if gate.should_alert(risk, timestamp_sec):
            event = AlertEvent(
                risk_level=risk,
                persons_in_rip=result.persons_in_rip,
                total_persons=result.total_persons,
                rip_count=result.rip_count,
                frame_idx=frame_idx,
                timestamp_sec=timestamp_sec,
                video_source=source_name,
            )
            if config.save_snapshot:
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                snapshot = SNAPSHOT_DIR / f"{stamp}_{event.event_id}.jpg"
                cv2.imwrite(str(snapshot), annotated)
                event.snapshot_path = str(snapshot)

            if store:
                store.add(event)
            if notifier:
                sent = notifier.send(event)
                if store:
                    store.update_notify(event.event_id, "sent" if sent.ok else "failed", sent.detail)

        yield Step(
            result=result,
            risk=risk,
            annotated_bgr=annotated,
            event=event,
            progress=(frame_idx / total) if total else 0.0,
        )


def _cli() -> None:
    parser = argparse.ArgumentParser(description="이안류 탐지 파이프라인 (UI 없이 실행)")
    parser.add_argument("video")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--skip", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--notifier", default="console", choices=["console", "telegram"])
    args = parser.parse_args()

    detector, is_real = build_detector(args.weights)
    print(f"detector={'YOLO' if is_real else 'Fake(데모 모드)'} names={getattr(detector, 'names', {})}")

    config = PipelineConfig(frame_skip=args.skip, max_frames=args.max_frames)
    store = EventStore()
    notifier = build_notifier(args.notifier)

    alerts = 0
    for step in run(args.video, detector, config, store, notifier):
        if step.event:
            alerts += 1
        print(
            f"frame {step.result.frame_idx:>6} t={step.result.timestamp_sec:6.2f}s "
            f"{step.risk:<9} in-zone={step.result.persons_in_rip} "
            f"total={step.result.total_persons} {'ALERT' if step.event else ''}"
        )
    print(f"\n완료. 발송 이벤트 {alerts}건")


if __name__ == "__main__":
    _cli()
