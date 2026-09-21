"""업로드한 영상을 '분석하는 것처럼' 재생해 주는 화면용 피드.

원본 프로젝트에서 이 자리는 core/pipeline.py(실제 추론 + 알림 발송)가 맡는다.
UI 버전에는 모델도 알림도 없으므로, 화면 구성을 확인할 수 있도록
영상 프레임 위에 시연용 구역·사람을 그려서 같은 모양의 결과를 흘려보낸다.

세 등급이 순서대로 한 번씩 나오게 짜여 있다.
  진행률 0~35%   관찰 (구역 없음)
  진행률 35~60%  경고 (구역 있음, 안에 사람 없음)
  진행률 60~100% 긴급 (구역 안에 사람 있음 → 알림 이벤트 발생)

화면 코드가 기대하는 모양:
    for step in play(video_path):
        step.annotated_bgr / step.result / step.risk / step.progress / step.event
실제 pipeline.run() 도 같은 모양을 돌려주므로, 나중에 import 만 바꿔 끼우면 된다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from core.render import draw_overlay
from core.rules import grade, mark_persons_in_rip
from core.scenario import Beach, _box, _rip_conf, _rip_polygon, _synth_backdrop
from core.schemas import AlertEvent, FrameResult, RipRegion, RiskLevel

MAX_W = 960          # 화면에 띄우기 전에 이 너비로 줄인다 (원본이 4K 여도 느려지지 않게)
FALLBACK_FRAMES = 120  # 영상을 못 읽을 때 대신 돌릴 합성 프레임 수

_DEMO_BEACH = Beach("U", "업로드 영상", "업로드 영상", "UPLOAD", True, True, 3, seed=7)


@dataclass
class DemoStep:
    """한 프레임분 화면 갱신 단위."""

    annotated_bgr: np.ndarray
    result: FrameResult
    risk: RiskLevel
    progress: float
    event: AlertEvent | None = None


def _phase(progress: float) -> tuple[bool, int]:
    """진행률 → (구역을 그릴지, 구역 안 인원 수)."""
    if progress < 0.35:
        return False, 0
    if progress < 0.60:
        return True, 0
    if progress < 0.85:
        return True, 1
    return True, 2   # 인원이 늘어나 '인원 증가' 재발송이 한 번 더 뜨는 구간


def _make_result(frame: np.ndarray, frame_idx: int, t: float,
                 progress: float) -> FrameResult:
    h, w = frame.shape[:2]
    has_rip, inside = _phase(progress)
    seed = _DEMO_BEACH.seed

    rips: list[RipRegion] = []
    if has_rip:
        cx = w * (0.46 + 0.05 * math.sin(t * 0.5 + seed))
        rips.append(RipRegion(_rip_polygon(cx, h * 0.55, w, h, t, seed), _rip_conf(t, seed)))

    # 물가에 서 있는 사람들 (항상 구역 밖)
    persons = [
        _box(w, h, 0.14 + 0.19 * i + 0.006 * math.sin(t * 1.1 + i), 0.72 + 0.04 * (i % 2),
             0.68 + 0.05 * i)
        for i in range(_DEMO_BEACH.people_on_shore)
    ]
    # 구역 안으로 들어간 사람
    for k in range(inside):
        fx = 0.46 + 0.03 * math.sin(t * 0.9 + seed) + 0.04 * k
        fy = 0.55 + 0.02 * math.cos(t * 0.7) - 0.03 * k
        persons.append(_box(w, h, fx, fy, 0.84 - 0.04 * k))

    mark_persons_in_rip(persons, rips)
    for i, person in enumerate(persons):
        person.track_id = i + 1
    # 추론 시간 칸을 비워두면 개발자 모드 화면이 허전해서, 그럴듯한 값을 넣어 둔다.
    return FrameResult(frame_idx, t, w, h, rips, persons, infer_ms=18.0 + 6.0 * abs(math.sin(t)))


def _frames(video_path: Path | None, frame_skip: int,
            max_frames: int) -> Iterator[tuple[int, float, np.ndarray, float]]:
    """(프레임 번호, 영상 위치 초, 프레임, 진행률) 을 차례로 내놓는다."""
    capture = cv2.VideoCapture(str(video_path)) if video_path else None

    if capture is None or not capture.isOpened():
        if capture is not None:
            capture.release()
        # 영상을 못 읽어도 화면 확인은 되어야 한다 → 합성 배경으로 대체
        for i in range(FALLBACK_FRAMES):
            yield i, i / 10.0, _synth_backdrop(_DEMO_BEACH, i, (MAX_W, int(MAX_W * 9 / 16))), \
                (i + 1) / FALLBACK_FRAMES
        return

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    planned = min(max_frames, total // max(frame_skip, 1)) if total else max_frames

    idx = used = 0
    try:
        while used < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            if idx % max(frame_skip, 1) == 0:
                h, w = frame.shape[:2]
                if w > MAX_W:
                    frame = cv2.resize(frame, (MAX_W, int(h * MAX_W / w)))
                used += 1
                yield idx, idx / fps, frame, used / max(planned, 1)
            idx += 1
    finally:
        capture.release()


def play(video_path: Path | None, frame_skip: int = 2, max_frames: int = 200,
         show_foot: bool = False, source_name: str = "") -> Iterator[DemoStep]:
    """영상을 한 프레임씩 그려서 내놓는다. 추론도 발송도 하지 않는다."""
    source = source_name or (video_path.name if video_path else "demo")
    last_sent_inside = 0

    for frame_idx, t, frame, progress in _frames(video_path, frame_skip, max_frames):
        result = _make_result(frame, frame_idx, t, progress)
        risk = grade(result)
        annotated = draw_overlay(frame, result, risk, show_conf=True, show_foot=show_foot)

        event = None
        if risk == "emergency" and result.persons_in_rip > last_sent_inside:
            # 실제 앱에서는 이 자리에서 알림이 나간다. 여기서는 기록만 만든다.
            event = AlertEvent(
                risk_level=risk,
                persons_in_rip=result.persons_in_rip,
                total_persons=result.total_persons,
                rip_count=result.rip_count,
                frame_idx=frame_idx,
                timestamp_sec=t,
                video_source=source,
                trigger_kind="auto" if last_sent_inside == 0 else "escalation",
            )
            last_sent_inside = result.persons_in_rip

        yield DemoStep(annotated, result, risk, progress, event)
