"""관제 화면에 흘려보낼 영상과 표시 데이터를 만든다 (시연용 값).

원본 프로젝트의 core/scenario.py 에서 모델을 쓰지 않는 'scenario' 모드만 남긴 것이다.
실제 추론 경로(hybrid / real)는 이 UI 버전에 들어 있지 않다.

  - 영상 파일이 있으면 그 프레임을 배경으로 쓴다
  - 없으면 합성 해변 배경을 그려서라도 화면이 비지 않게 한다

실제 해수욕장 이름은 쓰지 않는다. 오탐이 실제 지명과 엮이면 안 되기 때문이다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from core.rules import grade, mark_persons_in_rip
from core.schemas import FrameResult, PersonBox, RipRegion, RiskLevel

_HERE = Path(__file__).resolve().parent.parent

# 이 폴더 안의 data/cctv/ 를 먼저 보고, 없으면 원본 프로젝트의 data/cctv/ 를 본다.
# 둘 다 없으면 합성 배경으로 돌아간다 (영상 없이도 화면은 뜬다).
CCTV_DIR = next(
    (p for p in (_HERE / "data" / "cctv", _HERE.parent / "data" / "cctv") if p.exists()),
    _HERE / "data" / "cctv",
)

PANEL_W, PANEL_H = 480, 270      # 작은 썸네일용
MAIN_W, MAIN_H = 720, 405        # 메인 뷰어용
MAIN_FRAMES = 24          # 메인 뷰어가 구역마다 메모리에 올릴 프레임 수
BUFFER_FRAMES = 60        # 순환 재생할 프레임 수

MAIN_BEACH = "A"          # 순환을 끄면 이 구역만 띄운다
ROTATE_SEC = 10.0         # 구역 순환 간격(초)


@dataclass
class Beach:
    """가상 관측구역 하나."""

    code: str                 # "A"
    name: str                 # "A 해변"
    zone: str                 # "가상 관측구역 A-1"
    camera_id: str            # "CCTV-A01"
    has_rip: bool             # 이안류 의심 구역 존재 여부
    person_in_rip: bool       # 구역 안에 사람이 있는지
    people_on_shore: int = 3  # 구역 밖에 그릴 인원
    seed: int = 0

    @property
    def expected_risk(self) -> RiskLevel:
        if not self.has_rip:
            return "watch"
        return "emergency" if self.person_in_rip else "warn"


# 위험 등급 3단계가 한 화면에 모두 보이도록 구성했다.
BEACHES: list[Beach] = [
    Beach("A", "A 해변", "가상 관측구역 A-1", "CCTV-A01", True, True, 3, seed=11),
    Beach("B", "B 해변", "가상 관측구역 B-2", "CCTV-B02", False, False, 2, seed=23),
    Beach("C", "C 해변", "가상 관측구역 C-1", "CCTV-C01", True, False, 4, seed=37),
    Beach("D", "D 해변", "가상 관측구역 D-3", "CCTV-D03", False, False, 3, seed=41),
    Beach("E", "E 해변", "가상 관측구역 E-1", "CCTV-E01", True, False, 2, seed=59),
]


def resolve_sources() -> dict[str, Path | None]:
    """data/cctv/ 에서 해변별 영상 파일을 찾는다. 없으면 None."""
    sources: dict[str, Path | None] = {}
    for beach in BEACHES:
        found = None
        if CCTV_DIR.exists():
            for ext in (".mp4", ".avi", ".mov", ".mkv"):
                candidate = CCTV_DIR / f"{beach.code}{ext}"
                if candidate.exists():
                    found = candidate
                    break
        sources[beach.code] = found
    return sources


# --------------------------------------------------------------------------
# 배경 프레임
# --------------------------------------------------------------------------

def load_frames(beach: Beach, video: Path | None, count: int,
                size: tuple[int, int] = (PANEL_W, PANEL_H)) -> list[np.ndarray]:
    """영상에서 count개 프레임만 메모리에 올린다. 영상이 없으면 합성 배경을 만든다."""
    if video is None:
        return [_synth_backdrop(beach, i, size) for i in range(count)]

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        return [_synth_backdrop(beach, i, size) for i in range(count)]

    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(total // count, 1) if total else 1
    frames, idx = [], 0
    try:
        while len(frames) < count:
            ok, frame = capture.read()
            if not ok:
                break
            if idx % step == 0:
                frames.append(cv2.resize(frame, size))
            idx += 1
    finally:
        capture.release()

    return frames or [_synth_backdrop(beach, i, size) for i in range(count)]


def _synth_backdrop(beach: Beach, tick: int,
                    size: tuple[int, int] = (PANEL_W, PANEL_H)) -> np.ndarray:
    """영상 파일이 없을 때 쓰는 합성 해변 배경."""
    w, h = size
    rng = np.random.default_rng(beach.seed)
    img = np.zeros((h, w, 3), np.uint8)

    horizon = int(h * 0.30)
    sand_line = int(h * 0.78)

    for y in range(horizon):
        f = y / max(horizon, 1)
        img[y, :] = (205 - int(25 * f), 178 - int(18 * f), 132 - int(10 * f))

    tone = int(rng.integers(-28, 28))
    for y in range(horizon, sand_line):
        f = (y - horizon) / max(sand_line - horizon, 1)
        img[y, :] = (152 + int(28 * f) + tone, 108 + int(22 * f), 58 + int(16 * f))

    img[sand_line:] = (146 + tone // 2, 192, 216)

    for k in range(6):
        phase = tick * 0.12 + k * 0.9 + beach.seed
        y = int(horizon + (sand_line - horizon) * (0.14 + 0.14 * k) + 4 * math.sin(phase))
        for x in range(0, w, 4):
            amp = 2.0 + 1.4 * math.sin(x * 0.045 + phase)
            cv2.line(img, (x, y), (x + 3, int(y + amp)), (205, 190, 170), 1, cv2.LINE_AA)

    return img


# --------------------------------------------------------------------------
# 표시용 구역·사람 생성
# --------------------------------------------------------------------------

def _rip_polygon(cx: float, cy: float, w: int, h: int, t: float,
                 seed: int) -> list[tuple[float, float]]:
    rx, ry = w * 0.15, h * 0.19
    return [
        (
            cx + rx * math.cos(a) * (1.0 + 0.16 * math.sin(a * 3 + t * 0.8 + seed)),
            cy + ry * math.sin(a) * (1.0 + 0.10 * math.cos(a * 2 + t * 0.6)),
        )
        for a in np.linspace(0, 2 * math.pi, 22, endpoint=False)
    ]


def _rip_conf(t: float, seed: int) -> float:
    return 0.58 + 0.12 * abs(math.sin(t * 0.4 + seed))


def _box(w: int, h: int, fx: float, fy: float, conf: float) -> PersonBox:
    bw, bh = w * 0.030, h * 0.070
    cx, cy = w * fx, h * fy
    return PersonBox(xyxy=(cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2), conf=conf)


def scenario_result(beach: Beach, frame, frame_idx: int, t: float) -> FrameResult:
    """구역과 사람을 시연용 값으로 만든다. 모델도 영상도 없이 화면을 채운다."""
    h, w = frame.shape[:2]

    rips: list[RipRegion] = []
    if beach.has_rip:
        cx = w * (0.46 + 0.05 * math.sin(t * 0.5 + beach.seed))
        cy = h * 0.55
        rips.append(RipRegion(_rip_polygon(cx, cy, w, h, t, beach.seed), _rip_conf(t, beach.seed)))

    persons: list[PersonBox] = []
    rng = np.random.default_rng(beach.seed)
    for i in range(beach.people_on_shore):
        fx = float(rng.uniform(0.08, 0.92)) + 0.006 * math.sin(t * 1.1 + i)
        fy = float(rng.uniform(0.68, 0.90))
        persons.append(_box(w, h, fx, fy, 0.68 + 0.06 * i))

    if beach.has_rip and beach.person_in_rip:
        fx = 0.46 + 0.03 * math.sin(t * 0.9 + beach.seed)
        fy = 0.55 + 0.02 * math.cos(t * 0.7)
        persons.append(_box(w, h, fx, fy, 0.84))

    mark_persons_in_rip(persons, rips)
    return FrameResult(frame_idx, t, w, h, rips, persons)


# --------------------------------------------------------------------------
# 화면이 사용하는 단일 인터페이스
# --------------------------------------------------------------------------

class BeachStream:
    """한 해변의 프레임과 표시 데이터를 tick 으로 조회한다.

    화면은 `.at(tick)` 만 호출하면 된다. 나중에 실제 추론을 붙일 때도
    이 메서드의 반환 모양(프레임, FrameResult, 등급)만 지키면 화면은 그대로 쓴다.
    """

    def __init__(self, beach: Beach, video: Path | None,
                 size: tuple[int, int] = (PANEL_W, PANEL_H)):
        self.beach = beach
        self.mode = "scenario"
        count = min(BUFFER_FRAMES, MAIN_FRAMES) if size == (MAIN_W, MAIN_H) else BUFFER_FRAMES
        self.frames = load_frames(beach, video, count, size)

    def __len__(self) -> int:
        return len(self.frames)

    def at(self, tick: int) -> tuple[np.ndarray, FrameResult, RiskLevel]:
        i = tick % len(self.frames)
        frame = self.frames[i].copy()
        result = scenario_result(self.beach, frame, tick, tick * 0.7)
        return frame, result, grade(result)


def build_streams(size: tuple[int, int] = (PANEL_W, PANEL_H)) -> dict[str, BeachStream]:
    sources = resolve_sources()
    return {b.code: BeachStream(b, sources[b.code], size) for b in BEACHES}


def build_main_streams() -> list[BeachStream]:
    """A~E 전체를 메인 해상도로 준비한다. 순환 재생용이며 BEACHES 순서를 따른다."""
    sources = resolve_sources()
    return [BeachStream(b, sources[b.code], (MAIN_W, MAIN_H)) for b in BEACHES]


def rotating_index(elapsed_sec: float, count: int, interval: float = ROTATE_SEC) -> int:
    """경과 시간으로 지금 보여줄 구역의 순번을 정한다."""
    return int(elapsed_sec / max(interval, 0.1)) % max(count, 1)
