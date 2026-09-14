"""상시 관제 화면의 영상 공급과 탐지.

세 가지 모드를 지원한다. 화면 코드는 셋 다 똑같이 `.at(tick)` 만 호출한다.

  scenario : 이안류도 사람도 전부 가짜. 모델과 영상이 없어도 돌아간다
  hybrid   : 사람은 실제 탐지, 이안류 구역만 시나리오. 모델 오기 전 권장
  real     : 이안류와 사람 모두 실제 추론. 모델을 받은 뒤 사용

hybrid와 real은 첫 로딩 때 미리 추론해서 결과를 메모리에 담아둔다(사전 렌더링).
매 프레임 추론하면 5분할 화면이 CPU에서 버티지 못한다.

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

CCTV_DIR = Path(__file__).resolve().parent.parent / "data" / "cctv"

PANEL_W, PANEL_H = 480, 270      # 작은 썸네일용
MAIN_W, MAIN_H = 720, 405        # 메인 뷰어용
MAIN_FRAMES = 24          # 메인 뷰어가 구역마다 메모리에 올릴 프레임 수
                          # A~E 5개를 전부 올리므로 늘리면 메모리가 그만큼 커진다
BUFFER_FRAMES = 60        # scenario 모드에서 순환 재생할 프레임 수
INFER_FRAMES = 40         # hybrid/real 모드에서 사전 추론할 프레임 수 (늘리면 로딩이 길어진다)

MAIN_BEACH = "A"          # 순환을 끄면 이 구역만 띄운다
ROTATE_SEC = 10.0         # 구역 순환 간격(초)

MODES = ("scenario", "hybrid", "real")
MODE_LABEL = {
    "scenario": "시나리오 (모델 없음)",
    "hybrid": "사람만 실제 탐지",
    "real": "전체 실제 추론",
}


@dataclass
class Beach:
    """가상 관측구역 하나."""

    code: str                 # "A"
    name: str                 # "A 해변"
    zone: str                 # "가상 관측구역 A-1"
    camera_id: str            # "CCTV-A01"
    has_rip: bool             # 이안류 의심 구역 존재 여부
    person_in_rip: bool       # 구역 안에 사람이 있는지
    people_on_shore: int = 3  # scenario 모드에서 그릴 구역 밖 인원
    seed: int = 0

    @property
    def expected_risk(self) -> RiskLevel:
        if not self.has_rip:
            return "watch"
        return "emergency" if self.person_in_rip else "warn"


# 기획서 FR-03 기준으로 3단계가 한 화면에 모두 보이도록 구성했다.
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
# 이안류 구역 생성 (scenario / hybrid 공용)
# --------------------------------------------------------------------------

def _rip_polygon(cx: float, cy: float, w: int, h: int, t: float, seed: int) -> list[tuple[float, float]]:
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


# --------------------------------------------------------------------------
# 모드별 결과 생성
# --------------------------------------------------------------------------

def scenario_result(beach: Beach, frame, frame_idx: int, t: float) -> FrameResult:
    """전부 가짜. 모델도 영상도 없이 화면을 채운다."""
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


def hybrid_result(beach: Beach, detector, person_ids, frame, frame_idx: int, t: float) -> FrameResult:
    """사람은 실제 탐지, 이안류 구역만 시나리오.

    구역 위치를 실제 탐지 결과에 맞춰 잡는 것이 핵심이다.
      person_in_rip=True  -> 실제로 탐지된 사람 위에 구역을 그린다 (A 해변)
      person_in_rip=False -> 탐지된 사람에게서 가장 먼 수면에 구역을 그린다 (C, E 해변)
    덕분에 빈 바다에 박스가 떠 있거나, 구역 안에 아무도 없는데 긴급이 뜨는 일이 없다.
    """
    detected = detector.predict(frame, frame_idx, t, (), person_ids)
    h, w = detected.height, detected.width
    persons = detected.persons

    rips: list[RipRegion] = []
    if beach.has_rip:
        cx, cy = _pick_rip_center(beach, persons, w, h)
        rips.append(RipRegion(_rip_polygon(cx, cy, w, h, t, beach.seed), _rip_conf(t, beach.seed)))

    mark_persons_in_rip(persons, rips)
    return FrameResult(frame_idx, t, w, h, rips, persons, detected.infer_ms)


def _pick_rip_center(beach: Beach, persons, w: int, h: int) -> tuple[float, float]:
    water = [p for p in persons if p.center[1] < h * 0.80]  # 수면 쪽에 있는 사람만 후보

    if beach.person_in_rip:
        if water:  # 화면 중앙에 가장 가까운 사람 위에 구역을 얹는다
            return min(water, key=lambda p: abs(p.center[0] - w / 2)).center
        return w * 0.46, h * 0.55  # 탐지가 없으면 기본 위치

    # 사람이 없어야 하는 구역: 후보 지점 중 사람과 가장 먼 곳을 고른다
    candidates = [(w * fx, h * 0.50) for fx in (0.20, 0.35, 0.50, 0.65, 0.80)]
    if not persons:
        return candidates[2]
    return max(candidates, key=lambda pt: min(math.dist(pt, p.center) for p in persons))


def real_result(beach: Beach, detector, rip_ids, person_ids, frame, frame_idx: int, t: float) -> FrameResult:
    """이안류와 사람 모두 실제 추론. 모델을 받은 뒤 이 경로를 쓴다."""
    return detector.predict(frame, frame_idx, t, rip_ids, person_ids)


# --------------------------------------------------------------------------
# 화면이 사용하는 단일 인터페이스
# --------------------------------------------------------------------------

class BeachStream:
    """한 해변의 프레임과 탐지 결과를 tick으로 조회한다.

    scenario 모드는 매번 계산하고(가벼움), hybrid/real 모드는 생성 시 한 번만 추론한다.
    화면은 모드를 몰라도 되고 `.at(tick)` 만 호출하면 된다.
    """

    def __init__(self, beach: Beach, video: Path | None, mode: str = "scenario",
                 detector=None, rip_ids=(), person_ids=(),
                 size: tuple[int, int] = (PANEL_W, PANEL_H)):
        self.beach = beach
        self.mode = mode if mode in MODES else "scenario"
        if self.mode != "scenario" and detector is None:
            self.mode = "scenario"

        count = BUFFER_FRAMES if self.mode == "scenario" else INFER_FRAMES
        if size == (MAIN_W, MAIN_H):
            count = min(count, MAIN_FRAMES)
        self.frames = load_frames(beach, video, count, size)
        self._cache: list[FrameResult] | None = None

        if self.mode != "scenario":
            self._cache = []
            for i, frame in enumerate(self.frames):
                t = i * 0.7
                if self.mode == "hybrid":
                    self._cache.append(hybrid_result(beach, detector, person_ids, frame, i, t))
                else:
                    self._cache.append(real_result(beach, detector, rip_ids, person_ids, frame, i, t))

    def __len__(self) -> int:
        return len(self.frames)

    def at(self, tick: int) -> tuple[np.ndarray, FrameResult, RiskLevel]:
        i = tick % len(self.frames)
        frame = self.frames[i].copy()
        result = self._cache[i] if self._cache is not None else scenario_result(
            self.beach, frame, tick, tick * 0.7
        )
        return frame, result, grade(result)


def build_streams(mode: str = "scenario", detector=None, rip_ids=(), person_ids=(),
                  size: tuple[int, int] = (PANEL_W, PANEL_H)) -> dict[str, BeachStream]:
    sources = resolve_sources()
    return {
        b.code: BeachStream(b, sources[b.code], mode, detector, rip_ids, person_ids, size)
        for b in BEACHES
    }


def build_main_stream(mode: str = "scenario", detector=None, rip_ids=(), person_ids=(),
                      code: str = MAIN_BEACH) -> BeachStream:
    """상시 화면용 단일 스트림."""
    beach = next(b for b in BEACHES if b.code == code)
    video = resolve_sources()[beach.code]
    return BeachStream(beach, video, mode, detector, rip_ids, person_ids, (MAIN_W, MAIN_H))


def build_main_streams(mode: str = "scenario", detector=None,
                       rip_ids=(), person_ids=()) -> list[BeachStream]:
    """A~E 전체를 메인 해상도로 준비한다. 순환 재생용이며 BEACHES 순서를 따른다."""
    sources = resolve_sources()
    return [
        BeachStream(b, sources[b.code], mode, detector, rip_ids, person_ids, (MAIN_W, MAIN_H))
        for b in BEACHES
    ]


def rotating_index(elapsed_sec: float, count: int, interval: float = ROTATE_SEC) -> int:
    """경과 시간으로 지금 보여줄 구역의 순번을 정한다."""
    return int(elapsed_sec / max(interval, 0.1)) % max(count, 1)
