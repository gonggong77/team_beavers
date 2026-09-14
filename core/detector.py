"""모델 어댑터.

웹 코드는 Detector 프로토콜만 알고 있으면 된다.
모델 파일이 없으면 FakeDetector, 있으면 YoloDetector가 자동으로 선택된다.
모델을 받으면 models/ 폴더에 넣고 .env의 RIP_MODEL_PATH만 바꾸면 끝난다.
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Protocol, Sequence, runtime_checkable

import numpy as np

from core.rules import mark_persons_in_rip
from core.schemas import FrameResult, PersonBox, RipRegion

DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.7
DEFAULT_IMGSZ = 640


@runtime_checkable
class Detector(Protocol):
    """모든 탐지기가 지켜야 할 최소 계약."""

    names: dict[int, str]

    def predict(
        self,
        frame_bgr: np.ndarray,
        frame_idx: int,
        timestamp_sec: float,
        rip_ids: Sequence[int],
        person_ids: Sequence[int],
    ) -> FrameResult: ...


class FakeDetector:
    """모델이 오기 전까지 UI를 완성하기 위한 더미 탐지기.

    frame_idx만 보고 결정론적으로 움직이는 이안류 구역과 사람을 만들어낸다.
    약 6초 주기로 사람 한 명이 구역 안으로 들어갔다 나오므로
    관찰 -> 경고 -> 긴급 전환과 알림 발송까지 전부 검증할 수 있다.
    """

    names = {0: "rip_current", 1: "person"}

    def __init__(self, fps_hint: float = 30.0, seed: int = 7):
        self.fps_hint = max(fps_hint, 1.0)
        self._rng = np.random.default_rng(seed)

    def predict(self, frame_bgr, frame_idx, timestamp_sec, rip_ids, person_ids) -> FrameResult:
        started = time.perf_counter()
        h, w = frame_bgr.shape[:2]

        rips: list[RipRegion] = []
        # 12초 주기 중 앞 9초 동안만 이안류 구역이 보이도록 (관찰 단계를 만들기 위함)
        if (timestamp_sec % 12.0) < 9.0:
            cx = w * (0.45 + 0.06 * math.sin(timestamp_sec * 0.6))
            cy = h * 0.58
            rx, ry = w * 0.13, h * 0.20
            polygon = [
                (
                    cx + rx * math.cos(a) * (1.0 + 0.18 * math.sin(a * 3 + timestamp_sec)),
                    cy + ry * math.sin(a),
                )
                for a in np.linspace(0, 2 * math.pi, 24, endpoint=False)
            ]
            rips.append(RipRegion(polygon=polygon, conf=0.62 + 0.1 * math.sin(timestamp_sec)))

        persons: list[PersonBox] = []
        # 고정된 구경꾼 3명
        for i, (fx, fy) in enumerate([(0.14, 0.80), (0.78, 0.74), (0.62, 0.86)]):
            jitter = 0.008 * math.sin(timestamp_sec * 1.3 + i)
            persons.append(self._box(w, h, fx + jitter, fy, 0.70 + 0.05 * i))

        # 구역 안팎을 오가는 물놀이객 1명
        phase = (timestamp_sec % 12.0) / 12.0
        swim_x = 0.20 + 0.30 * math.sin(phase * 2 * math.pi) + 0.25
        persons.append(self._box(w, h, swim_x, 0.58, 0.81))

        mark_persons_in_rip(persons, rips)
        return FrameResult(
            frame_idx=frame_idx,
            timestamp_sec=timestamp_sec,
            width=w,
            height=h,
            rips=rips,
            persons=persons,
            infer_ms=(time.perf_counter() - started) * 1000,
        )

    @staticmethod
    def _box(w: int, h: int, fx: float, fy: float, conf: float) -> PersonBox:
        bw, bh = w * 0.025, h * 0.06
        cx, cy = w * fx, h * fy
        return PersonBox(xyxy=(cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2), conf=conf)


class YoloDetector:
    """Ultralytics YOLO 어댑터.

    좌표계 주의:
      boxes.xyxy 와 masks.xy 는 이미 원본 프레임 픽셀 좌표로 변환되어 나온다.
      masks.data 는 마스크 해상도 기준이라 원본 크기와 다르므로 사용하지 않는다.
    """

    def __init__(
        self,
        weights: str | Path,
        conf: float = DEFAULT_CONF,
        iou: float = DEFAULT_IOU,
        imgsz: int = DEFAULT_IMGSZ,
        device: str | None = None,
    ):
        from ultralytics import YOLO  # 무거우므로 지연 임포트

        self.weights = str(weights)
        self.model = YOLO(self.weights)
        self.names = dict(self.model.names)
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = device

    def predict(self, frame_bgr, frame_idx, timestamp_sec, rip_ids, person_ids) -> FrameResult:
        started = time.perf_counter()
        h, w = frame_bgr.shape[:2]

        kwargs = dict(conf=self.conf, iou=self.iou, imgsz=self.imgsz, verbose=False)
        if self.device:
            kwargs["device"] = self.device
        result = self.model.predict(frame_bgr, **kwargs)[0]

        rips: list[RipRegion] = []
        persons: list[PersonBox] = []

        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            classes = boxes.cls.int().tolist()
            confs = boxes.conf.tolist()
            xyxy = boxes.xyxy.tolist()
            # 탐지가 0건이면 result.masks 가 None 이다. 가장 흔한 크래시 지점.
            polygons = result.masks.xy if result.masks is not None else [None] * len(classes)

            rip_set, person_set = set(rip_ids), set(person_ids)
            for i, cls_id in enumerate(classes):
                if cls_id in rip_set:
                    poly = polygons[i] if i < len(polygons) else None
                    if poly is None or len(poly) < 3:
                        continue  # seg 모델이 아니면 이안류 구역을 만들 수 없다
                    rips.append(RipRegion(polygon=[tuple(p) for p in np.asarray(poly).tolist()], conf=float(confs[i])))
                elif cls_id in person_set:
                    persons.append(PersonBox(xyxy=tuple(xyxy[i]), conf=float(confs[i])))

        mark_persons_in_rip(persons, rips)
        return FrameResult(
            frame_idx=frame_idx,
            timestamp_sec=timestamp_sec,
            width=w,
            height=h,
            rips=rips,
            persons=persons,
            infer_ms=(time.perf_counter() - started) * 1000,
        )


def resolve_weights(explicit: str | None = None) -> Path | None:
    """가중치 경로를 찾는다. 없으면 None."""
    candidate = explicit or os.getenv("RIP_MODEL_PATH", "")
    if candidate:
        path = Path(candidate)
        return path if path.exists() else None

    models_dir = Path(__file__).resolve().parent.parent / "models"
    for pattern in ("*.pt", "*.onnx"):
        found = sorted(models_dir.glob(pattern))
        if found:
            return found[0]
    return None


def build_detector(
    weights: str | Path | None = None,
    conf: float = DEFAULT_CONF,
    iou: float = DEFAULT_IOU,
    imgsz: int = DEFAULT_IMGSZ,
) -> tuple[Detector, bool]:
    """(탐지기, 실제_모델_여부)를 돌려준다."""
    path = resolve_weights(str(weights) if weights else None)
    if path is None:
        return FakeDetector(), False
    return YoloDetector(path, conf=conf, iou=iou, imgsz=imgsz), True
