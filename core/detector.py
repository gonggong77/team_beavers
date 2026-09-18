"""모델 어댑터.

웹 코드는 Detector 프로토콜만 알고 있으면 된다.
모델은 이안류 탐지 모델(segment)과 사람(표류자) 탐지 모델(detect), 두 개를 따로 돌려
DualDetector가 결과를 합친다. 한쪽 모델 파일이 없으면 그 쪽만 FakeDetector로 대체된다.

이안류 모델은 models/rip/ (또는 .env의 RIP_MODEL_PATH)에,
사람 모델은 models/person/ (또는 .env의 PERSON_MODEL_PATH)에 넣으면 끝난다.
기존처럼 models/ 바로 아래에 이안류 모델을 둔 경우도 계속 인식한다.
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Protocol, Sequence, runtime_checkable

import numpy as np

from core.model_probe import PERSON_HINTS, RIP_HINTS, match_class_ids, matches_hint
from core.rules import mark_persons_in_rip
from core.schemas import FrameResult, PersonBox, RipRegion

# 이안류 구역과 원거리 CCTV의 작은 사람은 적정 conf 임계값이 서로 다르므로
# 모델별로 따로 둔다 (한 값을 공유하면 한쪽이 반드시 손해를 본다).
DEFAULT_RIP_CONF = 0.25
DEFAULT_PERSON_CONF = 0.25
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
        conf: float = DEFAULT_RIP_CONF,
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
                    if poly is not None and len(poly) >= 3:
                        points = [tuple(p) for p in np.asarray(poly).tolist()]
                    else:
                        # detect(박스) 전용 모델은 마스크가 없으므로 박스를 사각형 폴리곤으로 대신 쓴다.
                        x1, y1, x2, y2 = xyxy[i]
                        points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                    rips.append(RipRegion(polygon=points, conf=float(confs[i])))
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


def _target_class_ids(detector: Detector, hints: tuple[str, ...]) -> list[int]:
    """서브 모델에서 그 모델의 목적에 해당하는 클래스만 고른다.

    이름이 힌트에 걸리면 그 클래스만 쓰고, 하나도 안 걸리면 전용 모델로 보고 전체를 쓴다.
    사람 모델 자리에 COCO 사전학습 모델(yolo11n.pt 등)을 붙였을 때
    배나 새까지 사람으로 세는 것을 막는 장치다.
    """
    names = dict(getattr(detector, "names", {}) or {})
    return match_class_ids(names, hints) or list(names) or [0]


class DualDetector:
    """이안류 탐지 모델과 사람 탐지 모델을 각각 돌려 결과를 합친다.

    두 모델은 클래스 구성이 서로 다르므로, 각 서브 모델에서 쓸 클래스는 생성 시점에
    이름으로 정해둔다 (_target_class_ids). 따라서 predict 에 넘어오는
    rip_ids/person_ids 는 클래스 필터가 아니라 "그 종류가 필요한가"라는 on/off 스위치로만 쓴다.
    비어 있으면 그 모델은 아예 돌리지 않는다. 이안류 구역을 시나리오로 그리는
    core.scenario.hybrid_result 가 사람 모델만 쓰려고 rip_ids 에 () 를 넘긴다.
    """

    names = {0: "rip_current", 1: "person"}

    def __init__(self, rip_detector: Detector, person_detector: Detector):
        self.rip_detector = rip_detector
        self.person_detector = person_detector
        self.rip_class_ids = _target_class_ids(rip_detector, RIP_HINTS)
        self.person_class_ids = _target_class_ids(person_detector, PERSON_HINTS)

    def predict(
        self,
        frame_bgr: np.ndarray,
        frame_idx: int,
        timestamp_sec: float,
        rip_ids: Sequence[int],
        person_ids: Sequence[int],
    ) -> FrameResult:
        h, w = frame_bgr.shape[:2]
        infer_ms = 0.0

        rips: list[RipRegion] = []
        if len(rip_ids) > 0:
            rip_result = self.rip_detector.predict(
                frame_bgr, frame_idx, timestamp_sec, self.rip_class_ids, []
            )
            rips = rip_result.rips
            infer_ms += rip_result.infer_ms

        persons: list[PersonBox] = []
        if len(person_ids) > 0:
            person_result = self.person_detector.predict(
                frame_bgr, frame_idx, timestamp_sec, [], self.person_class_ids
            )
            persons = person_result.persons
            infer_ms += person_result.infer_ms

        mark_persons_in_rip(persons, rips)

        return FrameResult(
            frame_idx=frame_idx,
            timestamp_sec=timestamp_sec,
            width=w,
            height=h,
            rips=rips,
            persons=persons,
            infer_ms=infer_ms,
        )


def _scan_models_dir(subdir: str | None = None, skip_hints: tuple[str, ...] = ()) -> Path | None:
    models_dir = Path(__file__).resolve().parent.parent / "models"
    if subdir:
        models_dir = models_dir / subdir
    for pattern in ("*.pt", "*.onnx"):
        found = sorted(p for p in models_dir.glob(pattern) if not matches_hint(p.stem, skip_hints))
        if found:
            return found[0]
    return None


def resolve_weights(
    explicit: str | None = None,
    env_var: str = "RIP_MODEL_PATH",
    subdir: str | None = None,
    skip_hints: tuple[str, ...] = (),
) -> Path | None:
    """가중치 경로를 찾는다. 없으면 None."""
    candidate = explicit or os.getenv(env_var, "")
    if candidate:
        path = Path(candidate)
        return path if path.exists() else None
    return _scan_models_dir(subdir, skip_hints)


def resolve_rip_weights(explicit: str | None = None) -> Path | None:
    """이안류 탐지 모델 가중치. models/rip/ 를 먼저 보고, 없으면 기존 위치인 models/ 바로 아래도 찾는다.

    루트를 훑을 때 파일명이 사람 모델처럼 보이는 것(person_best.pt 등)은 건너뛴다.
    두 모델을 모두 models/ 바로 아래에 두면 이름순으로 사람 모델이 먼저 걸려
    이안류 모델 자리에 조용히 들어앉는 사고가 난다.
    """
    return resolve_weights(explicit, "RIP_MODEL_PATH", "rip") or resolve_weights(
        explicit, "RIP_MODEL_PATH", None, skip_hints=PERSON_HINTS
    )


def resolve_person_weights(explicit: str | None = None) -> Path | None:
    """사람(표류자) 탐지 모델 가중치. models/person/ 또는 .env의 PERSON_MODEL_PATH."""
    return resolve_weights(explicit, "PERSON_MODEL_PATH", "person")


def build_detector(
    rip_weights: str | Path | None = None,
    person_weights: str | Path | None = None,
    rip_conf: float = DEFAULT_RIP_CONF,
    rip_iou: float = DEFAULT_IOU,
    rip_imgsz: int = DEFAULT_IMGSZ,
    person_conf: float = DEFAULT_PERSON_CONF,
    person_iou: float = DEFAULT_IOU,
    person_imgsz: int = DEFAULT_IMGSZ,
) -> tuple[Detector, bool, bool]:
    """(탐지기, 이안류_모델_실제_여부, 사람_모델_실제_여부)를 돌려준다.

    conf/iou/imgsz 는 모델별로 따로 받는다. 이안류 구역과 원거리 CCTV의 작은 사람은
    적정 임계값이 서로 다르기 때문에 한 값을 공유하면 한쪽이 반드시 손해를 본다.

    둘 다 없으면 기존처럼 완전한 FakeDetector 하나를 돌려준다.
    한쪽만 있으면 DualDetector가 있는 쪽은 실제 추론, 없는 쪽은 FakeDetector로 채운다.
    """
    rip_path = resolve_rip_weights(str(rip_weights) if rip_weights else None)
    person_path = resolve_person_weights(str(person_weights) if person_weights else None)

    if rip_path is None and person_path is None:
        return FakeDetector(), False, False

    rip_detector: Detector = (
        YoloDetector(rip_path, conf=rip_conf, iou=rip_iou, imgsz=rip_imgsz)
        if rip_path
        else FakeDetector()
    )
    person_detector: Detector = (
        YoloDetector(person_path, conf=person_conf, iou=person_iou, imgsz=person_imgsz)
        if person_path
        else FakeDetector()
    )
    return DualDetector(rip_detector, person_detector), rip_path is not None, person_path is not None
