"""모델 어댑터.

웹 코드는 Detector 프로토콜만 알고 있으면 된다.
모델은 이안류 탐지 모델(segment)과 사람(표류자) 탐지 모델(detect), 두 개를 따로 돌려
DualDetector가 결과를 합친다. 한쪽 모델 파일이 없으면 그 쪽만 FakeDetector로 대체된다.

이안류 모델은 models/rip/ (또는 .env의 RIP_MODEL_PATH)에,
사람 모델은 models/person/ (또는 .env의 PERSON_MODEL_PATH)에 넣으면 끝난다.
기존처럼 models/ 바로 아래에 이안류 모델을 둔 경우도 계속 인식한다.
"""

from __future__ import annotations

import atexit
import math
import os
import shutil
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Protocol, Sequence, runtime_checkable

import numpy as np

from core.model_probe import PERSON_HINTS, RIP_HINTS, match_class_ids, matches_hint
from core.rules import mark_persons_in_rip
from core.schemas import FrameResult, PersonBox, RipRegion

# 이안류 구역과 원거리 CCTV의 작은 사람은 적정 임계값이 서로 다르므로
# 모델별로 따로 둔다 (한 값을 공유하면 한쪽이 반드시 손해를 본다).
# 사람 쪽 값은 모델팀 추론 스크립트(rip_person_infer.py)에서 그대로 가져왔다.
# 작은 입수자를 잡기 위해 해상도를 키우고 conf 를 크게 낮춘 조합이다.
DEFAULT_RIP_CONF = 0.25
DEFAULT_IOU = 0.7
DEFAULT_IMGSZ = 640

DEFAULT_PERSON_CONF = 0.10
DEFAULT_PERSON_IOU = 0.5
DEFAULT_PERSON_IMGSZ = 1024

# 사람은 ByteTrack 으로 추적한다. 같은 사람에게 프레임 내내 같은 번호가 붙어야
# core.rules.TrackSmoother 가 '이 사람이 정말 구역 안인가'를 판단할 수 있다.
#
# 기본 bytetrack.yaml 을 그대로 쓰면 안 된다. 그쪽 기준값(new_track_thresh 0.6,
# track_high_thresh 0.5)은 conf 0.25 이상으로 탐지하는 일반적인 상황에 맞춰져 있는데,
# 우리 사람 모델은 원거리 CCTV의 작은 입수자를 잡으려고 conf 0.10 으로 돌린다.
# 실제 탐지 점수가 0.15~0.25 언저리라 기준값을 넘는 탐지가 하나도 없고,
# 그 결과 트랙이 단 하나도 생성되지 않아 boxes.id 가 계속 None 으로 나온다.
# 그래서 아래 배수로 conf 에 맞춘 설정 파일을 만들어 쓴다.
#
# 배수를 크게 잡으면 안 된다. model.track() 은 트랙에 붙지 못한 탐지를 결과에서
# 아예 빼버리므로, 새 트랙 기준이 높으면 점수가 낮은 진짜 입수자가 화면에서
# 통째로 사라진다 — conf 를 낮춘 이유가 바로 그 사람들이다.
TRACK_HIGH_RATIO = 1.2    # 새 트랙을 시작할 수 있는 점수 = conf * 이 값
_TRACKER_TEMPLATE = """\
# core.detector 가 사람 모델의 conf 에 맞춰 자동 생성한 파일입니다. 직접 고치지 마세요.
tracker_type: bytetrack
track_high_thresh: {high:.4f}
track_low_thresh: {low:.4f}
new_track_thresh: {high:.4f}
track_buffer: 30
match_thresh: 0.8
# 점수와 IoU 거리를 곱해 쓰는 옵션. 점수가 0.2 대인 우리 상황에서는
# 제대로 겹친 박스까지 전부 나쁜 후보로 만들어 매칭을 망가뜨린다.
fuse_score: False
"""

# 트래커 설정 파일을 모아 둘 프로세스 전용 임시 폴더. 프로세스가 끝나면 통째로 지운다.
_TRACKER_DIR: Path | None = None

# resolve_device() 결과 캐시. torch.cuda.is_available() 은 첫 호출이 느리고,
# 실행 중에 결과가 바뀔 일도 없다.
_DEVICE: str | None = None


def resolve_device() -> str:
    """추론에 쓸 장치를 정한다. "cuda:0" 또는 "cpu".

    .env 의 INFER_DEVICE 로 강제할 수 있다 (GPU가 있어도 CPU로 비교해보고 싶을 때).
    torch 가 CPU 전용 빌드로 깔려 있으면 GPU가 꽂혀 있어도 cuda 를 못 쓴다 —
    그 경우 조용히 cpu 로 떨어지므로, 화면(app.py 사이드바)에 항상 표시한다.
    """
    global _DEVICE
    if _DEVICE is None:
        forced = os.getenv("INFER_DEVICE", "").strip()
        if forced:
            _DEVICE = forced
        else:
            try:
                import torch  # 무거우므로 지연 임포트

                _DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
            except Exception:
                _DEVICE = "cpu"
    return _DEVICE


def device_label(device: str | None = None) -> str:
    """사이드바/CLI 에 띄울 장치 설명. 예: "GPU(NVIDIA GeForce RTX 4060) FP16"."""
    device = device or resolve_device()
    if not device.startswith("cuda"):
        return f"{device.upper()} (GPU 미사용)"
    try:
        import torch

        index = int(device.split(":")[1]) if ":" in device else 0
        return f"GPU({torch.cuda.get_device_name(index)}) FP16"
    except Exception:
        return "GPU FP16"


def _tracker_dir() -> Path:
    """트래커 설정 파일을 둘 폴더. 처음 부를 때 만들고 종료 시 삭제를 예약한다.

    ultralytics 는 트래커 설정을 파일 경로로만 받기 때문에 파일을 만들 수밖에 없다.
    프로세스 전용 폴더에 모아 두고 atexit 로 통째로 지우면, 앱을 껐을 때 아무것도
    남지 않는다 (강제 종료로 죽으면 폴더 하나가 남지만 수백 바이트 수준이다).
    """
    global _TRACKER_DIR
    if _TRACKER_DIR is None:
        _TRACKER_DIR = Path(tempfile.mkdtemp(prefix="beavers_tracker_"))
        atexit.register(shutil.rmtree, _TRACKER_DIR, ignore_errors=True)
    return _TRACKER_DIR


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

    def reset(self) -> None:
        """새 영상을 시작하기 전에 누적된 상태(트래커 등)를 버린다.

        트래킹은 프레임 사이의 상태를 들고 있는데, 앱은 detector 를
        st.cache_resource 로 전역 공유한다. 이걸 부르지 않으면 앞 영상의
        트랙 번호가 다음 영상 사람에게 그대로 이어붙는다.
        """
        ...


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

    def reset(self) -> None:
        """frame_idx 만 보고 결정론적으로 움직이므로 버릴 상태가 없다."""

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
        half: bool | None = None,
        track: bool = False,
        augment: bool = False,
    ):
        from ultralytics import YOLO  # 무거우므로 지연 임포트

        self.weights = str(weights)
        self.model = YOLO(self.weights)
        self.names = dict(self.model.names)
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = device or resolve_device()
        self.on_cuda = self.device.startswith("cuda")
        # FP16 은 GPU 에서만 쓴다. CPU 에 half=True 를 주면 ultralytics 가 죽는다.
        self.half = self.on_cuda if half is None else (half and self.on_cuda)
        self.track = track
        self.augment = augment
        self._tracker_cfg: str | None = None

        if self.on_cuda:
            # 가중치를 미리 GPU 로 올린다. 첫 프레임에서 로딩이 겹치지 않게.
            self.model.to(self.device)
            import torch

            # 프레임 크기가 영상 내내 고정이라 cuDNN 오토튜닝이 확실히 이득이다.
            torch.backends.cudnn.benchmark = True

    def _tracker_config(self) -> str:
        """conf 에 맞춘 ByteTrack 설정 파일을 만들고 경로를 돌려준다.

        파일명에 conf 를 넣어 고정 경로를 쓴다. 같은 conf 면 매번 같은 파일을
        덮어쓰므로, detector 를 새로 만들어도(개발자가 사이드바에서 값을 조절할
        때마다 새로 만들어진다) 파일이 쌓이지 않는다. 폴더째로 종료 시 지워진다.
        """
        if self._tracker_cfg is None:
            high = min(self.conf * TRACK_HIGH_RATIO, 0.9)
            path = _tracker_dir() / f"bytetrack_{self.conf:.4f}.yaml"
            path.write_text(_TRACKER_TEMPLATE.format(high=high, low=self.conf), encoding="utf-8")
            self._tracker_cfg = str(path)
        return self._tracker_cfg

    def reset(self) -> None:
        """트래커를 버린다. 다음 추론에서 1번부터 새로 번호가 매겨진다.

        ultralytics 는 predictor 안에 트래커를 들고 있고 공개 API 로 비우는
        방법이 없다. predictor 를 통째로 떨어뜨리면 다음 호출에서 다시 만든다
        (가중치는 self.model 에 남아 있어 파일을 다시 읽지는 않는다).
        """
        self.model.predictor = None

    def warmup(self) -> None:
        """검은 프레임으로 한 번 돌려 첫 추론의 준비 비용을 미리 치른다.

        GPU 에서는 CUDA 컨텍스트 생성과 cuDNN 오토튜닝 때문에 첫 호출만 2~5초 걸린다.
        모델 로딩 시점(앱 시작)으로 옮겨두지 않으면 '분석 시작' 을 누른 직후
        화면이 멈춘 것처럼 보인다.

        워밍업 프레임은 앞뒤가 이어지지 않으므로 반드시 추적을 끈 채로 돌린다
        (켜두면 다음 영상의 1번 트랙이 이 검은 프레임에서 시작된다).
        """
        blank = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        with tracking_disabled(self):
            self._infer(blank)
        self.reset()

    def _infer(self, frame_bgr):
        kwargs = dict(
            conf=self.conf, iou=self.iou, imgsz=self.imgsz,
            augment=self.augment, half=self.half, verbose=False,
        )
        if self.device:
            kwargs["device"] = self.device
        if self.track:
            # persist=True 라야 프레임 사이에 트랙이 이어진다.
            return self.model.track(
                frame_bgr, persist=True, tracker=self._tracker_config(), **kwargs
            )[0]
        return self.model.predict(frame_bgr, **kwargs)[0]

    def predict(self, frame_bgr, frame_idx, timestamp_sec, rip_ids, person_ids) -> FrameResult:
        started = time.perf_counter()
        h, w = frame_bgr.shape[:2]

        result = self._infer(frame_bgr)

        rips: list[RipRegion] = []
        persons: list[PersonBox] = []

        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            classes = boxes.cls.int().tolist()
            confs = boxes.conf.tolist()
            xyxy = boxes.xyxy.tolist()
            # 탐지가 0건이면 result.masks 가 None 이다. 가장 흔한 크래시 지점.
            polygons = result.masks.xy if result.masks is not None else [None] * len(classes)
            # 트래킹을 켜도 추적이 막 시작된 프레임에서는 boxes.id 가 None 이다.
            track_ids = boxes.id.int().tolist() if boxes.id is not None else [None] * len(classes)

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
                    persons.append(
                        PersonBox(
                            xyxy=tuple(xyxy[i]),
                            conf=float(confs[i]),
                            track_id=track_ids[i] if i < len(track_ids) else None,
                        )
                    )

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


@contextmanager
def tracking_disabled(detector: Detector):
    """이 블록 안에서만 추적을 끈다.

    프레임이 연속이 아닌 구간에서 반드시 써야 한다. ByteTrack 은 직전 프레임과의
    겹침으로 같은 사람을 찾는데, 관제 화면의 사전 추론처럼 영상 전체에서 몇 초씩
    건너뛰며 뽑은 프레임에서는 매칭이 전부 실패한다. 그러면 프레임마다 번호가
    새로 생길 뿐 아니라, model.track() 이 트랙에 붙지 못한 탐지를 결과에서 빼기
    때문에 화면에 보이는 사람이 오히려 줄어든다.
    """
    subs = [
        sub for sub in (
            detector,
            getattr(detector, "rip_detector", None),
            getattr(detector, "person_detector", None),
        )
        if getattr(sub, "track", False)
    ]
    for sub in subs:
        sub.track = False
    try:
        yield detector
    finally:
        for sub in subs:
            sub.track = True


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

    def __init__(
        self,
        rip_detector: Detector,
        person_detector: Detector,
        inside_rule: str = "bottom",
    ):
        self.rip_detector = rip_detector
        self.person_detector = person_detector
        self.rip_class_ids = _target_class_ids(rip_detector, RIP_HINTS)
        self.person_class_ids = _target_class_ids(person_detector, PERSON_HINTS)
        # 서브 detector 안에서도 한 번 매겨지지만 여기서 덮어쓰므로,
        # 침범 판정 기준점이 실제로 적용되는 곳은 이 클래스뿐이다.
        #
        # 앱과 CLI 는 이 값을 넘기지 않는다 — 발끝(bottom) 고정이다. 화면에 찍는
        # 판정 점(core.render 의 show_foot)도 발끝을 그리므로, 여기를 "center" 로
        # 바꾸면 화면과 실제 판정이 어긋난다. 라이브러리로 직접 쓸 때만 건드릴 것.
        self.inside_rule = inside_rule

    def reset(self) -> None:
        for sub in (self.rip_detector, self.person_detector):
            reset = getattr(sub, "reset", None)
            if callable(reset):
                reset()

    def warmup(self) -> None:
        """서브 모델에 위임한다. FakeDetector 에는 warmup 이 없으므로 건너뛴다."""
        for sub in (self.rip_detector, self.person_detector):
            warmup = getattr(sub, "warmup", None)
            if callable(warmup):
                warmup()

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

        mark_persons_in_rip(persons, rips, rule=self.inside_rule)

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
    person_iou: float = DEFAULT_PERSON_IOU,
    person_imgsz: int = DEFAULT_PERSON_IMGSZ,
    person_track: bool = True,
    person_augment: bool = False,
    inside_rule: str = "bottom",
    device: str | None = None,
    half: bool | None = None,
) -> tuple[Detector, bool, bool]:
    """(탐지기, 이안류_모델_실제_여부, 사람_모델_실제_여부)를 돌려준다.

    conf/iou/imgsz 는 모델별로 따로 받는다. 이안류 구역과 원거리 CCTV의 작은 사람은
    적정 임계값이 서로 다르기 때문에 한 값을 공유하면 한쪽이 반드시 손해를 본다.

    device 를 비워두면 resolve_device() 가 GPU 유무를 보고 정한다 (GPU 면 FP16 도 자동).

    트래킹과 TTA(augment)는 사람 모델에만 붙인다. 이안류 구역은 개체를 세는
    대상이 아니라 번호를 이어줄 이유가 없고, 두 기법 모두 추론 시간을 늘린다.

    둘 다 없으면 기존처럼 완전한 FakeDetector 하나를 돌려준다.
    한쪽만 있으면 DualDetector가 있는 쪽은 실제 추론, 없는 쪽은 FakeDetector로 채운다.
    """
    rip_path = resolve_rip_weights(str(rip_weights) if rip_weights else None)
    person_path = resolve_person_weights(str(person_weights) if person_weights else None)

    if rip_path is None and person_path is None:
        return FakeDetector(), False, False

    rip_detector: Detector = (
        YoloDetector(
            rip_path, conf=rip_conf, iou=rip_iou, imgsz=rip_imgsz,
            device=device, half=half,
        )
        if rip_path
        else FakeDetector()
    )
    person_detector: Detector = (
        YoloDetector(
            person_path,
            conf=person_conf,
            iou=person_iou,
            imgsz=person_imgsz,
            device=device,
            half=half,
            track=person_track,
            augment=person_augment,
        )
        if person_path
        else FakeDetector()
    )
    detector = DualDetector(rip_detector, person_detector, inside_rule=inside_rule)
    # 첫 추론의 준비 비용(GPU 컨텍스트 생성 등)을 모델 로딩 시점에 미리 치른다.
    detector.warmup()
    return detector, rip_path is not None, person_path is not None
