"""실제 AI 추론 및 비동기 경보 발송 통합 파이프라인.

- 듀얼 AI 모델: YOLO11m(이안류) + SAHI YOLO11m(수영객 768px 타일 슬라이싱)
- 탐지 성능/속도 최적화: 1080p 고해상도 타일링 + torch.inference_mode() 가속
- 제로 딜레이 경보: 스냅샷 JPG 인코딩 및 FCM 발송 전체 백그라운드 스레드 위임
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import torch
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from ultralytics import YOLO

from alert.fcm_alert import SERVICE_ACCOUNT_PATH
from alert.gate import AlertGate
from alert.notifier import build_notifier
from alert.sender import _ALERT_EXECUTOR, dispatch, save_snapshot
from alert.store import EventStore
from core.render import draw_overlay
from core.rules import grade, point_in_polygon
from core.schemas import AlertEvent, FrameResult, PersonBox, RipRegion, RiskLevel

MAX_W = 960

DEFAULT_RIP_MODEL = "best_yolo11m_integrated_v2_first.pt"
DEFAULT_SWIMMER_MODEL = "best_swimmer_yolo11m_b8.pt"


@dataclass
class DemoStep:
    annotated_bgr: np.ndarray
    result: FrameResult
    risk: RiskLevel
    progress: float
    event: AlertEvent | None = None


class SurveillanceInferenceEngine:
    """AI 모델 싱글톤 인스턴스 관리자."""

    _instance = None

    def __init__(
        self,
        rip_model_path: str = DEFAULT_RIP_MODEL,
        swimmer_model_path: str = DEFAULT_SWIMMER_MODEL,
        swimmer_conf: float = 0.05,
    ):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.rip_model = YOLO(rip_model_path)
        self.swimmer_model = AutoDetectionModel.from_pretrained(
            model_type="yolov8",
            model_path=str(Path(swimmer_model_path).resolve()),
            confidence_threshold=swimmer_conf,
            device=self.device,
        )

    @classmethod
    def get_engine(cls, rip_path: str, swimmer_path: str, conf: float):
        if cls._instance is None:
            cls._instance = cls(rip_path, swimmer_path, conf)
        return cls._instance


def mark_persons_robust(
    persons: list[PersonBox], rips: list[RipRegion], margin: float = 15.0
) -> None:
    """중심점/수면점 및 경계선 마진(15px)을 적용하여 조난자 판정 누락 방지."""
    for person in persons:
        pts_to_check = [person.bottom_center, person.center]
        in_rip = False
        for rip in rips:
            # 1. BBox 기준점 포함 여부
            if any(point_in_polygon(pt, rip.polygon) for pt in pts_to_check):
                in_rip = True
                break

            # 2. 파도 거품 경계선 여유 마진 검사
            poly_np = np.asarray(rip.polygon, dtype=np.float32).reshape(-1, 1, 2)
            for pt in pts_to_check:
                dist = cv2.pointPolygonTest(poly_np, (float(pt[0]), float(pt[1])), True)
                if dist >= -margin:
                    in_rip = True
                    break
            if in_rip:
                break
        person.in_rip = in_rip


def run_pipeline(
    video_path: Path,
    frame_skip: int = 2,
    max_frames: int = 400,
    show_foot: bool = True,
    source_name: str = "",
    rip_model_path: str = DEFAULT_RIP_MODEL,
    swimmer_model_path: str = DEFAULT_SWIMMER_MODEL,
    swimmer_conf: float = 0.05,
) -> Iterator[DemoStep]:
    """실시간 비디오 분석 및 모바일 발송 파이프라인 제너레이터."""
    source = source_name or video_path.name
    engine = SurveillanceInferenceEngine.get_engine(
        rip_model_path, swimmer_model_path, swimmer_conf
    )

    # 1. 모바일 알림 인프라 초기화 (중복 억제 해제, 키 미배치 시 안전 전환)
    alert_gate = AlertGate(min_consecutive=1, cooldown_sec=0.0)
    use_fcm = SERVICE_ACCOUNT_PATH.exists() and os.getenv("NOTIFIER", "fcm") == "fcm"
    notifier = build_notifier("fcm" if use_fcm else "console")
    store = EventStore()

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    planned = min(max_frames, total // max(frame_skip, 1)) if total else max_frames

    idx, used = 0, 0
    cached_rips: list[RipRegion] = []

    try:
        while used < max_frames:
            ok, frame = capture.read()
            if not ok:
                break

            if idx % max(frame_skip, 1) == 0:
                t_start = time.time()
                orig_h, orig_w = frame.shape[:2]
                used += 1
                progress = used / max(planned, 1)
                t_sec = idx / fps

                # -------------------------------------------------------------
                # 2. 이안류 추론 (5스텝 주기 갱신으로 GPU 연산 여유 확보)
                # -------------------------------------------------------------
                if used % 5 == 1 or len(cached_rips) == 0:
                    with torch.inference_mode():
                        rip_res = engine.rip_model(
                            frame,
                            imgsz=640,
                            conf=0.20,
                            iou=0.55,
                            device=engine.device,
                            verbose=False,
                            half=True if "cuda" in engine.device else False,
                        )[0]

                    new_rips = []
                    for b in rip_res.boxes:
                        x1, y1, x2, y2 = map(float, b.xyxy[0].cpu().numpy())
                        score = float(b.conf[0].cpu().numpy())
                        polygon = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                        new_rips.append(RipRegion(polygon=polygon, conf=score))
                    cached_rips = new_rips

                # -------------------------------------------------------------
                # 3. 수영객 SAHI 추론 (원본 해상도 768px 타일링, conf=0.05)
                # -------------------------------------------------------------
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                with torch.inference_mode():
                    sahi_res = get_sliced_prediction(
                        frame_rgb,
                        engine.swimmer_model,
                        slice_height=768,
                        slice_width=768,
                        overlap_height_ratio=0.18,
                        overlap_width_ratio=0.18,
                        perform_standard_pred=False,
                        verbose=0,
                    )

                persons: list[PersonBox] = []
                for p_idx, obj in enumerate(sahi_res.object_prediction_list):
                    box_coords = tuple(map(float, obj.bbox.to_xyxy()))
                    persons.append(
                        PersonBox(
                            xyxy=box_coords,
                            conf=float(obj.score.value),
                            track_id=p_idx + 1,
                        )
                    )

                # -------------------------------------------------------------
                # 4. 침범 판정 및 오버레이 렌더링
                # -------------------------------------------------------------
                mark_persons_robust(persons, cached_rips, margin=15.0)
                infer_ms = (time.time() - t_start) * 1000.0

                result = FrameResult(
                    frame_idx=idx,
                    timestamp_sec=t_sec,
                    width=orig_w,
                    height=orig_h,
                    rips=cached_rips,
                    persons=persons,
                    infer_ms=infer_ms,
                )

                risk = grade(result)
                annotated = draw_overlay(
                    frame, result, risk, show_conf=True, show_foot=show_foot
                )

                # -------------------------------------------------------------
                # 5. 경보 게이트 판정 및 제로 딜레이 백그라운드 발송
                # -------------------------------------------------------------
                trigger_kind = alert_gate.check(risk, t_sec, result.persons_in_rip)
                event = None

                if trigger_kind is not None:
                    event = AlertEvent(
                        risk_level=risk,
                        persons_in_rip=result.persons_in_rip,
                        total_persons=result.total_persons,
                        rip_count=result.rip_count,
                        frame_idx=idx,
                        timestamp_sec=t_sec,
                        video_source=source,
                        trigger_kind=trigger_kind,
                        notify_status="dispatched",
                    )

                    # 메인 루프 지연을 없애기 위해 JPG 인코딩과 저장을 백그라운드 스레드로 위임
                    snap_frame = annotated.copy()

                    def _async_job(ev=event, img=snap_frame):
                        ok_enc, buf = cv2.imencode(".jpg", img)
                        if ok_enc:
                            save_snapshot(buf.tobytes(), ev)
                        dispatch(ev, notifier=notifier, store=store, sync=True)

                    _ALERT_EXECUTOR.submit(_async_job)

                # -------------------------------------------------------------
                # 6. 웹 송출용 해상도 리사이즈 (960px)
                # -------------------------------------------------------------
                if orig_w > MAX_W:
                    disp_h = int(orig_h * (MAX_W / orig_w))
                    annotated_disp = cv2.resize(
                        annotated, (MAX_W, disp_h), interpolation=cv2.INTER_LINEAR
                    )
                else:
                    annotated_disp = annotated

                yield DemoStep(annotated_disp, result, risk, progress, event)

            idx += 1
    finally:
        capture.release()