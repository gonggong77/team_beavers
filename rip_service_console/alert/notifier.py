"""알림 어댑터. 웹에서는 notifier.send(event) 한 줄만 호출한다."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable

# ⭐️ core 대신 alert 패키지 경로로 수정
from alert.fcm_alert import SERVICE_ACCOUNT_PATH, send_rip_current_alert, snapshot_url
from core.schemas import AlertEvent


@dataclass
class SendResult:
    ok: bool
    detail: str = ""
    image_url: str = ""


@runtime_checkable
class Notifier(Protocol):
    name: str

    def send(self, event: AlertEvent) -> SendResult: ...


class ConsoleNotifier:
    """발송 없이 흐름만 확인할 때 쓰는 기본 구현. 터미널에 문안을 찍는다."""

    name = "console"
    setup_hint = ""
    configured = True

    def send(self, event: AlertEvent) -> SendResult:
        print("=" * 60)
        print(event.summary_text())
        print("=" * 60)
        return SendResult(ok=True, detail="stdout")


class FcmNotifier:
    """안드로이드 앱으로 FCM 토픽 경보 발송. 스냅샷은 Streamlit 정적 서빙 URL로 전달한다."""

    name = "fcm"
    setup_hint = "Firebase 서비스 계정 키가 없어 발송이 실패합니다. serviceAccountKey.json 위치를 확인해 주세요."

    def __init__(
        self,
        latitude: float | None = None,
        longitude: float | None = None,
        camera_id: str | None = None,
    ):
        self.latitude = latitude if latitude is not None else float(os.getenv("RIP_ALERT_LAT", "35.1587"))
        self.longitude = longitude if longitude is not None else float(os.getenv("RIP_ALERT_LON", "129.1604"))
        self.camera_id = camera_id or os.getenv("RIP_ALERT_CAMERA_ID", "CCTV-E01")

    @property
    def configured(self) -> bool:
        return SERVICE_ACCOUNT_PATH.exists()

    def send(self, event: AlertEvent) -> SendResult:
        if not self.configured:
            return SendResult(ok=False, detail=self.setup_hint)

        image_url = ""
        upload_error = ""
        if event.snapshot_path and os.path.exists(event.snapshot_path):
            try:
                image_url = snapshot_url(event.snapshot_path)
            except Exception as exc:
                upload_error = f" (이미지 URL 생성 실패: {type(exc).__name__}: {exc})"

        try:
            detected_at = datetime.fromisoformat(event.occurred_at).strftime("%H:%M:%S")
        except ValueError:
            detected_at = datetime.now().strftime("%H:%M:%S")

        try:
            message_id = send_rip_current_alert(
                location_name=event.video_source,
                person_count=event.total_persons,
                latitude=self.latitude,
                longitude=self.longitude,
                image_url=image_url,
                risk_level=event.risk_level,
                persons_in_rip=event.persons_in_rip,
                total_persons=event.total_persons,
                rip_count=event.rip_count,
                camera_id=self.camera_id,
                detected_at=detected_at,
            )
            return SendResult(ok=True, detail=f"sent ({message_id}){upload_error}", image_url=image_url)
        except Exception as exc:
            return SendResult(ok=False, detail=f"{type(exc).__name__}: {exc}{upload_error}", image_url=image_url)


def build_notifier(kind: str = "console") -> Notifier:
    if kind == "fcm":
        return FcmNotifier()
    return ConsoleNotifier()


if __name__ == "__main__":
    sample = AlertEvent(
        risk_level="emergency",
        persons_in_rip=2,
        total_persons=5,
        rip_count=1,
        frame_idx=412,
        timestamp_sec=13.7,
        video_source="haeundae_demo.mp4",
    )
    notifier = build_notifier(os.getenv("NOTIFIER", "console"))
    print(notifier.name, notifier.send(sample))