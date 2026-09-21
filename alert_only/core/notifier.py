"""알림 어댑터. 웹에서는 notifier.send(event) 한 줄만 호출한다.

원본 core/notifier.py 를 그대로 옮긴 것이다. 발송 방식이 늘어나면
여기에 클래스를 하나 더 만들고 build_notifier 에 분기만 추가하면 된다.

지켜야 할 규칙 두 가지 (docs/NOTIFIER_CONTRACT.md):
  1. send() 는 예외를 밖으로 던지지 않는다. 실패는 SendResult(ok=False) 로 돌려준다.
     알림 실패가 영상 분석을 멈추면 안 되기 때문이다.
  2. 인증 정보를 코드에 적지 않는다. serviceAccountKey.json 은 .gitignore 대상이다.

단독 테스트:
    python -m core.notifier            # 콘솔 출력만
    NOTIFIER=fcm python -m core.notifier   # 실제 발송 (서비스 계정 키 필요)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable

from core.schemas import AlertEvent


@dataclass
class SendResult:
    ok: bool
    detail: str = ""
    # 발송 시 스냅샷을 가리키는 URL(Streamlit 정적 서빙). FCM 이 아닌 알림은 채우지 않는다.
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
    """안드로이드 앱으로 FCM 토픽 경보 발송. 스냅샷은 Streamlit 정적 서빙 URL로 전달한다.

    좌표와 카메라 ID는 탐지 결과가 아니라 배포 설정이라 AlertEvent에는 없다.
    생성자 인자로 받거나 env(RIP_ALERT_*)에서 읽는다.

    지점 이름만은 설정이 아니라 영상 파일명(event.video_source)을 쓴다.
    지금 MVP 가 영상 업로드 기반이라 그쪽이 화면에서 더 쓸모 있다.
    """

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
        from core.fcm_alert import SERVICE_ACCOUNT_PATH

        return SERVICE_ACCOUNT_PATH.exists()

    def send(self, event: AlertEvent) -> SendResult:
        if not self.configured:
            return SendResult(ok=False, detail=self.setup_hint)

        from core.fcm_alert import send_rip_current_alert, snapshot_url

        image_url = ""
        upload_error = ""
        if event.snapshot_path and os.path.exists(event.snapshot_path):
            try:
                image_url = snapshot_url(event.snapshot_path)
            except Exception as exc:  # URL 생성 실패가 경보 발송 자체를 막으면 안 된다
                upload_error = f" (이미지 URL 생성 실패: {type(exc).__name__}: {exc})"

        try:
            detected_at = datetime.fromisoformat(event.occurred_at).strftime("%H:%M:%S")
        except ValueError:
            detected_at = datetime.now().strftime("%H:%M:%S")

        try:
            message_id = send_rip_current_alert(
                # 영상 파일명을 그대로 지점 이름으로 쓴다. 어느 영상이 낸 경보인지
                # 알림 배너와 앱 화면(tvZone)에서 바로 구분된다.
                # zone 은 넘기지 않는다 — send_rip_current_alert 가 location_name 으로 채운다.
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
        except Exception as exc:  # 알림 실패가 영상 분석을 멈추면 안 된다
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
