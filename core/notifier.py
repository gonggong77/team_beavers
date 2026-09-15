"""알림 어댑터. (담당: 김종호)

웹 쪽에서는 notifier.send(event) 한 줄만 호출한다.
알림 담당자는 이 파일의 TelegramNotifier.send 안쪽만 채우면 되고,
웹 코드는 건드릴 필요가 없다.

단독 테스트:
    export TELEGRAM_BOT_TOKEN=... TELEGRAM_CHAT_ID=...
    python -m core.notifier
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


@runtime_checkable
class Notifier(Protocol):
    name: str

    def send(self, event: AlertEvent) -> SendResult: ...


class ConsoleNotifier:
    """알림 서버 없이 웹을 테스트할 때 쓰는 기본 구현."""

    name = "console"
    setup_hint = ""

    def send(self, event: AlertEvent) -> SendResult:
        print("=" * 60)
        print(event.summary_text())
        print("=" * 60)
        return SendResult(ok=True, detail="stdout")


class TelegramNotifier:
    """텔레그램 Bot API 발송.

    봇 토큰은 절대 코드에 적지 말고 .env 또는 st.secrets로 넘긴다.
    """

    name = "telegram"
    setup_hint = "텔레그램 토큰이 설정되지 않아 발송이 실패합니다. .env를 확인해 주세요."

    def __init__(self, bot_token: str | None = None, chat_id: str | None = None, timeout: float = 10.0):
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "")
        self.timeout = timeout

    @property
    def configured(self) -> bool:
        return bool(self.bot_token and self.chat_id)

    def send(self, event: AlertEvent) -> SendResult:
        if not self.configured:
            return SendResult(ok=False, detail="TELEGRAM_BOT_TOKEN 또는 TELEGRAM_CHAT_ID 미설정")

        import requests

        base = f"https://api.telegram.org/bot{self.bot_token}"
        caption = event.summary_text()

        try:
            if event.snapshot_path and os.path.exists(event.snapshot_path):
                with open(event.snapshot_path, "rb") as fp:
                    response = requests.post(
                        f"{base}/sendPhoto",
                        data={"chat_id": self.chat_id, "caption": caption},
                        files={"photo": fp},
                        timeout=self.timeout,
                    )
            else:
                response = requests.post(
                    f"{base}/sendMessage",
                    data={"chat_id": self.chat_id, "text": caption},
                    timeout=self.timeout,
                )
            if response.status_code == 200:
                return SendResult(ok=True, detail="sent")
            return SendResult(ok=False, detail=f"HTTP {response.status_code} {response.text[:200]}")
        except Exception as exc:  # 알림 실패가 영상 분석을 멈추면 안 된다
            return SendResult(ok=False, detail=f"{type(exc).__name__}: {exc}")


class FcmNotifier:
    """안드로이드 앱으로 FCM 토픽 경보 발송. 스냅샷은 Firebase Storage에 올려 서명 URL로 전달한다.

    카메라 위치 정보는 탐지 결과가 아니라 배포 설정이라 AlertEvent에는 없다.
    생성자 인자로 받거나 env(RIP_ALERT_*)에서 읽는다.
    """

    name = "fcm"
    setup_hint = "Firebase 서비스 계정 키가 없어 발송이 실패합니다. serviceAccountKey.json 위치를 확인해 주세요."

    def __init__(
        self,
        location_name: str | None = None,
        latitude: float | None = None,
        longitude: float | None = None,
        camera_id: str | None = None,
    ):
        self.location_name = location_name or os.getenv("RIP_ALERT_LOCATION", "해운대 해수욕장")
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

        from core.fcm_alert import send_rip_current_alert, upload_snapshot

        image_url = ""
        upload_error = ""
        if event.snapshot_path and os.path.exists(event.snapshot_path):
            try:
                image_url = upload_snapshot(event.snapshot_path)
            except Exception as exc:  # 업로드 실패가 경보 발송 자체를 막으면 안 된다
                upload_error = f" (이미지 업로드 실패: {type(exc).__name__}: {exc})"

        try:
            detected_at = datetime.fromisoformat(event.occurred_at).strftime("%H:%M:%S")
        except ValueError:
            detected_at = datetime.now().strftime("%H:%M:%S")

        try:
            message_id = send_rip_current_alert(
                location_name=self.location_name,
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
            return SendResult(ok=True, detail=f"sent ({message_id}){upload_error}")
        except Exception as exc:  # 알림 실패가 영상 분석을 멈추면 안 된다
            return SendResult(ok=False, detail=f"{type(exc).__name__}: {exc}{upload_error}")


def build_notifier(kind: str = "console") -> Notifier:
    if kind == "telegram":
        return TelegramNotifier()
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
