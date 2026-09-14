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


def build_notifier(kind: str = "console") -> Notifier:
    return TelegramNotifier() if kind == "telegram" else ConsoleNotifier()


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
