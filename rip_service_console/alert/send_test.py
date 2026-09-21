"""화면 없이 경보 한 건을 발송하는 CLI.

    python send_test.py                      # 콘솔 출력만 (앱 없이 문안 확인)
    python send_test.py --notifier fcm       # 안드로이드 앱으로 실제 발송
    python send_test.py --notifier fcm --image static/snapshots/xxx.jpg
    python send_test.py --notifier fcm --risk warn --in-rip 0 --total 4

발송 전 확인 (fcm):
    1) 앱을 실행하고 `adb logcat -d -s FCM_CHECK` 로
       "✅ 토픽 구독 완료: rip_current_alert" 를 확인한다 (구독 전파에 수 초 걸림).
    2) 이 스크립트를 실행한다.
    3) 다시 `adb logcat -d -s FCM_CHECK` 로 "🚀 [수신 성공]" 을 확인한다.

    "전송 완료" 가 떠도 구독자가 0명이면 아무 기기에도 도착하지 않는다 —
    send() 의 성공은 FCM 서버 접수를 뜻할 뿐이다. 그래서 1) 을 반드시 먼저 한다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from alert.notifier import build_notifier
from core.schemas import AlertEvent
from alert.sender import dispatch
from alert.store import EventStore

# Windows 기본 콘솔(cp949)은 이모지를 못 그려 UnicodeEncodeError 로 죽는다.
# 발송은 이미 끝난 뒤라, 출력 실패로 성공 여부를 오판하지 않도록 방어한다.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main() -> None:
    parser = argparse.ArgumentParser(description="이안류 경보 한 건 발송")
    parser.add_argument("--notifier", default="console", choices=["console", "fcm"])
    parser.add_argument("--risk", default="emergency", choices=["watch", "warn", "emergency"])
    parser.add_argument("--in-rip", type=int, default=2, help="구역 내 인원")
    parser.add_argument("--total", type=int, default=5, help="화면 전체 인원")
    parser.add_argument("--rip-count", type=int, default=1, help="의심 구역 수")
    parser.add_argument("--source", default="sample_beach.mp4", help="영상 이름 (앱에 지점명으로 표시)")
    parser.add_argument("--image", default="", help="경보 이미지 경로. static/ 아래에 있어야 URL이 만들어진다")
    parser.add_argument("--no-store", action="store_true", help="DB에 기록하지 않는다")
    args = parser.parse_args()

    event = AlertEvent(
        risk_level=args.risk,
        persons_in_rip=args.in_rip,
        total_persons=args.total,
        rip_count=args.rip_count,
        frame_idx=412,
        timestamp_sec=13.7,
        video_source=args.source,
        snapshot_path=str(Path(args.image).resolve()) if args.image else "",
    )

    notifier = build_notifier(args.notifier)
    store = None if args.no_store else EventStore()
    result = dispatch(event, notifier, store)

    print(f"발송 방식: {notifier.name}")
    print(f"결과: {'성공' if result.ok else '실패'} · {result.detail}")
    if result.image_url:
        print(f"이미지 URL: {result.image_url}")
    if args.notifier == "fcm":
        print(
            "※ 구독자가 없어도 '성공' 으로 표시됩니다. 실제 도착 여부는 "
            "'adb logcat -d -s FCM_CHECK' 에서 확인하세요."
        )


if __name__ == "__main__":
    main()
