"""이안류 경보를 안드로이드 앱에 FCM 토픽으로 발송한다.

토큰이 아니라 토픽(TOPIC) 구독 방식을 쓴다. 앱이 켜질 때 스스로
`MainActivity.ALERT_TOPIC` 을 구독하므로, 기기가 바뀌거나 재설치돼도
토큰을 Logcat에서 옮겨 적을 필요가 없다. 구독한 모든 기기가 동시에 받는다.

주의: 아래 TOPIC 문자열은 앱의 MainActivity.kt::ALERT_TOPIC 과
글자 하나까지 같아야 한다. 다르면 아무도 못 받으면서도 발송 자체는
'성공'으로 표시된다 (아래 실행 방법 참고).

실행 방법
    cd rip_current_app/rip_current_app
    python send_alert.py

    1) 먼저 앱을 실행하고 `adb logcat -d -s FCM_CHECK` 로
       "✅ 토픽 구독 완료: rip_current_alert" 가 찍혔는지 확인한다.
       (구독 전파에는 수 초 걸릴 수 있다.)
    2) 이 스크립트를 실행한다.
    3) 다시 `adb logcat -d -s FCM_CHECK` 로 "🚀 [수신 성공]" 을 확인한다.

    이 스크립트가 "전송 완료"를 출력해도 구독자가 0명이면 아무 기기에도
    도착하지 않는다 — send()의 성공은 FCM 서버 접수를 뜻할 뿐 수신을
    보장하지 않는다. 그래서 1)번 확인을 반드시 먼저 한다.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import firebase_admin
from firebase_admin import credentials, exceptions, messaging

# Windows 기본 콘솔(cp949)은 이모지를 못 그려 UnicodeEncodeError 로 죽는다.
# 발송 자체는 이미 끝난 뒤라 출력 실패로 성공 여부를 오판하지 않도록 방어한다.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 앱의 MainActivity.kt::ALERT_TOPIC 과 동일해야 한다.
TOPIC = "rip_current_alert"

# core/schemas.py::RiskLevel 과 동일한 3단계.
RISK_LEVELS = ("watch", "warn", "emergency")

# image_url 변수 저장
IMAGE_URL = "https://wimg.dt.co.kr/news/cms/2026/08/04/news-p.v1.20260804.dbde925d8a254dee9ed6f55c7742fc57_P1.png"

_SERVICE_ACCOUNT_PATH = Path(__file__).resolve().parent / "serviceAccountKey.json"

if not firebase_admin._apps:
    if not _SERVICE_ACCOUNT_PATH.exists():
        raise FileNotFoundError(
            f"서비스 계정 키를 찾을 수 없습니다: {_SERVICE_ACCOUNT_PATH}\n"
            "Firebase 콘솔에서 발급받은 JSON을 이 경로에 두세요."
        )
    cred = credentials.Certificate(str(_SERVICE_ACCOUNT_PATH))
    firebase_admin.initialize_app(cred)


def send_rip_current_alert(
    location_name: str,
    person_count: int,
    latitude: float,
    longitude: float,
    image_url: str,
    risk_level: str = "emergency",
    persons_in_rip: int | None = None,
    total_persons: int | None = None,
    rip_count: int = 1,
    zone: str | None = None,
    camera_id: str = "CCTV-E01",
    detected_at: str | None = None,
) -> str:
    """이안류 경보를 발송한다.

    risk_level/persons_in_rip/total_persons/rip_count/zone/camera_id/detected_at 는
    안드로이드 관제 화면(MainActivity.applyIntentData)이 그대로 읽는 필드다.
    이름을 하나라도 다르게 보내면 그 항목만 화면 기본값으로 표시된다.
    """
    if risk_level not in RISK_LEVELS:
        raise ValueError(f"risk_level 은 {RISK_LEVELS} 중 하나여야 합니다: {risk_level!r}")

    if persons_in_rip is None:
        persons_in_rip = person_count
    if total_persons is None:
        total_persons = person_count
    if zone is None:
        zone = location_name
    if detected_at is None:
        detected_at = datetime.now().strftime("%H:%M:%S")

    # notification 키를 제거하고 data와 high priority만 전송한다.
    # 이렇게 해야 앱이 꺼져 있어도 안드로이드 서비스 코드가 직접 헤즈업 배너를 생성한다.
    # 이 구조는 의도된 것이므로 notification 블록을 다시 넣지 말 것 — 배너가 이중으로 뜬다.
    message = messaging.Message(
        data={
            "title": "⚠️ 이안류 위험 경보 발생!",
            "body": f"{location_name}에서 이안류 위험이 감지되었습니다.",
            "location_name": location_name,
            "person_count": str(person_count),
            "latitude": str(latitude),
            "longitude": str(longitude),
            "image_url": image_url,
            "risk_level": risk_level,
            "persons_in_rip": str(persons_in_rip),
            "total_persons": str(total_persons),
            "rip_count": str(rip_count),
            "zone": zone,
            "camera_id": camera_id,
            "detected_at": detected_at,
        },
        android=messaging.AndroidConfig(
            priority="high",
        ),
        topic=TOPIC,
    )

    return messaging.send(message)


if __name__ == "__main__":
    try:
        # 앱의 대기 화면(푸시 전 기본값)과 값을 맞춰서, 도착해도 화면이 튀지 않게 한다.
        res = send_rip_current_alert(
            location_name="해운대 해수욕장 2번 망루 앞", # 테스트 완료
            person_count=3, # 테스트 완료
            latitude=35.1587,
            longitude=129.1604,
            image_url=IMAGE_URL, # 테스트 완료
            persons_in_rip=3, # 테스트 완료
            total_persons=3, # 테스트 완료
            rip_count=2, # 테스트 완료
            camera_id="CCTV-E02", # 테스트 완료
        )
        print(f"전송 완료 (ID: {res})")
        print(f"토픽: {TOPIC}")
        print(
            "※ 구독자가 없어도 위 메시지는 뜹니다. 실제 도착 여부는 "
            "'adb logcat -d -s FCM_CHECK' 에서 🚀 [수신 성공] 을 확인하세요."
        )
    except exceptions.FirebaseError as e:
        print(f"전송 실패: {e.code} - {e}")
        print("확인할 것: 토픽 이름 오타, serviceAccountKey.json 의 프로젝트 일치 여부")
