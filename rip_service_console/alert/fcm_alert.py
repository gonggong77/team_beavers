"""Firebase 연동. 경보 이미지 URL을 만들고 FCM 토픽으로 경보를 보낸다.

원본 core/fcm_alert.py 와 발송 로직은 같다. 경로(서비스 계정 키, static 폴더)만
이 폴더 단독으로도, 원본 프로젝트 안에서도 찾히도록 후보를 늘렸다.

이미지는 별도 스토리지에 올리지 않는다. static/ 아래 스냅샷을 Streamlit이
이미 /app/static/ 으로 서빙하고 있으므로, 그 위에 IMAGE_BASE_URL만 붙이면
된다 (`.streamlit/config.toml` 의 server.enableStaticServing 필요). 앱은
그 URL을 그대로 내려받는다.

로컬 시연은 `adb reverse tcp:8501 tcp:8501` 로 열어둔 127.0.0.1 을 쓴다
(이유는 IMAGE_BASE_URL 주석 참고). Streamlit Community Cloud 에 배포하면
RIP_IMAGE_BASE_URL 을 https://<앱이름>.streamlit.app 으로 바꾸기만 하면 된다 —
이때는 HTTPS라 adb reverse 도 안드로이드의 평문 허용 설정도 필요 없다.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import firebase_admin
from firebase_admin import credentials, messaging

# ⭐️ 기준 경로 명확화
_HERE = Path(__file__).resolve().parent          # alert/ 폴더
_PROJECT_ROOT = _HERE.parent                     # rip_current/ 프로젝트 루트

# 서비스 계정 키 탐색 순서
_KEY_CANDIDATES = (
    _HERE / "serviceAccountKey.json",
    _PROJECT_ROOT / "serviceAccountKey.json",
)


def _resolve_key() -> Path:
    env = os.getenv("FIREBASE_CREDENTIALS")
    if env:
        return Path(env)
    for candidate in _KEY_CANDIDATES:
        if candidate.exists():
            return candidate
    return _KEY_CANDIDATES[0]


SERVICE_ACCOUNT_PATH = _resolve_key()

# 로컬 시연 기본값. 앱이 이 주소로 오려면 adb reverse 로 포트를 열어둬야 한다:
#
#     adb reverse tcp:8501 tcp:8501
#
# 에뮬레이터 표준 주소인 10.0.2.2 를 쓰지 않는다. 그 NAT 별칭은 adb shell(shell UID)
# 에서는 붙지만 앱 프로세스(untrusted_app UID)에서는 10초 타임아웃이 난다 — 확인된 동작이다.
# adb reverse 는 adb 채널로 직접 넘겨 이 문제를 통째로 피한다.
#
# Cloud 배포 시에는 RIP_IMAGE_BASE_URL 을 https://<앱이름>.streamlit.app 으로 주면
# adb reverse 도, 앱의 평문 HTTP 허용도 필요 없어진다.
IMAGE_BASE_URL = os.getenv("RIP_IMAGE_BASE_URL", "http://127.0.0.1:8501")

# 스냅샷이 놓이는 곳. 이 폴더를 단독으로 돌리면 alert_only/static/,
# 원본 프로젝트 안에서 돌리면 team_beavers/static/ 이 쓰인다.
STATIC_DIR = _PROJECT_ROOT / "static"
SNAPSHOT_DIR = STATIC_DIR / "snapshots"
_STATIC_CANDIDATES = (STATIC_DIR,)

# 앱의 MainActivity.kt::ALERT_TOPIC 과 동일해야 한다.
TOPIC = "rip_current_alert"

# core/schemas.py::RiskLevel 과 동일한 3단계.
RISK_LEVELS = ("watch", "warn", "emergency")


def _ensure_app() -> None:
    if firebase_admin._apps:
        return
    if not SERVICE_ACCOUNT_PATH.exists():
        raise FileNotFoundError(
            f"서비스 계정 키를 찾을 수 없습니다: {SERVICE_ACCOUNT_PATH}\n"
            "Firebase 콘솔에서 발급받은 JSON을 이 경로에 두거나 FIREBASE_CREDENTIALS 로 경로를 지정하세요."
        )
    cred = credentials.Certificate(str(SERVICE_ACCOUNT_PATH))
    firebase_admin.initialize_app(cred)


def snapshot_url(image_path: str | Path) -> str:
    """static/ 아래의 스냅샷을 Streamlit이 서빙하는 URL로 바꾼다.

    이미지를 어디에도 올리지 않는다. Streamlit이 이미 띄워둔 서버가
    파일을 그대로 내보내고, 앱은 그 주소를 Coil로 내려받는다.

    static/ 밖의 파일이 들어오면 ValueError 를 낸다. 서빙되지 않는 주소를
    만들어 보내면 앱에서 이미지만 조용히 비어 원인을 찾기 어렵기 때문이다.
    """
    resolved = Path(image_path).resolve()
    for base in _STATIC_CANDIDATES:
        try:
            relative = resolved.relative_to(base)
        except ValueError:
            continue
        return f"{IMAGE_BASE_URL}/app/static/{relative.as_posix()}"
    raise ValueError(
        f"스냅샷이 static/ 밖에 있습니다: {resolved}\n"
        f"서빙되는 위치: {', '.join(str(p) for p in _STATIC_CANDIDATES)}"
    )


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
    """이안류 경보를 발송한다. 돌려주는 값은 FCM 메시지 ID다.

    risk_level/persons_in_rip/total_persons/rip_count/zone/camera_id/detected_at 는
    안드로이드 관제 화면(MainActivity.applyIntentData)이 그대로 읽는 필드다.
    이름을 하나라도 다르게 보내면 그 항목만 화면 기본값으로 표시된다.

    image_url 은 빈 문자열이어도 된다. 앱이 비어 있으면 이미지 로딩을
    건너뛰므로, URL 생성이 실패해도 경보 자체는 내보낼 수 있다.

    주의: send() 의 성공은 FCM 서버가 접수했다는 뜻일 뿐, 기기 도착을
    보장하지 않는다. 구독자가 0명이어도 '성공'으로 나온다.
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

    _ensure_app()

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
