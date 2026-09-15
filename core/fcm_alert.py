"""Firebase 연동. 스냅샷을 Storage에 올리고 FCM 토픽으로 경보를 보낸다.

Firebase 초기화는 반드시 이 모듈 한 곳에서만 한다. firebase_admin은
initialize_app을 한 번만 받아들이므로, 다른 곳에서 storageBucket 없이
먼저 초기화해 버리면 업로드가 조용히 깨진다.

이미지는 바이트를 그대로 푸시에 실을 수 없다 — FCM data 페이로드는 4KB
제한이라 썸네일조차 들어가지 않는다. 그래서 Storage에 올린 뒤 만료형
서명 URL만 싣고, 앱이 그 URL을 받아 내려받는다. 서명 URL은 Storage 보안
규칙을 거치지 않으므로 앱에 Firebase Storage SDK를 넣을 필요가 없다.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import firebase_admin
from firebase_admin import credentials, messaging, storage

_REPO_ROOT = Path(__file__).resolve().parent.parent

SERVICE_ACCOUNT_PATH = Path(
    os.getenv("FIREBASE_CREDENTIALS")
    or _REPO_ROOT / "rip_current_app" / "rip_current_app" / "serviceAccountKey.json"
)

# google-services.json 의 storage_bucket 과 같아야 한다. gs:// 접두어는 붙이지 않는다.
STORAGE_BUCKET = os.getenv("FIREBASE_STORAGE_BUCKET", "rip-current-alert.firebasestorage.app")

# 앱의 MainActivity.kt::ALERT_TOPIC 과 동일해야 한다.
TOPIC = "rip_current_alert"

# core/schemas.py::RiskLevel 과 동일한 3단계.
RISK_LEVELS = ("watch", "warn", "emergency")

# V4 서명의 만료 상한. 이걸 넘기면 서명 자체가 거부된다.
MAX_SIGNED_URL_DAYS = 7

MAX_EDGE_PX = 1280
JPEG_QUALITY = 80


def _ensure_app() -> None:
    if firebase_admin._apps:
        return
    if not SERVICE_ACCOUNT_PATH.exists():
        raise FileNotFoundError(
            f"서비스 계정 키를 찾을 수 없습니다: {SERVICE_ACCOUNT_PATH}\n"
            "Firebase 콘솔에서 발급받은 JSON을 이 경로에 두거나 FIREBASE_CREDENTIALS 로 경로를 지정하세요."
        )
    cred = credentials.Certificate(str(SERVICE_ACCOUNT_PATH))
    firebase_admin.initialize_app(cred, {"storageBucket": STORAGE_BUCKET})


def upload_snapshot(image_path: str | Path, expires_days: int = MAX_SIGNED_URL_DAYS) -> str:
    """스냅샷을 Storage에 올리고 만료형 서명 URL을 돌려준다.

    원본 해상도 그대로 올리지 않는다. 휴대폰 화면에 띄우는 용도라 긴 변
    1280px면 충분하고, 업로드 지연과 저장 용량이 함께 줄어든다.

    오래된 파일 정리는 코드가 아니라 버킷의 수명주기 규칙이 담당한다
    (Cloud Storage 콘솔 → 수명 주기 → Age > 14일이면 삭제).
    """
    if expires_days > MAX_SIGNED_URL_DAYS:
        raise ValueError(f"서명 URL은 최대 {MAX_SIGNED_URL_DAYS}일까지만 유효합니다: {expires_days}")

    path = Path(image_path)
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"이미지를 읽을 수 없습니다: {path}")

    height, width = image.shape[:2]
    scale = MAX_EDGE_PX / max(height, width)
    if scale < 1.0:
        image = cv2.resize(
            image,
            (round(width * scale), round(height * scale)),
            interpolation=cv2.INTER_AREA,
        )

    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise ValueError(f"JPEG 인코딩에 실패했습니다: {path}")

    _ensure_app()
    # 파일명에 이미 타임스탬프와 event_id가 들어있다 (core/pipeline.py 참고).
    blob = storage.bucket().blob(f"snapshots/{datetime.now():%Y%m%d}/{path.name}")
    blob.upload_from_string(buffer.tobytes(), content_type="image/jpeg")

    return blob.generate_signed_url(
        version="v4",
        expiration=timedelta(days=expires_days),
        method="GET",
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
    """이안류 경보를 발송한다.

    risk_level/persons_in_rip/total_persons/rip_count/zone/camera_id/detected_at 는
    안드로이드 관제 화면(MainActivity.applyIntentData)이 그대로 읽는 필드다.
    이름을 하나라도 다르게 보내면 그 항목만 화면 기본값으로 표시된다.

    image_url 은 빈 문자열이어도 된다. 앱이 비어 있으면 이미지 로딩을
    건너뛰므로, 업로드가 실패해도 경보 자체는 내보낼 수 있다.
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
