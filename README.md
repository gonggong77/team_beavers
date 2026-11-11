# 비버즈(BEAVERS) – 이안류 긴급 알리미

해안 CCTV 영상에서 **이안류 구역**과 **입수자(수영객)** 를 함께 탐지하고, 입수자가 이안류 구역에 들어가면 **안전요원 스마트폰으로 긴급 푸시 알림**을 보내는 AI 관제 시스템입니다.

> **현재 범위(MVP):** 동영상 업로드 기반 탐지·알림까지. 실제 CCTV 실시간 연동은 아직 포함하지 않습니다.

## 왜 만들었나

- 이안류는 자력 탈출이 어렵고, 익수 골든타임은 약 4분입니다.
- 기존 감시는 이안류 발생 여부만 보고, 구역 안에 사람이 있는지는 알려 주지 못합니다.
- 넓은 해역에서 작은 사람을 육안으로 찾는 데는 시간이 오래 걸립니다.

## 동작 흐름

1. 영상 업로드 → 분석 시작
2. 듀얼 AI 모델 추론 (이안류 구역 + 수영객)
3. 구역 침범 판정 → 위험 등급 산정 (관찰 / 경고 / 긴급)
4. 긴급 시 알림 게이트(연속 판정 + 쿨다운) 통과 여부 확인 → FCM 발송 (백그라운드 스레드)
5. 모바일 앱에서 알림 수신 + 웹 화면에 경보 기록 및 확인 처리

## 주요 기능

- **듀얼 모델:** 이안류 구역(YOLO11m)과 수영객(SAHI 타일 슬라이싱 + YOLO11m)을 별도 모델로 운용
- **구역 침범 판정:** 사람의 발 위치가 이안류 폴리곤 안(경계 15px 여유 포함)에 있는지 판정
- **스마트 알림 게이트:** 연속 판정 + 쿨다운으로 알림 난사를 막고, 위험 인원이 늘면 즉시 재발송
- **논블로킹 알림:** 스냅샷 저장과 FCM 발송을 백그라운드 스레드로 분리해 분석 속도 유지
- **경보 이력 관리:** SQLite에 저장하고 화면에서 "확인 완료" 처리

## 구조

```
rip_service_console/      웹 관제 콘솔 (Streamlit)
├── app.py                진입점
├── views/                화면 (영상 업로드·분석)
├── core/                 추론 파이프라인, 판정 규칙, 렌더링
└── alert/                FCM 발송, 알림 게이트, 이벤트 저장
rip_current_app/          Android 알림 앱 (Kotlin, RipCurrentAlert)
```

## 기술 스택

| 영역 | 기술 |
|---|---|
| 웹 UI | Streamlit |
| 영상 처리 | OpenCV |
| 탐지 모델 | Ultralytics YOLO11m, SAHI |
| 추론 가속 | PyTorch (GPU FP16) |
| 저장 | SQLite |
| 푸시 알림 | Firebase Cloud Messaging (firebase-admin) |
| 모바일 | Kotlin, Android |

## 실행 방법

```bash
cd rip_service_console
pip install -r requirements.txt
streamlit run app.py
```

- 모델 가중치(`*.pt`)와 Firebase 서비스 계정 키(`alert/serviceAccountKey.json`)는 저장소에 포함되어 있지 않습니다. 직접 준비해서 위치에 넣어야 합니다.
- 실기기 알림 테스트: 안드로이드 폰을 연결한 뒤 `adb reverse tcp:8501 tcp:8501`로 폰이 PC의 Streamlit 서버에 접근하도록 설정합니다.
- 알림 수신 앱은 `rip_current_app/rip_current_app/AndroidStudioProjects/RipCurrentAlert`를 Android Studio로 열어 설치합니다.

## 향후 과제

- 실제 CCTV 실시간 연동
- 관제 파라미터(임계값 등) UI 설정
- 관리자별 알림 타겟팅
