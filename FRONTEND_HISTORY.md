# 프론트엔드(Streamlit) 개발 히스토리

팀 비버즈 이안류 조난자 긴급 탐지 알림 시스템 — 웹 데모(Streamlit) 파트 개발 기록.
담당: 이현경. 아래는 세션별/커밋별로 정리한 구현 흐름이다.

> 이 히스토리의 시작점: "Streamlit app을 모바일 앱으로 변환" 세션 — 처음엔 웹(Streamlit)
> 데모로 시작했다가, 실제 사용 시나리오(현장 요원이 폰으로 알림을 받는 구조)에 맞춰
> Android 앱(FCM 푸시 수신)을 덧붙이는 방향으로 확장되었다.

## 1. 초기 골격 — `프론트엔드 추가(streamlit)` (2a18797, 2026-09-14)

Streamlit 기반 웹 데모의 첫 커밋. 모델 파일이 없어도 바로 실행되도록 `FakeDetector`로
가상 탐지 결과를 만들어, 관찰 → 경고 → 긴급 전환과 알림 발송 흐름 전체를 모델 없이도
검증할 수 있게 설계했다.

- **`app.py`**: 라우팅 + 사이드바 전용. 판정 로직은 넣지 않는 원칙을 README에 명시.
- **`views/dashboard.py`**: 기본 화면. 샘플 CCTV(A구역) 영상을 0.7초 주기로 계속 재생하며
  상시 관제 화면을 흉내냄. 영상이 없으면 합성 배경으로 대체.
- **`views/test_run.py`**: 업로드한 영상을 분석하는 화면.
- **`core/`**: `scenario.py`(가상 해변 A~E 시나리오), `schemas.py`(팀 공통 데이터 계약 —
  FrameResult, AlertEvent), `detector.py`(Detector 프로토콜 + FakeDetector + YoloDetector),
  `model_probe.py`(모델 클래스명/imgsz 자동 인식), `rules.py`(위험 등급 판정 + 알림 스팸
  방지 게이트), `render.py`(오버레이 그리기), `notifier.py`(알림 어댑터), `store.py`
  (이벤트 SQLite 기록), `pipeline.py`(영상 순회 + 전체 흐름 결합, CLI 실행 가능).
- `docs/MODEL_CONTRACT.md`, `docs/NOTIFIER_CONTRACT.md`: 모델팀·알림 담당자에게 그대로
  전달할 인터페이스 계약 문서.
- 이 시점 저장소 루트에는 초기 프로토타입 시절의 `dashboard.py`, `scenario.py`(zip으로
  받은 잔재)도 함께 섞여 들어와 있었음 — 이후 `views/`, `core/` 구조로 완전히 대체되고
  정리 대상으로 남음(§6 참고).

## 2. Android 환경 준비 — `android studio 환경` (599b916, 2026-09-15)

Streamlit 데모와는 별도로, FCM 푸시를 실제 수신할 Android Studio 프로젝트
(`RipCurrentAlert/`)를 추가. `MainActivity.kt`, `MyFirebaseMessagingService.kt` 등
기본 골격 구성. 이때부터 "웹 데모 + 모바일 알림 수신"의 이중 구조가 시작됨.

이어서 실수로 커밋됐던 `serviceAccountKey.json`(Firebase 서비스 계정 키)을 곧바로
삭제하는 커밋(70ca385)이 따라붙음 — 키 파일은 커밋하지 않는다는 원칙이 이때 세워짐.

## 3. FCM 알림 배선 — `backup` (acfb706, 2026-09-15)

Streamlit 쪽에서 실제로 FCM을 쏘는 `core/fcm_alert.py`를 추가하고 `notifier.py`와
연동. `app.py`, `views/test_run.py`를 알림 발송 흐름에 맞게 수정.

## 4. 이중 모델 구조 + FCM 안정화 — `이중 모델(이안류+사람) 구조 및 FCM 알림 안정화` (092f133, 2026-09-17)

가장 큰 폭의 변경. 이안류 탐지 모델과 사람 탐지 모델을 함께 쓰는 이중 모델 구조로
`core/detector.py`, `core/pipeline.py`, `core/rules.py`를 대폭 수정. `.streamlit/config.toml`
추가로 Streamlit 실행 설정을 프로젝트에 포함. FCM 알림 안정화 작업과 함께
`REAL_TEST_GUIDE.md`(실기기 연동 테스트 가이드) 신설. `model_simulation/simulation.ipynb`도
이 커밋에서 대량 추가되어 모델 검증 작업이 노트북으로 병행됨.

## 5. 실기기 시연 문서화 — 09-18 (49954d8, c57b5f3)

- `REAL_TEST_GUIDE.md`에 실기기 무선(adb wireless) 시연 절차 보강.
- 판정 기준(임계값)을 완화하고, 알림 방식 기본값을 Android(FCM)로 변경 — 데모 시
  콘솔 로그가 아니라 실제 폰 알림이 기본으로 뜨도록 조정.

## 6. 폴더 분리 및 재구성 — 진행 중 (미커밋, 2026-09-21 기준)

시연/개발 편의를 위해 원본 앱을 역할별로 쪼개는 작업이 진행 중:

- **`ui_only/`**: 모델 추론·알림 발송을 걷어내고 화면 코드만 남긴 버전. GPU, 모델
  가중치, Firebase 키 없이 UI 작업자가 바로 띄워볼 수 있도록 분리.
- **`alert_only/`**: 반대로 알림 발송 관련 코드만 모은 버전. 모델 추론·관제 화면 없이
  GPU/가중치 없이 발송 로직(`alert/fcm_alert.py`, `gate.py`, `notifier.py`, `sender.py`,
  `store.py`)만 붙잡고 작업 가능.
- **`QUICK_DEMO_RUN.md`**: 실기기 무선 시연을 복붙 수준으로 바로 실행할 수 있도록 IP
  설정까지 채워 넣은 요약 절차서 (`REAL_TEST_GUIDE.md`의 축약판).
- 루트에 남아있던 구시대 잔재 `dashboard.py`, `scenario.py`(§1에서 언급한 초기 zip
  잔재)를 삭제하고 `views/`, `core/` 쪽 최신 버전으로 정리.
- `requirements.txt`, `core/detector.py`, `core/rules.py` 등도 이중 모델 구조 안정화에
  맞춰 추가 수정 중.

### `rip_service_console/` — 팀원이 이어받아 재구성한 버전

`rip_service_console/`는 원래 이현경이 금요일(2026-09-18)까지 작업해 둔 웹 데모를
팀원이 넘겨받아 직접 구조를 재정리해서 다시 준 폴더다. 그 사이의 세부 작업 과정은
이 히스토리에 기록되어 있지 않고, 결과물만 새 폴더로 들어왔다. 주요 변화:

- `core/notifier.py` + `core/fcm_alert.py`에 섞여 있던 알림 로직을 `alert/` 패키지로
  독립시킴(`alert/fcm_alert.py`, `alert/gate.py`, `alert/notifier.py`, `alert/schemas.py`,
  `alert/sender.py`, `alert/store.py`, `alert/send_test.py`) — `alert_only/` 분리와
  같은 방향의 리팩터링을 본 구조에도 반영한 형태.
- `views/test_run.py` → `views/video_analysis.py`로 이름 변경.
- `core/demo_feed.py` 신설 (관제 화면용 데모 피드 로직 분리).
- 모델 가중치(`best_swimmer_yolo11m_b8.pt`, `best_yolo11m_integrated_v2_first.pt`)와
  테스트 영상(`test_video/rip_current_testv1.mp4`)을 폴더 안에 직접 포함.
- ⚠️ `alert/serviceAccountKey.json`이 폴더 안에 그대로 들어있음 — §2에서 한 번 삭제
  이력이 있는 파일과 같은 종류이므로, 커밋 전에 반드시 제외해야 함.

## 요약 흐름

```
Streamlit 웹 데모 첫 골격 (views/ + core/, FakeDetector)
        ↓
Android 앱 골격 추가 (FCM 수신 준비)
        ↓
Streamlit → FCM 알림 실제 연동
        ↓
이중 모델(이안류+사람) 구조로 탐지 로직 고도화 + FCM 안정화
        ↓
실기기 무선 시연 절차 문서화 + 판정 기준/알림 기본값 조정
        ↓
역할별 폴더 분리(ui_only / alert_only) + 루트 잔재 정리   ← 이현경 작업분 (~09-18)
        ↓
rip_service_console/ — 팀원이 구조 재정리해서 인계         ← 팀원 작업분 (09-21 반영)
```
