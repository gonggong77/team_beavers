# 이안류 경보 — 알림 발송 전용 폴더

원본 앱에서 **알림 발송에 관련된 코드만** 모은 폴더입니다.
모델 추론도, 관제 화면도 들어 있지 않아 GPU나 가중치 없이 발송만 붙잡고 작업할 수 있습니다.

## 실행

```bash
cd alert_only
pip install -r requirements.txt

streamlit run app.py          # 발송 콘솔 화면
python send_test.py           # 화면 없이 한 건 발송 (콘솔 출력)
```

FCM으로 실제 발송하려면 `serviceAccountKey.json` 이 필요합니다 (아래 참고).
없어도 `console` 방식으로 문안과 흐름은 전부 확인됩니다.

## 폴더 구조

```
alert_only/
├─ app.py                  # 발송 콘솔 (Streamlit). 경보 값을 채워 실제로 보내 본다
├─ send_test.py            # 화면 없이 한 건 발송하는 CLI
├─ core/
│  ├─ schemas.py           # AlertEvent — 팀 공통 계약. 탐지 쪽에서 이 모양으로 넘어온다
│  ├─ notifier.py          # ★ ConsoleNotifier / FcmNotifier / build_notifier
│  ├─ fcm_alert.py         # ★ Firebase 실제 발송 + 스냅샷 URL 생성
│  ├─ sender.py            # 기록 → 발송 → 결과 반영 (원본 pipeline 의 발송 부분)
│  ├─ gate.py              # AlertGate — 스팸 방지 (연속 판정 + 쿨다운)
│  └─ store.py             # 이벤트/발송 상태 SQLite 기록
├─ .streamlit/config.toml  # static/ 서빙 켜기 (이미지 URL이 이것에 의존)
└─ static/snapshots/       # 경보 이미지 저장 위치 (앱이 여기 주소로 내려받는다)
```

★ 표시가 알림 작업 시 주로 손댈 파일입니다.

## 발송 경로 한눈에

```
AlertEvent (탐지 쪽이 만듦)
   └─ sender.dispatch(event, notifier, store)
        ├─ store.add(event)                 기록 먼저 (발송 실패해도 이력은 남는다)
        ├─ notifier.send(event)
        │    └─ FcmNotifier
        │         ├─ fcm_alert.snapshot_url(...)          static/ 경로 → http URL
        │         └─ fcm_alert.send_rip_current_alert(...) FCM 토픽 발송
        └─ store.update_notify(...)         sent / failed 반영
```

## 지켜야 할 규칙 두 가지

1. **`send()` 는 예외를 밖으로 던지지 않습니다.** 실패는 `SendResult(ok=False, detail=...)` 로
   돌려주세요. 알림 실패가 영상 분석을 멈추면 안 됩니다.
2. **인증 정보를 코드에 적지 마세요.** `serviceAccountKey.json` 은 `.gitignore` 대상입니다.

## 서비스 계정 키

찾는 순서는 이렇습니다.

1. 환경변수 `FIREBASE_CREDENTIALS` 가 가리키는 경로
2. `alert_only/serviceAccountKey.json`  ← 이 폴더만 받았을 때 여기에 두세요
3. 원본 프로젝트의 `rip_current_app/rip_current_app/serviceAccountKey.json`

## 환경변수

| 이름 | 기본값 | 설명 |
|---|---|---|
| `FIREBASE_CREDENTIALS` | (위 2·3번 순서로 탐색) | 서비스 계정 키 경로 |
| `RIP_IMAGE_BASE_URL` | `http://127.0.0.1:8501` | 앱이 이미지를 받아갈 주소 |
| `RIP_ALERT_LAT` / `RIP_ALERT_LON` | 35.1587 / 129.1604 | 경보 좌표 (앱 지도 표시용) |
| `RIP_ALERT_CAMERA_ID` | `CCTV-E01` | 카메라 ID |

`.env` 파일은 로드되지 않습니다. 실행하는 셸에서 직접 지정하세요.

```bash
RIP_IMAGE_BASE_URL=https://내앱.streamlit.app streamlit run app.py
```

## 실기기 시연 시 주의

- **토픽 이름**은 앱의 `MainActivity.kt::ALERT_TOPIC` 과 글자 하나까지 같아야 합니다
  (`rip_current_alert`). 다르면 아무도 못 받으면서 발송은 '성공' 으로 표시됩니다.
- **발송 성공 ≠ 도착.** 구독자가 0명이어도 성공으로 나옵니다.
  실제 수신은 `adb logcat -d -s FCM_CHECK` 의 `🚀 [수신 성공]` 으로 확인하세요.
- **이미지가 안 뜰 때**: 로컬 시연이면 `adb reverse tcp:8501 tcp:8501` 을 먼저 실행해야 합니다.
  에뮬레이터 주소 `10.0.2.2` 는 앱 프로세스에서 타임아웃이 납니다 (adb shell 에서만 붙습니다).
- **이미지는 `static/` 아래에 있어야** URL이 만들어집니다. 그 밖의 경로를 주면
  `snapshot_url` 이 ValueError 를 내고, 경보는 이미지 없이 나갑니다.

## 원본 앱에 합칠 때

원본 `core/pipeline.py` 가 `AlertEvent` 를 만든 뒤 하는 일이 `sender.dispatch()` 와 같습니다.
탐지 쪽 코드는 그대로 두고, 이 폴더의 `core/notifier.py`·`core/fcm_alert.py` 를
원본 `core/` 에 덮어쓰면 됩니다 (`schemas.py` 의 `AlertEvent` 필드가 같은지만 확인하세요).
`core/gate.py` 의 `AlertGate` 는 원본에서 `core/rules.py` 안에 있습니다.
