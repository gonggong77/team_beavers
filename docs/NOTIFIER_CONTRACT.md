# 알림 연동 요청서

수신: 김종호 · 발신: 이현경(웹) · 대상 파일: `core/notifier.py`

> 텔레그램 알림은 더 이상 쓰지 않아 코드에서 제거했습니다. 알림은 이제
> `ConsoleNotifier`(화면 기록용 기본값)와 `FcmNotifier`(안드로이드 앱 푸시) 둘뿐입니다.

## 해야 할 일은 하나입니다

`FcmNotifier.send(event) -> SendResult` (`core/notifier.py`) 안쪽을 다듬어 주세요.
실제 발송 로직은 `core/fcm_alert.py` 의 `send_rip_current_alert` 에 있습니다.
웹 코드는 전혀 건드릴 필요가 없습니다.

## 받게 되는 데이터

`core/schemas.py` 의 `AlertEvent` 입니다. JSON 직렬화가 항상 보장됩니다.

| 필드 | 타입 | 설명 |
| --- | --- | --- |
| `event_id` | str | 12자리 고유 ID |
| `occurred_at` | str | 발생 시각 (ISO8601, 타임존 포함) |
| `risk_level` | str | `watch` / `warn` / `emergency` |
| `persons_in_rip` | int | 이안류 의심 구역 **안**의 인원 수 |
| `total_persons` | int | 화면 전체 인원 수 |
| `rip_count` | int | 의심 구역 개수 |
| `frame_idx` | int | 프레임 번호 |
| `timestamp_sec` | float | 영상 내 위치(초) |
| `video_source` | str | 영상 파일명 |
| `snapshot_path` | str | 오버레이가 그려진 현장 캡처 이미지 경로 |
| `note` | str | 안전 문구. 메시지 하단에 반드시 포함 |

`event.to_dict()` 로 dict, `event.summary_text()` 로 기본 메시지 문안을 얻을 수 있습니다.

## 단독 테스트

웹이 없어도 됩니다.

```bash
NOTIFIER=fcm python -m core.notifier
```

샘플 `AlertEvent` 하나를 만들어 실제로 발송합니다. `serviceAccountKey.json` 이 필요합니다
(REAL_TEST_GUIDE.md 4단계 참고).

전체 흐름을 보려면 CLI로 영상을 돌려보세요.

```bash
python -m core.pipeline data/sample_beach.mp4 --skip 10 --notifier fcm
```

## 지켜야 할 규칙 두 가지

**1. 예외를 절대 밖으로 던지지 마세요.**
알림 실패가 영상 분석을 멈추면 안 됩니다. 실패는 `SendResult(ok=False, detail=...)` 로 돌려주세요.
웹은 이 값을 받아 이벤트 로그에 `failed` 로 기록하고 계속 진행합니다.

**2. 인증 정보를 코드에 적지 마세요.** `serviceAccountKey.json` 은 `.gitignore` 대상이고,
배포 시에는 `st.secrets` 로 옮겨야 합니다 (REAL_TEST_GUIDE.md 참고).

## 스팸 방지는 웹에서 이미 처리합니다

`core/rules.py` 의 `AlertGate` 가 아래 두 조건을 모두 통과한 경우에만 `send()` 를 호출합니다.

- 긴급 판정이 N회 연속 (기본 3회)
- 마지막 발송 후 N초 경과 (기본 30초)

30fps 영상에서 긴급이 200프레임 이어져도 메시지는 1통만 갑니다.
시연 중 알림이 폭주하는 사고를 막는 장치이니, 알림 쪽에서 중복 제거를 또 넣을 필요는 없습니다.
