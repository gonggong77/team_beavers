# 이안류 조난자 탐지 — 화면(UI) 전용 버전

원본 앱에서 **모델 추론**과 **알림 발송**을 걷어내고 **화면 코드만** 남긴 폴더입니다.
UI 작업을 하는 사람이 모델 가중치·GPU·Firebase 키 없이 바로 띄워 볼 수 있습니다.

## 실행

```bash
cd ui_only
pip install -r requirements.txt
streamlit run app.py
```

가중치 파일도, `.env` 도, GPU도 필요 없습니다. 설치는 1분이면 끝납니다.

## 무엇이 빠져 있나

| 빠진 것 | 원본 위치 | 이 폴더에서는 |
|---|---|---|
| YOLO 추론 (이안류/사람) | `core/detector.py`, `core/model_probe.py` | 없음 — 화면에 뜨는 구역·사람은 전부 시연용 값 |
| 분석 파이프라인 | `core/pipeline.py` | `core/demo_feed.py` 가 같은 모양의 결과만 흘려보냄 |
| FCM/콘솔 알림 발송 | `core/notifier.py`, `core/fcm_alert.py` | 없음 — 어디로도 메시지가 나가지 않음 |
| 이벤트 DB (SQLite) | `core/store.py` | `st.session_state` 에만 기록 (새로고침하면 사라짐) |
| 알림 게이트·트랙 스무딩 | `core/rules.py` 의 `AlertGate`, `TrackSmoother` | 없음 |
| 관제 화면의 실제 추론 모드 | `core/scenario.py` 의 hybrid / real | scenario 모드만 남김 |

> 화면에 보이는 이안류 구역과 사람 박스는 **실제 탐지 결과가 아닙니다.**
> 레이아웃·색·문구·반응 속도를 확인하기 위한 값입니다.

## 폴더 구조

```
ui_only/
├─ app.py                 # 진입점. 사이드바, 화면 모드, CSS
├─ core/
│  ├─ schemas.py          # 화면이 그리는 데이터의 모양 (팀 공통 계약과 동일)
│  ├─ rules.py            # 등급 판정 + 구역 침범 판정 (화면 색을 정하는 최소 규칙)
│  ├─ render.py           # ★ 오버레이 그리기 + 등급 표시줄/상황 패널 HTML
│  ├─ scenario.py         # 관제 화면용 가상 구역 A~E 영상·표시 데이터
│  └─ demo_feed.py        # 업로드 영상을 '분석하는 것처럼' 재생
├─ views/
│  ├─ dashboard.py        # ★ 상시 관제 화면
│  └─ test_run.py         # ★ 테스트 영상 재생 화면
└─ requirements.txt
```

★ 표시가 UI 작업 시 주로 손댈 파일입니다.

## 화면 두 개

1. **상시 관제 화면** (기본): 가상 구역 A~E 를 10초마다 자동 순환. 0.7초 주기로 갱신.
2. **테스트 영상 화면**: 사이드바에서 영상을 올리면 전환. `재생 시작` 을 누르면
   관찰 → 경고 → 긴급 순서로 세 등급이 모두 한 번씩 나옵니다 (긴급 구간에서 경보 카드·
   이벤트 표·경보 이미지 갤러리가 채워집니다).

사이드바 `⚙️ 설정` 에서 **개발자 모드**로 바꾸면 프레임 간격, 최대 표시 장수,
발끝 기준점 표시가 나타납니다. 전부 화면 표시에만 영향을 줍니다.

## 화면을 고칠 때

- **색·문구·배지·패널 레이아웃** → `core/render.py`
  (`RISK_HEX`, `RISK_DESC`, `risk_bar_html`, `status_panel_html`)
- **오버레이(구역 채움, 사람 박스, 상단 띠)** → `core/render.py` 의 `draw_overlay`, `_draw_banner`
- **관제 화면 구성·순환 간격** → `views/dashboard.py` (`REFRESH_SEC`), `core/scenario.py` (`ROTATE_SEC`, `BEACHES`)
- **영상 화면 구성·이벤트 표** → `views/test_run.py`
- **전체 CSS·사이드바** → `app.py` (`_HEADER_CSS`, `sidebar`)

`core/scenario.py` 의 `BEACHES` 에서 구역별 `has_rip` / `person_in_rip` 을 바꾸면
관제 화면에서 원하는 등급 조합을 바로 만들어 볼 수 있습니다.

## 관제 화면 배경 영상 (선택)

`ui_only/data/cctv/` 에 `A.mp4` ~ `E.mp4` 를 넣으면 그 영상을 배경으로 씁니다.
(이 폴더가 원본 프로젝트 안에 있으면 `../data/cctv/` 도 자동으로 찾습니다.)
영상이 하나도 없으면 합성 해변 배경을 그려서라도 화면은 정상적으로 뜹니다.

## 원본 앱에 합칠 때

화면 코드는 `FrameResult` / `AlertEvent`(`core/schemas.py`)와
`step.annotated_bgr / result / risk / progress / event` 모양에만 의존합니다.
원본의 `core.pipeline.run()` 이 같은 모양을 돌려주므로,
`views/test_run.py` 의 `from core.demo_feed import play` 를 그쪽으로 바꾸는 것이
합치는 작업의 대부분입니다.
