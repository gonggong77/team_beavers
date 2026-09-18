# 실제 모델 적용 & 통합 테스트 가이드

학습한 모델을 붙이고, 영상 분석부터 안드로이드 앱 알림까지 실제로 동작하는지 확인하는 절차.

> 모델팀에 **무엇을 요청할지**는 [docs/MODEL_CONTRACT.md](docs/MODEL_CONTRACT.md),
> 프로젝트 구조 전반은 [README.md](README.md) 를 보세요. 이 문서는 **받은 뒤 어떻게 검증하는지**만 다룹니다.

핵심 원칙: **한 번에 다 켜서 테스트하지 마세요.** 아래 1→4단계 순서를 지키면
문제가 생겼을 때 어느 구간이 원인인지 바로 갈립니다. 반대로 하면 원인 찾는 데 몇 시간 걸립니다.

---

## 0. 준비물

| 항목 | 확인 |
| --- | --- |
| 이안류 모델 | `models/rip/` 또는 `.env` 의 `RIP_MODEL_PATH` (기존처럼 `models/` 바로 아래도 인식) |
| 사람(표류자) 모델 | `models/person/` 또는 `.env` 의 `PERSON_MODEL_PATH` |
| ultralytics | `requirements.txt` 주석 해제 + **모델팀과 같은 버전** |
| task | 이안류는 `segment` 권장 (`detect` 도 동작하지만 구역이 사각형이 됨 — 1단계 ② 참고) |
| 테스트 영상 | 관찰 → 경고 → 긴급 전환이 한 클립에 보이는 10~20초 분량 |

> 모델은 **두 개**입니다. 이안류 모델과 사람 모델을 따로 돌려
> `core/detector.py` 의 `DualDetector` 가 결과를 합치고, 사람이 이안류 구역 안에 있는지 판정합니다.
> 한쪽만 받은 상태여도 그 쪽만 실제 추론으로 돌고 나머지는 데모(Fake)로 채워집니다.

```bash
# 가상환경 활성화 (Windows Git Bash 기준)
source .venv/Scripts/activate

# ultralytics 설치 (requirements.txt 주석 해제 후)
pip install -r requirements.txt
```

---

## 1단계 — 모델만 검증 (폰·웹 없이)

가장 먼저, 가장 빠르게 확인할 수 있는 단계입니다. **여기서 막히면 뒷 단계는 볼 필요도 없습니다.**

```bash
python -m core.pipeline data/sample_beach.mp4 --notifier console --skip 5
```

### 확인할 것 3가지

**① 첫 줄에 두 모델이 모두 `YOLO` 로 찍히는가**

```
detector: rip=YOLO person=YOLO names={0: 'rip_current', 1: 'person'}          ← 정상
detector: rip=YOLO person=Fake(데모 모드) names={...}   ← 사람 모델을 못 찾음 (사람이 가짜로 그려짐)
```

`Fake` 가 나온 쪽은 가중치를 못 찾은 것입니다. 찾는 순서 (`core/detector.py`):

| 모델 | 순서 |
| --- | --- |
| 이안류 | `.env` 의 `RIP_MODEL_PATH` → `models/rip/*.pt` → `models/*.pt` (파일명에 `person` 이 들어간 건 건너뜀) |
| 사람 | `.env` 의 `PERSON_MODEL_PATH` → `models/person/*.pt` |

**경로가 설정돼 있는데 파일이 없으면 조용히 데모 모드로 빠집니다.** 오타를 제일 먼저 의심하세요.

> ⚠️ **사람 모델이 `Fake` 인 채로 실제 영상을 분석하면, 가짜 사람이 진짜 이안류 구역 안에
> 들어가서 진짜 긴급 알림(FCM 푸시 포함)이 나갑니다.** 시연 전 사이드바의
> `모델 · 이안류 … / 사람 …` 표시로 어느 쪽이 실제인지 반드시 확인하세요.

**② 등급이 실제로 변하는가**

```
frame  160 t=  5.33s emergency in-zone=1 total=4
frame  220 t=  7.33s warn      in-zone=0 total=4
frame  280 t=  9.33s watch     in-zone=0 total=4
```

- **`watch` 만 계속 나온다** → 이안류 모델이 구역을 하나도 못 찾은 것입니다. 구역이 0개면
  `grade()` 는 항상 `watch` 입니다. conf 임계값을 낮춰 보세요.
- **`warn` 까지만 가고 `emergency` 가 안 나온다** → 구역은 찾았는데 **구역 안에 사람이 없는** 것입니다.
  사람 모델이 `Fake` 이거나, 원거리 CCTV의 작은 사람을 conf 0.25 로는 못 잡는 경우가 흔합니다.

`detect` 로 학습한 이안류 모델도 동작합니다. 마스크가 없으면 박스를 사각형 폴리곤으로 대신 쓰기
때문입니다. 다만 사각형은 실제 이안류보다 넓어서 **구역 안/밖 판정이 후해집니다**(오탐 방향).
정확도가 필요하면 `segment` 로 재학습하는 편이 좋습니다.

**③ 맨 끝에 발송 건수가 찍히는가**

```
완료. 발송 이벤트 1건
```

0건이면 ②를 다시 보거나, 영상에 긴급 상황이 3프레임 연속으로 잡히는 구간이 없는 것입니다
(기본 `min_consecutive=3`).

---

## 2단계 — 클래스 매핑 확인

모델의 클래스 인덱스를 코드가 제대로 알아냈는지 봅니다.

```bash
python -c "
from core.detector import build_detector
d, rip_is_real, person_is_real = build_detector()
print('이안류 모델 :', '실제' if rip_is_real else '데모(Fake)')
print('사람 모델   :', '실제' if person_is_real else '데모(Fake)')
print('이안류 클래스 :', d.rip_class_ids, getattr(d.rip_detector, 'names', {}))
print('사람 클래스   :', d.person_class_ids, getattr(d.person_detector, 'names', {}))
"
```

모델이 둘로 나뉘었기 때문에 **각 모델에서 어떤 클래스를 쓰기로 했는지**가 핵심입니다.
`DualDetector` 는 클래스 이름이 힌트(`rip`/`person` 등)에 걸리면 **그 클래스만** 쓰고,
하나도 안 걸리면 전용 모델로 보고 전체 클래스를 씁니다.
그래서 사람 모델로 COCO 사전학습 모델(`yolo11n.pt`, 80클래스)을 붙여도
`person` 만 골라내고 배·새·차는 세지 않습니다.

자동 매핑 규칙 ([core/model_probe.py](core/model_probe.py)):

| 분류 | 이름에 이게 들어 있으면 |
| --- | --- |
| 이안류 | `rip`, `current`, `ripcurrent`, `rip_current`, `이안` |
| 사람 | `person`, `people`, `human`, `swimmer`, `사람` |

### ⚠️ 매핑이 틀렸을 때

모델이 둘로 나뉜 뒤로는 **한 모델 안에서 이안류와 사람을 구분할 일이 없어서** 매핑 사고가
크게 줄었습니다. 이안류 모델이 찾은 건 전부 구역, 사람 모델이 찾은 건 전부 사람입니다.

남아 있는 위험은 하나입니다 — **한 모델에 목적과 무관한 클래스가 섞여 있는 경우.**
이름이 힌트에 하나도 안 걸리면 전체 클래스를 그 용도로 쓰기 때문입니다.
(예: 이안류 모델에 `{0:'rip', 1:'wave'}` 가 있으면 `rip` 만 씁니다. 반면
`{0:'zone_a', 1:'zone_b'}` 처럼 힌트에 안 걸리면 둘 다 이안류 구역으로 씁니다.)

- (권장) 모델 학습 시 클래스 이름을 `rip_current`, `person` 으로 맞추기
- 확인 방법: 위 2단계 명령의 `이안류 클래스` / `사람 클래스` 출력이 의도와 같은지 보기

> `app.py` 의 `FALLBACK_RIP_IDS` / `FALLBACK_PERSON_IDS` 는 두 모델 구조에서는
> 실제 판정에 영향을 주지 않습니다. 고쳐도 결과가 안 바뀌니 시간 쓰지 마세요.

---

## 3단계 — 웹 화면 확인

```bash
streamlit run app.py
```

1. 사이드바 맨 위 **⚙️ 설정** → **개발자 모드** 선택 (모델 임계값을 만지려면 필수)
2. 사이드바 **테스트 영상**에 영상 업로드
3. **분석 시작** 클릭
4. 확인:
   - 큰 화면에 추론 결과(이안류 구역 + 사람 박스)가 프레임마다 갱신되는가
   - 우측 패널의 등급/인원 수가 영상과 맞는가
   - 하단 **발송된 경보 이미지** 갤러리에 섬네일이 쌓이는가

### 화면 모드 (사이드바 맨 위 ⚙️ 설정)

| 모드 | 보이는 것 | 용도 |
| --- | --- | --- |
| 사용자 모드 | 영상 업로드, 알림 발송 방식, 결과 화면 | 시연·발표 |
| 개발자 모드 | + 모델 설정(conf/iou/imgsz), 분석 설정, 판정 기준, 관제 표시 방식 | 모델 튜닝 |

### 모델 설정 — 두 모델의 임계값은 따로 잡습니다

사이드바 **모델 설정** 에서 이안류와 사람의 `conf` / `iou` / `imgsz` 를 각각 조절합니다.
한 값을 공유하면 한쪽은 반드시 손해를 봅니다.

- **이안류(detect)**: 구역이 사각형이라 넓게 잡히는 편. conf 를 너무 낮추면 바다 전체가 구역이 됩니다
- **사람(원거리 CCTV)**: 사람이 아주 작게 찍혀 기본값 0.25 로는 못 잡는 경우가 많습니다.
  `total=0` 이면 conf 를 0.10~0.15 까지 낮추고, `imgsz` 를 960~1280 으로 올려 보세요

> **값을 바꾸면 그 조합으로 모델을 새로 올립니다**(`@st.cache_resource` 키에 임계값이 들어감).
> 처음 한 번은 로딩이 걸립니다. 반대로 **모델 파일을 같은 경로에 덮어썼는데 결과가 그대로라면**
> 캐시가 경로 기준이라 그렇습니다. 앱을 재시작하거나 우측 상단 메뉴 → *Clear cache* 를 누르세요.

---

## 4단계 — 폰까지 (FCM 알림 + 이미지)

여기가 가장 함정이 많습니다. **순서를 지키세요.**

### 4-1. 준비

```bash
# 1) 에뮬레이터를 켜고 2~5분 기다린다 (아래 함정 ① 참고)

# 2) Streamlit 실행 (이미지 서빙용 — 끄면 폰에서 이미지가 깨진다)
streamlit run app.py

# 3) 포트 열기 ★ 빼먹기 제일 쉬움
adb reverse tcp:8501 tcp:8501

# 4) 앱 설치 (debug 빌드여야 함)
cd rip_current_app/rip_current_app/AndroidStudioProjects/RipCurrentAlert
./gradlew installDebug
```

> 터미널에서 `./gradlew` 가 `JAVA_HOME is not set` 으로 죽으면, Android Studio에 들어 있는
> JDK를 지정해 주세요 (Android Studio의 Run 버튼으로 설치하면 이 과정이 필요 없습니다).
> ```bash
> export JAVA_HOME="C:\Program Files\Android\Android Studio\jbr"
> ```

> **debug 빌드여야 하는 이유**: 평문 HTTP 허용이 `app/src/debug/AndroidManifest.xml` 에만
> 들어 있습니다. release 빌드로 설치하면 이미지가 안 뜹니다 (알림 자체는 옵니다).

### 4-2. 경로를 한 칸씩 확인

**① Streamlit이 이미지를 서빙하는가** (PC 브라우저에서)

```
http://localhost:8501/app/static/snapshots/<파일명>.jpg
```

이미지가 뜨지 않으면 앱 문제가 아니라 `.streamlit/config.toml` 문제입니다.

**② 에뮬레이터에서 그 주소에 닿는가**

```bash
adb shell run-as com.example.ripcurrentalert nc -w 5 127.0.0.1 8501 </dev/null
```

`nc: Timeout` 이 나오면 `adb reverse` 를 안 걸었거나 풀린 것입니다.
(에뮬레이터에 `curl` 은 없고 `nc` 만 있습니다.)

**③ 실제 발송**

```bash
python -m core.pipeline data/영상.mp4 --notifier fcm --skip 5
```

**④ 폰이 받았는지**

```bash
adb logcat -d -s FCM_CHECK
```

| 로그 | 의미 |
| --- | --- |
| `🚀 [수신 성공]` | FCM 도착 |
| `🔔 알림 배너 생성 완료` | 알림 생성됨 |
| `🖼️ 이미지 로딩 실패: ...` | 이미지만 실패 (여기에 원인이 찍힙니다) |

**⑤ 알림을 탭해서** 앱 화면에 CCTV 이미지가 뜨는지 눈으로 확인

---

## 5단계 — 실제 단말기에서 시연 (에뮬레이터 대신)

실기기(삼성 갤럭시)로 시연하되, **케이블 없이 무선으로** 쓰는 절차입니다.
`adb tcpip` 무선 연결은 로그 확인용일 뿐이고, FCM 배너 수신 자체와는 무관합니다
(배너는 구글 서버를 거쳐 오므로 이 연결이 끊겨도 옵니다). **같은 WiFi가 필요한 건
CCTV 이미지 쪽뿐**입니다 — 폰이 노트북 IP로 직접 HTTP 접속을 하기 때문입니다.

### 5-1. 개발자 모드 (삼성 기준)

```
설정 → 휴대전화 정보 → 소프트웨어 정보 → "빌드번호" 7회 연속 탭
  → PIN/패턴 입력 → "개발자 모드를 켰습니다" 토스트
```

삼성은 다른 제조사와 달리 **`소프트웨어 정보` 한 단계가 더** 들어갑니다.
`설정 → 개발자 옵션` 에서 **USB 디버깅**만 켜면 됩니다 (무선 디버깅 토글은 불필요 — 아래 참고).

### 5-2. USB로 최초 연결·설치

1. USB 케이블 연결 → 폰 알림에서 USB 모드를 **"파일 전송"** 으로 변경
   (기본값 "충전만" 이면 adb가 기기를 못 봄 — 가장 흔한 실수)
2. 폰에 뜨는 **"USB 디버깅을 허용하시겠습니까?"** → **"이 컴퓨터에서 항상 허용"** 체크 후 허용
3. `adb devices` 로 `device` 상태 확인 (`unauthorized` 면 아래 함정 ⑥)
4. Android Studio 상단 기기 드롭다운에서 실기기 선택 → **Run ▶** (또는 `./gradlew installDebug`)

### 5-3. 무선 adb로 전환 (시연 때 케이블을 뽑기 위해)

```bash
adb -s <기기ID> tcpip 5555          # 폰의 adb를 TCP 5555 로 전환
adb connect <폰의 WiFi IP>:5555     # 예: 192.168.0.232:5555
adb devices                          # <IP>:5555  device 확인
# 여기서 USB 케이블을 뽑아도 연결 유지됨
```

폰의 WiFi IP는 `adb -s <기기ID> shell ip -f inet addr show wlan0` 또는
`설정 → 연결 → Wi-Fi → 연결된 네트워크 톱니바퀴` 에서 확인합니다.

> **주의:** `adb tcpip` 은 폰을 재부팅하면 풀립니다. 그때는 USB를 다시 꽂고 위 과정을 반복하세요.
> 포트를 찾는 `adb pair` (페어링 코드) 방식은 쓰지 않습니다 — `tcpip` 이 훨씬 간단합니다.

### 5-4. 이미지 경로를 LAN IP로 — ★ 무선 시연의 핵심

에뮬레이터에서 쓰던 `adb reverse` 는 USB가 있어야 동작합니다. 무선에서는 폰이
노트북의 LAN IP로 직접 접속하게 해야 합니다. **코드 수정은 없고 환경변수만 바꿉니다.**

```bash
# 노트북 IP 확인 (DHCP라 바뀔 수 있으니 시연 직전에 매번 확인)
# PowerShell: Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.InterfaceAlias -eq "Wi-Fi" }

# Git Bash
RIP_IMAGE_BASE_URL=http://192.168.0.2:8501 streamlit run app.py
```

```cmd
:: cmd — 반드시 따옴표로 감쌀 것 (이유는 아래)
set "RIP_IMAGE_BASE_URL=http://192.168.0.2:8501"
streamlit run app.py
```

```powershell
# PowerShell
$env:RIP_IMAGE_BASE_URL = "http://192.168.0.2:8501"; streamlit run app.py
```

> **cmd 함정:** `set VAR=값 && 명령` 처럼 따옴표 없이 한 줄로 쓰면 **`&&` 앞의 공백까지
> 값에 포함됩니다.** URL이 `http://192.168.0.2:8501 /app/static/...` 이 되어
> **배너는 정상인데 이미지만 안 옵니다.** `set "VAR=값"` 형태로 쓰거나 두 줄로 나누세요.
> 의심되면 `data/events.db` 의 `image_url` 을 직접 확인하는 게 가장 빠릅니다.

> **`.env` 에 적어도 소용없습니다.** 이 프로젝트는 `load_dotenv()` 를 어디서도 호출하지
> 않아서 `.env` 파일 자체가 로드되지 않습니다. 환경변수는 반드시 실행하는 셸에서
> 직접 지정하세요.

기본값(`http://127.0.0.1:8501`)인 채로 두면 **폰이 자기 자신을 가리키게 되어
배너는 오는데 이미지만 깨집니다.**

폰 크롬에서 `http://<노트북IP>:8501` 이 뜨는지 앱을 켜기 전에 먼저 확인하세요.
안 뜨면:

| 원인 | 확인/조치 |
| --- | --- |
| 방화벽 | Windows Defender 방화벽에서 `python.exe` 인바운드가 **현재 WiFi 프로필(개인/공용)** 에 허용돼 있는지. 발표장 WiFi는 "공용"으로 잡히는 경우가 많음 |
| AP 격리 | 발표장 WiFi가 기기 간 통신을 차단(client isolation)하면 방법이 없음 → **폰 핫스팟에 노트북을 붙이는 방식**으로 우회 |
| IP 변경 | DHCP로 IP가 바뀜 → 위 명령으로 재확인 |

> **백업 플랜:** WiFi가 막히면 USB를 다시 꽂고
> `adb -s <기기ID> reverse tcp:8501 tcp:8501` + `RIP_IMAGE_BASE_URL` 없이(기본값
> `127.0.0.1`) 실행합니다. 발표 전 두 경로 다 한 번씩 리허설해 두세요.

### 5-5. 삼성 배터리 최적화 해제

삼성은 배터리 관리가 두 겹입니다. AOSP 표준 doze 예외는 adb로 걸 수 있지만,
**삼성 자체 "절전 앱" 목록은 폰 설정 UI에서만 풀립니다.**

```bash
adb -s <기기ID> shell dumpsys deviceidle whitelist +com.example.ripcurrentalert
```

추가로 폰에서 직접:

```
설정 → 배터리 → 백그라운드 사용 제한 → "절전 모드 앱" / "사용 안 함 앱" 목록에서
       RipCurrentAlert 가 있으면 제거
설정 → 애플리케이션 → RipCurrentAlert → 배터리 → "제한 없음(Unrestricted)" 선택
```

앱을 **홈 버튼으로 백그라운드에 내려서** 두 번째 경보로 배경 상태 수신까지
확인하세요 (최근 앱 스와이프로 지우면 안 되는 이유는 함정 ②와 동일).

### 5-6. 서명 — 신경 쓸 필요 없음

시연만 할 거면 별도 서명 설정이 필요 없습니다.

- Android Studio **Run ▶** / `./gradlew installDebug` 는 debug 키스토어로 **자동 서명**됩니다
- `app/build.gradle.kts` 에 `signingConfig` 가 없어서 **release 빌드는 미서명 → 설치 자체가 실패**합니다.
  반드시 debug 빌드로 설치하세요 (평문 HTTP 허용도 debug 전용이라 어차피 release는 이미지가 안 뜹니다)
- `app/google-services.json` 의 `oauth_client` 가 0개인 것으로 확인했습니다 — **FCM 메시징에
  SHA-1 지문 등록은 필요 없습니다.** (Google 로그인·Dynamic Links 같은 기능에만 필요)

---

## 증상별 원인 찾기

| 증상 | 먼저 볼 것 |
| --- | --- |
| `watch` 만 나오고 알림 0건 | `segment` 모델인지 (1단계 ②) |
| 사람/이안류가 뒤바뀜 | 클래스 매핑 (2단계) |
| 모델을 바꿨는데 결과가 그대로 | Streamlit 캐시 → 앱 재시작 |
| `FCM_CHECK` 로그가 **아무것도** 안 찍힘 | 에뮬레이터 GMS 초기화 대기 (함정 ①) |
| `🚀 수신 성공` 은 뜨는데 배너가 안 뜸 | 같은 알림이 이미 떠 있는지 (함정 ③) |
| 알림은 뜨는데 이미지 자리가 검음 | `adb reverse` + `🖼️ 이미지 로딩 실패` 로그 |
| 화면이 통째로 까맣게 멈춤 | 에뮬레이터 렌더러 (함정 ④) |
| 실기기에서 이미지만 검음 | `RIP_IMAGE_BASE_URL` 이 LAN IP인지. 기본값 `127.0.0.1` 이면 폰 자신을 가리킴 (5-4) |
| 폰 크롬에서 8501이 안 열림 | 방화벽 프로필(공용/개인) → AP 격리 → IP 변경 순으로 확인 (5-4) |
| 실기기에서 푸시가 늦게/안 옴 | 삼성 배터리 최적화 "제한 없음" 설정 (5-5) |
| `adb devices` 에 `unauthorized` | USB 디버깅 권한을 취소했다면 토글을 껐다 켜야 팝업이 다시 뜸 (함정 ⑥) |

---

## 알려진 함정

### ① 에뮬레이터를 켠 직후 2~5분은 FCM이 한 건도 안 온다

Google Play services가 자기 모듈을 다시 스캔·컴파일하는 동안 FCM 연결이 서지 않습니다.
이때는 `adb logcat -d -s FCM_CHECK` 에 **`🚀 [수신 성공]` 이 아예 안 찍힙니다.**
앱 문제로 오해하기 쉬우니, 로그가 통째로 비어 있으면 그냥 기다리세요.

확인:
```bash
adb logcat -d | grep -c "chimera module scan"   # 0이 아니면 아직 준비 중
```

### ② 앱을 최근 앱 목록에서 스와이프로 지우면 푸시가 안 온다

안드로이드가 "정지(stopped)" 상태 앱에는 브로드캐스트를 원천 차단합니다.
**홈 버튼으로 백그라운드에 내려두기만 하세요.**

### ③ 같은 알림이 떠 있으면 배너가 다시 안 뜬다

고정 ID(`NOTIFY_ID = 1001`)로 재발송하면 안드로이드가 "업데이트"로 처리해 배너도 소리도
생략합니다. 현재 코드는 `cancel()` 후 `notify()` 하도록 되어 있어 매번 다시 뜹니다
([MyFirebaseMessagingService.kt](rip_current_app/rip_current_app/AndroidStudioProjects/RipCurrentAlert/app/src/main/java/com/example/ripcurrentalert/MyFirebaseMessagingService.kt)).
이 부분을 되돌리지 마세요.

### ④ 화면이 까맣게 멈추면 전원 껐다 켜기

Windows 에뮬레이터의 렌더러가 프레임을 못 그리고 멈추는 문제입니다. 앱 잘못이 아닙니다.
전원 버튼 한 번 껐다 켜면 풀립니다. 자주 반복되면 AVD Graphics 설정을 `Software` 로 바꾸세요.

### ⑤ `10.0.2.2` 를 쓰지 마세요

에뮬레이터 표준 호스트 주소지만, **앱 프로세스에서는 10초 타임아웃**이 납니다
(`adb shell` 에서는 잘 되는데 앱 UID에서만 막힘 — 실측 확인).
`adb reverse` + `127.0.0.1` 을 쓰는 이유가 이것입니다.

### ⑥ `adb devices` 가 `unauthorized` 에서 안 풀린다

USB 디버깅 허용 팝업을 눌러도 상태가 안 바뀌는 경우, 대개 개발자 옵션에서
"USB 디버깅 권한 취소"를 한 번이라도 눌렀던 폰입니다. **권한 취소만으로는
재요청이 걸리지 않습니다.** 다음 순서로 강제로 다시 띄우세요.

1. 폰 잠금 해제 (팝업은 잠금 화면 위에 뜨지 않습니다)
2. 개발자 옵션 → **USB 디버깅 토글을 껐다가 다시 켜기** (재부팅 아님, 토글만)
3. 이때 뜨는 새 허용 팝업에서 "이 컴퓨터에서 항상 허용" 체크 후 허용

그래도 안 되면 `adb kill-server && adb start-server` 로 PC 쪽 데몬을 재시작해 보세요
(새 세션 키를 다시 보내면서 팝업이 재발생하는 경우가 있습니다).

---

## 문서와 코드가 다른 부분

| 문서의 설명 | 실제 |
| --- | --- |
| "사이드바 배지가 *데모 모드* → *모델 로드 완료* 로 바뀐다" | ✅ 있습니다. 사이드바 상단 `모델 · 이안류 … / 사람 …` |
| "`imgsz` 를 모델에서 읽어 쓴다" | ⚠️ 자동 적용은 아직입니다. `model_probe` 가 읽은 값은 쓰지 않고, **개발자 모드에서 직접 고른 값**을 씁니다(기본 640). 1280으로 학습했다면 사이드바에서 맞춰 주세요 |
| "클래스 매핑 실패 시 사이드바에서 직접 고른다" | ⚠️ 선택 위젯은 없습니다. 다만 모델이 둘로 나뉜 뒤로는 매핑 사고가 거의 없습니다(2단계 참고) |
| `.env.example` 이 ".env 로 복사해서 쓰세요"라고 안내 | ⚠️ **`.env` 는 실제로 로드되지 않습니다.** 프로젝트 어디에도 `load_dotenv()` 호출이 없습니다. 환경변수는 실행하는 셸에서 직접 지정하세요(`RIP_MODEL_PATH=... streamlit run app.py`). 다만 `RIP_MODEL_PATH` 는 `core/detector.py` 가 `models/rip/`, `models/person/` 을 직접 스캔하는 폴백이 있어 `.env` 없이도 모델 로딩 자체는 됩니다 |

넷 다 고치는 건 어렵지 않으니, 실제 모델을 붙인 뒤 필요해지면 이야기해 주세요.

---

## Streamlit Cloud 배포 시 달라지는 것

로컬 시연이 끝나고 배포로 넘어갈 때만 보세요.

1. `RIP_IMAGE_BASE_URL` 을 `https://<앱이름>.streamlit.app` 으로 설정
   → **`adb reverse` 도, 앱의 평문 HTTP 허용도 필요 없어집니다** (HTTPS라서)
2. `serviceAccountKey.json` 은 gitignore 되어 있어 리포에 없습니다.
   `st.secrets` 로 옮기고 `core/fcm_alert.py` 가 파일 대신 dict 로 인증서를 만들게 고쳐야 FCM이 동작합니다
3. **파일시스템이 휘발성입니다.** 앱이 재시작되면 `static/snapshots/` 와 `data/events.db` 가 사라져
   이전 푸시의 이미지 URL이 404가 되고 갤러리 기록도 초기화됩니다
4. `models/*.pt` 도 gitignore 대상 → Cloud에서는 가중치가 없어 `FakeDetector` 로 동작합니다
5. Community Cloud는 메모리 약 1GB, GPU 없음 → 영상 추론이 무거우면 OOM 가능성을 봐야 합니다
