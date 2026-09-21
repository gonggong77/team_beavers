# 실기기 무선 시연 — 복붙 실행 절차

IP는 이 프로젝트 환경 기준(노트북 `192.168.0.2`, 폰 `192.168.0.232`)으로 이미 채워져 있습니다.
IP가 바뀌었으면 [REAL_TEST_GUIDE.md](REAL_TEST_GUIDE.md) 5-4를 참고해 값만 바꾸세요.

## 1) Streamlit 재시작 (이미지 서빙 + IP 지정)

```cmd
set RIP_IMAGE_BASE_URL=http://192.168.0.2:8501
streamlit run app.py
```

이 터미널은 계속 켜둔 채로 둔다.

## 2) 새 터미널 — 무선 adb 연결

```powershell
adb connect 192.168.0.232:5555
adb devices
```

`192.168.0.232:5555   device` 로 나오면 정상.

## 3) 분석 실행

```powershell
python -m core.pipeline data/영상파일명.mp4 --notifier fcm --skip 5
```

`영상파일명.mp4` 부분만 실제 파일명으로 교체.

## 4) 폰 수신 확인

```powershell
adb -s 192.168.0.232:5555 logcat -d -s FCM_CHECK
```

- `🚀 [수신 성공]` 의 `image_url` 이 `http://192.168.0.2:8501/...` 로 찍히면 정상
- 여전히 `127.0.0.1` 로 찍히면 1번이 적용 안 된 것 → Streamlit 재시작부터 다시

## 안 될 때

| 증상 | 원인 |
| --- | --- |
| 이미지 자리만 검음 | `RIP_IMAGE_BASE_URL` 미설정 (기본값 `127.0.0.1`이 폰 자신을 가리킴) |
| `adb connect` 실패 | USB로 연결 후 `adb tcpip 5555` 먼저 실행 (자세한 건 REAL_TEST_GUIDE.md 5-3) |
| 폰에서 8501 자체가 안 열림 | 같은 WiFi인지, 방화벽/AP 격리 확인 (REAL_TEST_GUIDE.md 5-4 표) |
