"""이안류 경보 알림 발송 콘솔 (Streamlit).

    streamlit run app.py

원본 앱에서 알림 발송에 관련된 코드만 모은 폴더다. 모델 추론도, 관제 화면도 없다.
경보 내용을 손으로 채워 실제로 발송해 보고, 결과(성공/실패, 이미지 URL, 메시지 ID)를
바로 확인하는 용도다.

  - 발송 방식 console : 터미널에 문안만 찍는다. 앱이 없어도 흐름 확인 가능
  - 발송 방식 fcm     : 안드로이드 앱으로 실제 발송. serviceAccountKey.json 필요

이미지는 어디에도 업로드하지 않는다. static/snapshots/ 에 저장한 뒤
이 Streamlit 서버가 서빙하는 URL(/app/static/...)을 앱에 넘긴다.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from core.fcm_alert import (
    IMAGE_BASE_URL,
    SERVICE_ACCOUNT_PATH,
    SNAPSHOT_DIR,
    TOPIC,
)
from core.notifier import ConsoleNotifier, FcmNotifier
from core.schemas import RISK_LABEL, AlertEvent
from core.sender import dispatch, save_snapshot
from core.store import EventStore

st.set_page_config(
    page_title="이안류 경보 발송 콘솔",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

KIND_LABEL = {"console": "화면 기록만 (console)", "fcm": "안드로이드 앱 (FCM)"}
TRIGGER_LABEL = {"auto": "자동", "escalation": "인원 증가"}


@st.cache_resource
def get_store() -> EventStore:
    return EventStore()


def main() -> None:
    st.title("📡 이안류 경보 발송 콘솔")
    st.caption("경보 내용을 채워 실제로 발송해 보는 화면입니다. 탐지·관제 화면은 들어 있지 않습니다.")

    notifier = sidebar()
    send_form(notifier)
    history()


def sidebar() -> ConsoleNotifier | FcmNotifier:
    st.sidebar.markdown(
        "<div style='font-size:1.02rem;font-weight:700;line-height:1.35'>"
        "📡 이안류 경보 알림 발송</div>"
        "<div style='font-size:.78rem;opacity:.6;margin-bottom:10px'>팀 비버즈 · 알림 파트</div>",
        unsafe_allow_html=True,
    )

    kind = st.sidebar.radio(
        "발송 방식", list(KIND_LABEL), format_func=KIND_LABEL.get,
        help="console 은 터미널에 문안만 찍습니다. fcm 은 구독 중인 안드로이드 앱으로 실제 발송합니다.",
    )

    if kind == "console":
        st.sidebar.caption("실제로는 아무 데도 보내지 않습니다.")
        return ConsoleNotifier()

    # 좌표와 카메라 ID는 탐지 결과가 아니라 배포 설정이라 AlertEvent 에 없다.
    # 여기서 넣은 값이 그대로 FCM data 필드로 나가 앱 화면에 표시된다.
    with st.sidebar.expander("발송 설정 (FCM)", expanded=True):
        latitude = st.number_input("위도", value=35.1587, format="%.4f")
        longitude = st.number_input("경도", value=129.1604, format="%.4f")
        camera_id = st.text_input("카메라 ID", value="CCTV-E01")

    notifier = FcmNotifier(latitude=latitude, longitude=longitude, camera_id=camera_id)

    if notifier.configured:
        st.sidebar.success(f"서비스 계정 키 확인됨\n\n`{SERVICE_ACCOUNT_PATH.name}`")
    else:
        st.sidebar.error(
            f"서비스 계정 키가 없습니다.\n\n`{SERVICE_ACCOUNT_PATH}` 에 두거나 "
            "환경변수 `FIREBASE_CREDENTIALS` 로 경로를 지정하세요."
        )

    st.sidebar.caption(f"토픽 `{TOPIC}` · 이미지 주소 `{IMAGE_BASE_URL}`")
    # 발송이 '성공' 이어도 구독자가 없으면 아무 기기에도 도착하지 않는다.
    # 시연 중 가장 많이 헷갈리는 지점이라 화면에 상시로 띄운다.
    st.sidebar.info(
        "발송 성공은 FCM 서버 접수를 뜻할 뿐 도착을 보장하지 않습니다.\n\n"
        "실제 수신은 `adb logcat -d -s FCM_CHECK` 에서 확인하세요.\n\n"
        "로컬 시연이면 이미지가 뜨도록 `adb reverse tcp:8501 tcp:8501` 을 먼저 실행하세요."
    )
    return notifier


def send_form(notifier) -> None:
    st.subheader("경보 발송")

    with st.form("send"):
        left, right = st.columns(2)

        with left:
            risk = st.selectbox(
                "위험 등급", ["emergency", "warn", "watch"],
                format_func=lambda r: f"{RISK_LABEL[r]} ({r})",
            )
            persons_in_rip = int(st.number_input("구역 내 인원", 0, 999, 2))
            total_persons = int(st.number_input("화면 전체 인원", 0, 999, 5))
            rip_count = int(st.number_input("의심 구역 수", 0, 99, 1))

        with right:
            video_source = st.text_input(
                "영상 이름", value="sample_beach.mp4",
                help="앱 화면과 알림 배너에 지점 이름으로 그대로 표시됩니다.",
            )
            trigger_kind = st.selectbox(
                "발송 사유", ["auto", "escalation"], format_func=TRIGGER_LABEL.get,
                help="자동 = 연속 긴급 판정, 인원 증가 = 직전 발송보다 구역 내 인원이 늘어남",
            )
            frame_idx = int(st.number_input("프레임 번호", 0, 10_000_000, 412))
            timestamp_sec = float(st.number_input("영상 위치(초)", 0.0, 100_000.0, 13.7, step=0.1))

        snapshot = st.file_uploader(
            "경보 이미지 (선택)", type=["jpg", "jpeg", "png"],
            help="올리면 static/snapshots/ 에 저장하고 그 URL을 앱에 함께 보냅니다. "
                 "비워 두면 앱은 이미지 없이 경보만 띄웁니다.",
        )

        submitted = st.form_submit_button("경보 발송", type="primary", width="stretch")

    if not submitted:
        return

    event = AlertEvent(
        risk_level=risk,
        persons_in_rip=persons_in_rip,
        total_persons=total_persons,
        rip_count=rip_count,
        frame_idx=frame_idx,
        timestamp_sec=timestamp_sec,
        video_source=video_source,
        trigger_kind=trigger_kind,
    )

    if snapshot is not None:
        path = save_snapshot(snapshot.getvalue(), event)
        st.caption(f"이미지 저장: `{path}`")

    result = dispatch(event, notifier, get_store())

    if result.ok:
        st.success(f"발송 완료 · {result.detail}")
    else:
        st.error(f"발송 실패 · {result.detail}")

    detail, preview = st.columns([2, 1])
    with detail:
        st.markdown("**보낸 내용**")
        st.code(event.summary_text(), language=None)
        if result.image_url:
            st.markdown("**앱에 전달된 이미지 주소**")
            st.code(result.image_url, language=None)
        elif snapshot is not None:
            st.caption("이미지 URL이 만들어지지 않았습니다 (위 실패 사유 참고).")
    with preview:
        if snapshot is not None:
            st.image(snapshot, caption="발송된 이미지", width="stretch")


def history() -> None:
    st.divider()
    st.subheader("발송 기록")

    store = get_store()
    events = store.list_events()
    if not events:
        st.caption("아직 기록된 이벤트가 없습니다.")
        return

    frame = pd.DataFrame(events)[
        ["occurred_at", "risk_level", "persons_in_rip", "total_persons", "rip_count",
         "video_source", "trigger_kind", "notify_status", "notify_detail", "image_url"]
    ].rename(columns={
        "occurred_at": "발생 시각", "risk_level": "등급", "persons_in_rip": "구역 내 인원",
        "total_persons": "전체 인원", "rip_count": "구역 수", "video_source": "영상",
        "trigger_kind": "발송 사유", "notify_status": "발송 상태", "notify_detail": "상세",
        "image_url": "이미지 URL",
    })
    frame["발송 사유"] = frame["발송 사유"].map(TRIGGER_LABEL).fillna(frame["발송 사유"])
    frame["등급"] = frame["등급"].map(RISK_LABEL).fillna(frame["등급"])
    st.dataframe(frame, width="stretch", hide_index=True)

    left, right = st.columns(2)
    left.download_button(
        "이벤트 내려받기 (JSONL)", store.to_jsonl(),
        file_name="alert_events.jsonl", mime="application/x-ndjson", width="stretch",
    )
    # 되돌릴 수 없으므로 한 번 더 확인받는다.
    if right.button("기록 비우기", width="stretch"):
        store.clear()
        st.rerun()

    st.caption(f"저장 위치 · DB `{store.db_path}` · 이미지 `{SNAPSHOT_DIR}`")


if __name__ == "__main__":
    main()
