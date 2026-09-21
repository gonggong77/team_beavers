

import os
import cv2
import time
from ultralytics import YOLO

# ==========================================
# 1. 모델 가중치 파일 경로 설정
# ==========================================
# 1) 파인튜닝된 소형 사람/서퍼 탐지 모델 (YOLO11m)
person_model_path = "runs/detect/rip_person_opt/weights/best.pt"
if not os.path.exists(person_model_path):
    person_model_path = "best.pt"

# 2) 기존 이안류 탐지 모델
rip_model_path = "best_v8m_final.pt"

print("🔄 모델 로딩 중...")
model_rip = YOLO(rip_model_path)
model_person = YOLO(person_model_path)
print("✅ 두 모델 모두 성공적으로 로드되었습니다.")

# ==========================================
# 2. 영상 입력 및 출력 설정
# ==========================================
input_video = "testdata/ian1.mp4"  # 테스트할 비디오 경로
output_video = "rip_and_person_result.mp4"

cap = cv2.VideoCapture(input_video)
if not cap.isOpened():
    print(f"❌ 영상을 열 수 없습니다: {input_video}")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fps = 30.0 if (fps == 0 or fps is None) else fps

# [시간 계산용] 전체 프레임 수 및 전체 영상 길이(초)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
total_duration_sec = total_frames / fps if fps > 0 else 0.0

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# 윈도우 창 크기 설정
win_name = "Rip-Current & Person Dual Monitoring"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(win_name, 1280, 720)

prev_time = time.time()
frame_idx = 0

print("🚀 듀얼 추론 시작! ('q' 키를 누르면 종료됩니다)")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    annotated_frame = frame.copy()

    # --------------------------------------------------
    # [시간 계산] 현재 재생 위치 (초 단위 및 mm:ss 포맷)
    # --------------------------------------------------
    # POS_MSEC로 현재 프레임의 타임스탬프(밀리초) 확인 (실패 시 프레임/FPS 기준)
    curr_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
    current_sec = curr_msec / 1000.0 if curr_msec > 0 else (frame_idx / fps)
    
    # mm:ss 형태 변환
    curr_min, curr_s = divmod(int(current_sec), 60)
    total_min, total_s = divmod(int(total_duration_sec), 60)
    time_str = f"Time: {curr_min:02d}:{curr_s:02d} / {total_min:02d}:{total_s:02d} ({current_sec:.1f}s)"

    # ==========================================
    # 3. [모델 1] 이안류(Rip Current) 탐지
    # ==========================================
    results_rip = model_rip.predict(
        source=frame,
        imgsz=640,
        conf=0.25,
        iou=0.7,
        device="cuda:0",
        verbose=False
    )

    rip_boxes = []
    for r in results_rip:
        for box in r.boxes:
            rx1, ry1, rx2, ry2 = map(int, box.xyxy[0])
            r_conf = float(box.conf[0])
            rip_boxes.append((rx1, ry1, rx2, ry2))

            # 이안류 영역 표시 (빨간색 반투명 오버레이)
            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (rx1, ry1), (rx2, ry2), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.25, annotated_frame, 0.75, 0, annotated_frame)
            cv2.rectangle(annotated_frame, (rx1, ry1), (rx2, ry2), (0, 0, 255), 2)
            cv2.putText(
                annotated_frame,
                f"RIP CURRENT ({r_conf:.2f})",
                (rx1, max(25, ry1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

    # ==========================================
    # 4. [모델 2] 소형 입수자(Person) 트래킹
    # ==========================================
    results_person = model_person.track(
        source=frame,
        persist=True,
        imgsz=1024,
        conf=0.10,
        augment=True,
        iou=0.5,
        tracker="bytetrack.yaml",
        device="cuda:0",
        verbose=False,
    )

    danger_count = 0
    safe_count = 0

    if results_person[0].boxes is not None:
        for box in results_person[0].boxes:
            px1, py1, px2, py2 = map(int, box.xyxy[0])
            p_conf = float(box.conf[0])
            track_id = int(box.id[0]) if box.id is not None else -1

            # 입수자 하단 접점(발끝/수면 접점)
            foot_x = (px1 + px2) // 2
            foot_y = py2

            # 5. 이안류 구역 내부 침범 검사
            is_in_danger = False
            for (rx1, ry1, rx2, ry2) in rip_boxes:
                if rx1 <= foot_x <= rx2 and ry1 <= foot_y <= ry2:
                    is_in_danger = True
                    break

            if is_in_danger:
                danger_count += 1
                box_color = (0, 140, 255)  # 위험: 주황색
                label = f"! DANGER #{track_id}" if track_id != -1 else "! DANGER"
            else:
                safe_count += 1
                box_color = (0, 255, 0)    # 안전: 초록색
                label = f"Person #{track_id}" if track_id != -1 else f"Person ({p_conf:.2f})"

            cv2.rectangle(annotated_frame, (px1, py1), (px2, py2), box_color, 2)
            cv2.circle(annotated_frame, (foot_x, foot_y), 4, box_color, -1)
            cv2.putText(
                annotated_frame,
                label,
                (px1, max(15, py1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                box_color,
                2
            )

    # ==========================================
    # 6. 관제 상태 대시보드 오버레이 (시간 정보 포함)
    # ==========================================
    curr_time = time.time()
    fps_current = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
    prev_time = curr_time

    # 상단 정보 패널 (시간 표시를 위해 박스 크기 확장)
    cv2.rectangle(annotated_frame, (15, 15), (460, 150), (0, 0, 0), -1)
    
    # 1) 재생 시간 (초 단위 및 분:초)
    cv2.putText(annotated_frame, time_str, (25, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    # 2) FPS 및 처리 프레임 수
    cv2.putText(annotated_frame, f"FPS: {fps_current:.1f} | Frame: {frame_idx}/{total_frames}", (25, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    # 3) 이안류 구역 감지 현황
    cv2.putText(annotated_frame, f"Rip Current Zone: {len(rip_boxes)} detected", (25, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255) if len(rip_boxes) > 0 else (180, 180, 180), 2)
    # 4) 총 탐지 인원 수
    cv2.putText(annotated_frame, f"Total Swimmers: {safe_count + danger_count}", (25, 115),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    # 5) 위험 진입 인원 수
    cv2.putText(annotated_frame, f"In Danger (Rip): {danger_count}", (25, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 140, 255) if danger_count > 0 else (150, 150, 150), 2)

    # 화면 표시 및 저장
    cv2.imshow(win_name, annotated_frame)
    out.write(annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\n사용자에 의해 종료되었습니다.")
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"\n🎉 완료! 결과 비디오가 저장되었습니다: {os.path.abspath(output_video)}")