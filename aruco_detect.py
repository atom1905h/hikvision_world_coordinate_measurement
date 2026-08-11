import cv2
import numpy as np
import ctypes
import os
from dotenv import load_dotenv

# ===========================
# Hikvision RTSP 정보
# ===========================

load_dotenv()  

USERNAME = os.getenv("CCTV_USERNAME")
PASSWORD = os.getenv("CCTV_PASSWORD")
IP = os.getenv("CCTV_3_IP")

RTSP_URL = (
    f"rtsp://{USERNAME}:{PASSWORD}@{IP}:554/Streaming/Channels/101"
)


# ===========================
# 카메라 열기
# ===========================

cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print("RTSP 연결 실패")
    exit()

cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)


# ===========================
# ArUco 설정
# ===========================

ARUCO_DICTS = {
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
}

parameters = cv2.aruco.DetectorParameters()

detectors = {}

for dict_name, dict_id in ARUCO_DICTS.items():

    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)

    detectors[dict_name] = cv2.aruco.ArucoDetector(
        aruco_dict,
        parameters
    )


# ===========================
# 마지막 탐지 결과 저장
# ===========================

# {
#     (dict_name, marker_id): {
#         "corners": np.ndarray,
#         "center": np.ndarray,
#         "frame_count": int
#     }
# }
last_detected = {}


# NOTE: 한번 탐지된 마커는 다시 탐지되지 않아도 화면에서 사라지지 않도록
# 만료(삭제) 로직을 제거했습니다. (더 이상 MAX_KEEP_FRAMES를 사용하지 않습니다)


# ===========================
# Window
# ===========================

window_name = "Hikvision ArUco Detection"

user32 = ctypes.windll.user32

screen_width = user32.GetSystemMetrics(0)
screen_height = user32.GetSystemMetrics(1)

cv2.namedWindow(
    window_name,
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    window_name,
    screen_width,
    screen_height
)


# ===========================
# 메인 루프
# ===========================

frame_count = 0

while True:

    ret, frame = cap.read()

    if not ret:
        print("프레임 수신 실패")
        break

    frame_count += 1

    height, width = frame.shape[:2]

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


    # =====================================================
    # 1. 현재 프레임에서 ArUco 탐지
    # =====================================================

    current_detected = set()

    for dict_name, detector in detectors.items():

        corners, ids, rejected = detector.detectMarkers(gray)

        if ids is None:
            continue

        ids = np.array(ids).flatten()

        for marker_corner, marker_id in zip(corners, ids):

            marker_id = int(marker_id)

            pts = marker_corner.reshape((4, 2))

            center = pts.mean(axis=0)


            # 현재 프레임에서 탐지됨
            key = (dict_name, marker_id)

            current_detected.add(key)


            # =================================================
            # 최초 탐지 결과만 기록 (이후 재탐지되어도 갱신하지 않음)
            # =================================================

            if key not in last_detected:

                last_detected[key] = {
                    "corners": pts.copy(),
                    "center": center.copy(),
                    "frame_count": frame_count
                }


    # =====================================================
    # 2. 마지막으로 탐지된 ArUco 시각화
    #    (한번 탐지되면 계속 유지되며 삭제되지 않음)
    # =====================================================

    for key, data in last_detected.items():

        dict_name, marker_id = key

        pts = data["corners"]
        center = data["center"]

        detected_frame = data["frame_count"]

        elapsed_frames = frame_count - detected_frame


        # =================================================
        # 최초 탐지 위치에 고정 (재탐지 여부와 무관하게 동일하게 표시)
        # =================================================

        line_color = (0, 165, 255)

        status = f"LOCKED (+{elapsed_frames}f)"


        # =================================================
        # ArUco 외곽선
        # =================================================

        pts_int = pts.astype(int)

        cv2.polylines(
            frame,
            [pts_int],
            True,
            line_color,
            3
        )


        # =================================================
        # 4개 코너
        # =================================================

        for i, (x, y) in enumerate(pts):

            x = int(round(x))
            y = int(round(y))

            cv2.circle(
                frame,
                (x, y),
                6,
                (0, 0, 255),
                -1
            )

            text = f"P{i} ({x}, {y})"

            cv2.putText(
                frame,
                text,
                (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2
            )


        # =================================================
        # 중심점
        # =================================================

        center_int = center.astype(int)

        cv2.circle(
            frame,
            tuple(center_int),
            7,
            (255, 0, 0),
            -1
        )


        # =================================================
        # Marker ID
        # =================================================

        cv2.putText(
            frame,
            f"{dict_name} ID:{marker_id}",
            (
                center_int[0] + 10,
                center_int[1] + 25
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            line_color,
            2
        )


        # =================================================
        # 상태
        # =================================================

        cv2.putText(
            frame,
            status,
            (
                center_int[0] + 10,
                center_int[1] + 50
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            line_color,
            2
        )


    # =====================================================
    # 현재 CCTV 해상도
    # =====================================================

    cv2.putText(
        frame,
        f"Resolution: {width} x {height}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # =====================================================
    # 탐지된 Marker 개수
    # =====================================================

    cv2.putText(
        frame,
        f"Markers: {len(last_detected)}",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # =====================================================
    # 화면 출력
    # =====================================================

    cv2.imshow(
        window_name,
        frame
    )


    # ESC
    key = cv2.waitKey(1)

    if key == 27:
        break


# ===========================
# 종료
# ===========================

cap.release()
cv2.destroyAllWindows()