import cv2
import numpy as np
import ctypes
import os
from datetime import datetime
from dotenv import load_dotenv

# ===========================
# Hikvision RTSP 정보
# ===========================

load_dotenv()

USERNAME = os.getenv("CCTV_USERNAME")
PASSWORD = os.getenv("CCTV_PASSWORD")
IP = os.getenv("CCTV_1_IP")

RTSP_URL = (
    f"rtsp://{USERNAME}:{PASSWORD}@{IP}:554/Streaming/Channels/101"
)


# ===========================
# 호모그래피 불러오기
# ===========================

HOMOGRAPHY_FILE = "calibration/homography.npz"

if not os.path.exists(HOMOGRAPHY_FILE):
    print(f"[오류] 호모그래피 파일을 찾을 수 없습니다: {HOMOGRAPHY_FILE}")
    print("       먼저 homography_calibration_4markers.py를 실행해서 저장해주세요.")
    exit()

_data = np.load(HOMOGRAPHY_FILE, allow_pickle=True)

H = _data["H"].astype(np.float32)
REFERENCE_DICT = str(_data["reference_dict"])
REFERENCE_IDS = [int(x) for x in _data["reference_ids"]]
WORLD_COORDS_BY_ID = _data["world_coords_by_id"].item()

print(f"호모그래피 로드 완료: {HOMOGRAPHY_FILE}")
print(f"  기준 마커: {REFERENCE_DICT} ID:{REFERENCE_IDS}")
print(f"  기준 좌표: {WORLD_COORDS_BY_ID}")
print("H =\n", H)


# ===========================
# 화면 저장 설정
# ===========================

SAVE_DIR = "captures"

os.makedirs(SAVE_DIR, exist_ok=True)


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
# 좌표 변환 함수
# ===========================

def pixel_to_world(H, pixel_point):
    """
    호모그래피 H를 이용해 임의의 픽셀 좌표를 실제 좌표(cm)로 변환한다.

    H            : 3x3 호모그래피 행렬
    pixel_point  : (x, y) 픽셀 좌표

    return: (world_x, world_y) 실제 좌표 (cm)
    """

    pt = np.array(
        [[[pixel_point[0], pixel_point[1]]]],
        dtype=np.float32
    )

    world_pt = cv2.perspectiveTransform(pt, H)

    world_x = float(world_pt[0][0][0])
    world_y = float(world_pt[0][0][1])

    return world_x, world_y


# ===========================
# Window
# ===========================

window_name = "Hikvision ArUco World Coordinate (Real-time)"

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

while True:

    ret, frame = cap.read()

    if not ret:
        print("프레임 수신 실패")
        break

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


    # =====================================================
    # 1. 현재 프레임에서 ArUco 탐지
    # =====================================================

    detected_markers = []  # [(dict_name, marker_id, pts(4,2)), ...]

    for dict_name, detector in detectors.items():

        corners, ids, rejected = detector.detectMarkers(gray)

        if ids is None:
            continue

        ids = np.array(ids).flatten()

        for marker_corner, marker_id in zip(corners, ids):

            marker_id = int(marker_id)

            pts = marker_corner.reshape((4, 2))

            detected_markers.append((dict_name, marker_id, pts))


    # =====================================================
    # 2. 탐지된 모든 마커를 실시간으로 실제 좌표 변환 & 시각화
    #    (고정/LOCK 없이 매 프레임 갱신)
    # =====================================================

    for dict_name, marker_id, pts in detected_markers:

        world_corners = np.array(
            [pixel_to_world(H, (x, y)) for x, y in pts],
            dtype=np.float32
        )

        pts_int = pts.astype(int)

        center = pts.mean(axis=0)
        center_int = center.astype(int)

        world_center_x, world_center_y = pixel_to_world(
            H, (center[0], center[1])
        )

        is_reference = (
            dict_name == REFERENCE_DICT and marker_id in REFERENCE_IDS
        )

        line_color = (0, 255, 255) if is_reference else (0, 165, 255)


        # -----------------------------------------------
        # 외곽선
        # -----------------------------------------------

        cv2.polylines(
            frame,
            [pts_int],
            True,
            line_color,
            3
        )


        # -----------------------------------------------
        # 4개 꼭짓점 (실제 좌표 각각 표시, 소수점 첫째자리)
        # -----------------------------------------------

        corner_labels = ["P0(TL)", "P1(TR)", "P2(BR)", "P3(BL)"]

        for i in range(4):

            x, y = pts_int[i]
            wx, wy = world_corners[i]

            cv2.circle(
                frame,
                (int(x), int(y)),
                6,
                (0, 0, 255),
                -1
            )

            corner_text = f"{corner_labels[i]} ({wx:.1f}, {wy:.1f})cm"

            cv2.putText(
                frame,
                corner_text,
                (int(x) + 8, int(y) - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2
            )


        # -----------------------------------------------
        # 중심점 (실제 좌표 표시)
        # -----------------------------------------------

        cv2.circle(
            frame,
            tuple(center_int),
            7,
            (255, 0, 0),
            -1
        )

        cv2.putText(
            frame,
            f"({world_center_x:.1f}, {world_center_y:.1f})cm",
            (center_int[0] + 10, center_int[1] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 0, 0),
            2
        )


        # -----------------------------------------------
        # ID 표시
        # -----------------------------------------------

        # cv2.putText(
        #     frame,
        #     f"{dict_name} ID:{marker_id}",
        #     (center_int[0] + 10, center_int[1] + 25),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.6,
        #     line_color,
        #     2
        # )


    # =====================================================
    # 상태 텍스트
    # =====================================================

    cv2.putText(
        frame,
        f"Homography loaded from file (Reference: {REFERENCE_DICT} ID:{REFERENCE_IDS})",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Detected markers: {len(detected_markers)}",
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


    # 키 입력 처리
    key = cv2.waitKey(1) & 0xFF

    # ESC: 종료
    if key == 27:
        break

    # S: 현재 화면 저장
    if key == ord('s') or key == ord('S'):

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

        save_path = os.path.join(SAVE_DIR, f"capture_{timestamp}.png")

        cv2.imwrite(save_path, frame)

        print(f"화면 저장됨: {save_path}")


# ===========================
# 종료
# ===========================

cap.release()
cv2.destroyAllWindows()