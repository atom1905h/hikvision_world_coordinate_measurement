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
IP = os.getenv("CCTV_3_IP")

RTSP_URL = (
    f"rtsp://{USERNAME}:{PASSWORD}@{IP}:554/Streaming/Channels/101"
)


# ===========================
# 기준 마커 설정 (좌표계 원점 역할)
# ===========================

REFERENCE_DICT = "DICT_6X6_250"
REFERENCE_ID = 2

# 마커 한 변의 실제 길이 (cm)
MARKER_SIZE_CM = 18.0

# ArUco 코너 순서: pts[0]=TL, pts[1]=TR, pts[2]=BR, pts[3]=BL
# 사용자가 정의한 실제 좌표(cm):
#   P2(BR) = (0, 0)
#   P1(TR) = (0, 18)
#   P0(TL) = (-18, 18)
#   P3(BL) = (-18, 0)
# -> pts 순서(TL, TR, BR, BL)에 맞춰 동일한 순서로 배열
REFERENCE_WORLD_CORNERS = np.array(
    [
        [-MARKER_SIZE_CM, MARKER_SIZE_CM],  # TL -> P0
        [0.0, MARKER_SIZE_CM],              # TR -> P1
        [0.0, 0.0],                         # BR -> P2
        [-MARKER_SIZE_CM, 0.0],             # BL -> P3
    ],
    dtype=np.float32
)


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

def compute_homography(image_corners, world_corners):
    """
    기준 마커의 이미지(픽셀) 코너 4점과 실제 좌표 4점으로부터
    호모그래피 행렬 H를 계산한다.

    image_corners : (4, 2) ndarray, pixel 좌표 (TL, TR, BR, BL 순서)
    world_corners  : (4, 2) ndarray, 실제 좌표 (cm), 같은 순서

    return: 3x3 호모그래피 행렬 H (실패 시 None)
    """

    H, status = cv2.findHomography(
        image_corners.astype(np.float32),
        world_corners.astype(np.float32)
    )

    return H


def pixel_to_world(H, pixel_point):
    """
    호모그래피 H를 이용해 임의의 픽셀 좌표를 실제 좌표(cm)로 변환한다.
    단, 변환하려는 점은 기준 마커(ID=2)와 동일한 평면 위에 있어야 한다.

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

window_name = "Hikvision ArUco World Coordinate"

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
# 마커별 고정(lock) 결과 저장
# ===========================

# {
#     (dict_name, marker_id): {
#         "pixel_corners": (4,2) ndarray,   # 최초 탐지 시 픽셀 코너 (TL,TR,BR,BL)
#         "world_corners": (4,2) ndarray,   # 위 코너를 변환한 실제 좌표 (cm)
#         "frame_count": int                # 최초 탐지된 프레임 번호
#     }
# }
locked_markers = {}


# ===========================
# 메인 루프
# ===========================

frame_count = 0

# 마지막으로 계산된 호모그래피 (기준 마커가 잠깐 안 보여도 유지)
last_homography = None
homography_status = "NOT CALIBRATED"

while True:

    ret, frame = cap.read()

    if not ret:
        print("프레임 수신 실패")
        break

    frame_count += 1

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
    # 2. 기준 마커(ID=2)로 호모그래피 계산/갱신
    #    (기준 마커가 보일 때마다 최신 상태로 갱신)
    # =====================================================

    for dict_name, marker_id, pts in detected_markers:

        if dict_name == REFERENCE_DICT and marker_id == REFERENCE_ID:

            H = compute_homography(pts, REFERENCE_WORLD_CORNERS)

            if H is not None:
                last_homography = H
                homography_status = "CALIBRATED (LIVE)"

            break

    else:
        # 이번 프레임에 기준 마커가 안 보이면, 이전 호모그래피 유지
        if last_homography is not None:
            homography_status = "CALIBRATED (LAST KNOWN)"


    # =====================================================
    # 3. 최초 탐지된 마커를 고정(lock)
    #    - 호모그래피가 준비된 상태에서만 고정
    #      (실제 좌표를 못 구하면 아직 고정하지 않고 다음 프레임에 재시도)
    # =====================================================

    for dict_name, marker_id, pts in detected_markers:

        key = (dict_name, marker_id)

        if key in locked_markers:
            continue

        if last_homography is None:
            continue

        world_corners = np.array(
            [pixel_to_world(last_homography, (x, y)) for x, y in pts],
            dtype=np.float32
        )

        locked_markers[key] = {
            "pixel_corners": pts.copy(),
            "world_corners": world_corners,
            "frame_count": frame_count
        }


    # =====================================================
    # 4. 고정된 모든 마커 시각화 (계속 유지, 4개 꼭짓점 각각 표시)
    # =====================================================

    for (dict_name, marker_id), data in locked_markers.items():

        pts = data["pixel_corners"]
        world_corners = data["world_corners"]

        detected_frame = data["frame_count"]
        elapsed_frames = frame_count - detected_frame

        pts_int = pts.astype(int)

        center = pts.mean(axis=0)
        center_int = center.astype(int)

        is_reference = (
            dict_name == REFERENCE_DICT and marker_id == REFERENCE_ID
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
        # 4개 꼭짓점 (픽셀 좌표 + 실제 좌표 각각 표시)
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
        # 중심점
        # -----------------------------------------------

        cv2.circle(
            frame,
            tuple(center_int),
            7,
            (255, 0, 0),
            -1
        )


        # -----------------------------------------------
        # ID / 상태 표시
        # -----------------------------------------------

        cv2.putText(
            frame,
            f"{dict_name} ID:{marker_id}  LOCKED (+{elapsed_frames}f)",
            (center_int[0] + 10, center_int[1] + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            line_color,
            2
        )


    # =====================================================
    # 상태 텍스트
    # =====================================================

    cv2.putText(
        frame,
        f"Reference: {REFERENCE_DICT} ID:{REFERENCE_ID}  |  {homography_status}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Locked markers: {len(locked_markers)}",
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