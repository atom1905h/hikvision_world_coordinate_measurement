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
# 저장 경로
# ===========================

SAVE_DIR = "calibration"
HOMOGRAPHY_FILE = os.path.join(SAVE_DIR, "homography.npz")

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

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)


# ===========================
# 호모그래피 계산 함수
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


# ===========================
# Window
# ===========================

window_name = "Homography Calibration (Reference ID=2)"

user32 = ctypes.windll.user32
screen_width = user32.GetSystemMetrics(0)
screen_height = user32.GetSystemMetrics(1)

cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, screen_width, screen_height)


print("=" * 60)
print(f"기준 마커: {REFERENCE_DICT} ID:{REFERENCE_ID}  (한 변 {MARKER_SIZE_CM}cm)")
print("기준 마커가 한번이라도 탐지되면 그 위치로 호모그래피가 고정(LOCK)됩니다.")
print("[S] 고정된 호모그래피 저장   [ESC] 종료")
print("=" * 60)


# 한번 탐지되면 계속 유지되는 고정 값
locked_H = None
locked_pts = None

while True:

    ret, frame = cap.read()

    if not ret:
        print("프레임 수신 실패")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corners, ids, rejected = detector.detectMarkers(gray)


    # =====================================================
    # 아직 고정되지 않았다면, 기준 마커 탐지 시 고정
    # =====================================================

    if locked_H is None and ids is not None:

        ids_flat = np.array(ids).flatten()

        for marker_corner, marker_id in zip(corners, ids_flat):

            marker_id = int(marker_id)

            if marker_id != REFERENCE_ID:
                continue

            pts = marker_corner.reshape((4, 2))

            H = compute_homography(pts, REFERENCE_WORLD_CORNERS)

            if H is None:
                continue

            locked_H = H
            locked_pts = pts.copy()

            print("기준 마커 탐지됨 -> 호모그래피 고정(LOCK)")
            print("H =\n", locked_H)

            break


    # =====================================================
    # 고정된 결과 시각화 (계속 유지)
    # =====================================================

    if locked_pts is not None:

        pts_int = locked_pts.astype(int)

        cv2.polylines(frame, [pts_int], True, (0, 255, 255), 3)

        corner_labels = ["P0(TL)", "P1(TR)", "P2(BR)", "P3(BL)"]

        for i in range(4):

            x, y = pts_int[i]
            wx, wy = REFERENCE_WORLD_CORNERS[i]

            cv2.circle(frame, (int(x), int(y)), 6, (0, 0, 255), -1)

            cv2.putText(
                frame,
                f"{corner_labels[i]} ({wx:.1f}, {wy:.1f})cm",
                (int(x) + 8, int(y) - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2
            )


    # =====================================================
    # 상태 텍스트
    # =====================================================

    status_color = (0, 255, 0) if locked_H is not None else (0, 0, 255)
    status_text = "LOCKED" if locked_H is not None else "WAITING FOR REFERENCE..."

    cv2.putText(
        frame, status_text, (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2
    )

    cv2.putText(
        frame,
        "[S] Save homography   [ESC] Quit",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
    )

    cv2.imshow(window_name, frame)


    # =====================================================
    # 키 입력 처리
    # =====================================================

    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break

    elif key in (ord('s'), ord('S')):

        if locked_H is None:
            print("아직 기준 마커가 탐지되지 않아 저장할 수 없습니다.")
            continue

        np.savez(
            HOMOGRAPHY_FILE,
            H=locked_H,
            reference_dict=REFERENCE_DICT,
            reference_id=REFERENCE_ID,
            marker_size_cm=MARKER_SIZE_CM,
            reference_world_corners=REFERENCE_WORLD_CORNERS
        )

        print(f"호모그래피 저장 완료: {HOMOGRAPHY_FILE}")
        print("H =\n", locked_H)


# ===========================
# 종료
# ===========================

cap.release()
cv2.destroyAllWindows()