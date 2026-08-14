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
IP = os.getenv("CCTV_1_IP")

RTSP_URL = (
    f"rtsp://{USERNAME}:{PASSWORD}@{IP}:554/Streaming/Channels/101"
)


# ===========================
# 기준 마커 설정 (좌표계 원점 역할)
# ===========================

REFERENCE_DICT = "DICT_6X6_250"

# 사용할 4개 마커의 ID (아래 딕셔너리와 매칭됨)
REFERENCE_IDS = [0, 1, 2, 3]

# ArUco 코너 순서: pts[0]=TL, pts[1]=TR, pts[2]=BR, pts[3]=BL
# 각 마커의 P2(BR) 꼭짓점에 대응하는 실제 좌표(cm)
WORLD_COORDS_BY_ID = {
    0: (0.0, 180.0),
    1: (90.0, 0.0),
    2: (90.0, 180.0),
    3: (0.0, 0.0),
}


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

def compute_homography(image_points, world_points):
    """
    4개(이상) 마커의 이미지(픽셀) 좌표와 실제 좌표로부터
    호모그래피 행렬 H를 계산한다.

    image_points : (N, 2) ndarray, pixel 좌표
    world_points : (N, 2) ndarray, 실제 좌표 (cm), 같은 순서

    return: 3x3 호모그래피 행렬 H (실패 시 None)
    """

    H, status = cv2.findHomography(
        image_points.astype(np.float32),
        world_points.astype(np.float32)
    )

    return H


# ===========================
# Window
# ===========================

window_name = "Homography Calibration (4 Markers, P2 corners)"

user32 = ctypes.windll.user32
screen_width = user32.GetSystemMetrics(0)
screen_height = user32.GetSystemMetrics(1)

cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, screen_width, screen_height)


print("=" * 60)
print(f"기준 마커: {REFERENCE_DICT} ID:{REFERENCE_IDS}")
print("각 마커의 P2(BR) 꼭짓점을 사용합니다.")
print("4개 마커가 한 프레임에서 모두 탐지되면 그 위치로 호모그래피가 고정(LOCK)됩니다.")
print("[S] 고정된 호모그래피 저장   [ESC] 종료")
print("=" * 60)


# 한번 탐지되면 계속 유지되는 고정 값
locked_H = None
locked_pts_by_id = None  # {id: (x, y)} - 고정 당시 각 마커의 P2 픽셀 좌표

while True:

    ret, frame = cap.read()

    if not ret:
        print("프레임 수신 실패")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corners, ids, rejected = detector.detectMarkers(gray)


    # =====================================================
    # 아직 고정되지 않았다면, 현재 프레임에서 탐지된
    # 기준 마커들의 P2 좌표를 모아본다
    # =====================================================

    if locked_H is None and ids is not None:

        ids_flat = np.array(ids).flatten()

        detected_p2_by_id = {}

        for marker_corner, marker_id in zip(corners, ids_flat):

            marker_id = int(marker_id)

            if marker_id not in REFERENCE_IDS:
                continue

            pts = marker_corner.reshape((4, 2))

            # pts[0]=TL, pts[1]=TR, pts[2]=BR, pts[3]=BL
            p2 = pts[2]

            detected_p2_by_id[marker_id] = p2

        # 4개 마커가 전부 탐지된 경우에만 호모그래피 계산 & 고정
        if all(rid in detected_p2_by_id for rid in REFERENCE_IDS):

            image_points = np.array(
                [detected_p2_by_id[rid] for rid in REFERENCE_IDS],
                dtype=np.float32
            )

            world_points = np.array(
                [WORLD_COORDS_BY_ID[rid] for rid in REFERENCE_IDS],
                dtype=np.float32
            )

            H = compute_homography(image_points, world_points)

            if H is not None:
                locked_H = H
                locked_pts_by_id = {
                    rid: detected_p2_by_id[rid] for rid in REFERENCE_IDS
                }

                print("4개 기준 마커 모두 탐지됨 -> 호모그래피 고정(LOCK)")
                print("H =\n", locked_H)


    # =====================================================
    # 고정된 결과 시각화 (계속 유지)
    # =====================================================

    if locked_pts_by_id is not None:

        for rid in REFERENCE_IDS:

            x, y = locked_pts_by_id[rid]
            wx, wy = WORLD_COORDS_BY_ID[rid]

            cv2.circle(frame, (int(x), int(y)), 6, (0, 0, 255), -1)

            cv2.putText(
                frame,
                f"ID{rid} P2 ({wx:.1f}, {wy:.1f})cm",
                (int(x) + 8, int(y) - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2
            )

        # 4점을 잇는 폴리곤도 참고용으로 그려줌
        pts_int = np.array(
            [locked_pts_by_id[rid] for rid in REFERENCE_IDS],
            dtype=int
        )
        cv2.polylines(frame, [pts_int], True, (0, 255, 255), 2)


    # =====================================================
    # 상태 텍스트
    # =====================================================

    status_color = (0, 255, 0) if locked_H is not None else (0, 0, 255)
    status_text = "LOCKED" if locked_H is not None else "WAITING FOR ALL 4 REFERENCE MARKERS..."

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
            print("아직 4개 기준 마커가 모두 탐지되지 않아 저장할 수 없습니다.")
            continue

        np.savez(
            HOMOGRAPHY_FILE,
            H=locked_H,
            reference_dict=REFERENCE_DICT,
            reference_ids=REFERENCE_IDS,
            world_coords_by_id=WORLD_COORDS_BY_ID
        )

        print(f"호모그래피 저장 완료: {HOMOGRAPHY_FILE}")
        print("H =\n", locked_H)


# ===========================
# 종료
# ===========================

cap.release()
cv2.destroyAllWindows()