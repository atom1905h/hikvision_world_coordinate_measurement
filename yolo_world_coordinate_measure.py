import cv2
import numpy as np
import ctypes
import os
from datetime import datetime
from dotenv import load_dotenv
from ultralytics import YOLO

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
# YOLO 모델 설정
# ===========================

YOLO_MODEL_PATH = "yolov8n.pt"   # 사용할 가중치 경로로 교체하세요
CONF_THRESHOLD = 0.5

# 특정 클래스만 탐지하고 싶으면 이름을 넣어주세요 (예: ["person"])
# None이면 전체 클래스를 탐지합니다.
TARGET_CLASSES = None

model = YOLO(YOLO_MODEL_PATH)
class_names = model.names  # {class_id: class_name}


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

window_name = "YOLO Detection -> World Coordinate (Homography)"

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


    # =====================================================
    # 1. YOLO로 객체 탐지
    # =====================================================

    results = model(frame, conf=CONF_THRESHOLD, verbose=False)[0]

    detected_objects = []  # [(class_name, conf, (x1,y1,x2,y2)), ...]

    for box in results.boxes:

        class_id = int(box.cls[0])
        class_name = class_names[class_id]

        if TARGET_CLASSES is not None and class_name not in TARGET_CLASSES:
            continue

        conf = float(box.conf[0])

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        detected_objects.append((class_name, conf, (x1, y1, x2, y2)))


    # =====================================================
    # 2. 탐지된 객체를 실제 좌표로 변환 & 시각화
    #    - 바닥에 닿는 지점(바운딩박스 하단 중앙)을 기준점으로 사용
    # =====================================================

    for class_name, conf, (x1, y1, x2, y2) in detected_objects:

        # 바운딩박스 하단 중앙 (바닥 접지점) - 호모그래피는 지면 기준이므로
        # 박스 중심이 아니라 발밑 지점을 변환해야 오차가 적다
        foot_x = (x1 + x2) / 2.0
        foot_y = y2

        world_x, world_y = pixel_to_world(H, (foot_x, foot_y))

        p1 = (int(x1), int(y1))
        p2 = (int(x2), int(y2))
        foot_point = (int(foot_x), int(foot_y))

        box_color = (0, 255, 0)


        # -----------------------------------------------
        # 바운딩박스
        # -----------------------------------------------

        cv2.rectangle(
            frame,
            p1,
            p2,
            box_color,
            2
        )


        # -----------------------------------------------
        # 접지점 (실제 좌표 변환 기준점)
        # -----------------------------------------------

        cv2.circle(
            frame,
            foot_point,
            6,
            (0, 0, 255),
            -1
        )


        # -----------------------------------------------
        # 라벨 (클래스명 / 신뢰도 / 실제 좌표)
        # -----------------------------------------------

        label = f"{class_name} {conf:.2f}"
        coord_text = f"({world_x:.1f}, {world_y:.1f})cm"

        cv2.putText(
            frame,
            label,
            (p1[0], p1[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            box_color,
            2
        )

        cv2.putText(
            frame,
            coord_text,
            (foot_point[0] + 8, foot_point[1] + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2
        )


    # =====================================================
    # 상태 텍스트
    # =====================================================

    cv2.putText(
        frame,
        f"Homography loaded (Reference: {REFERENCE_DICT} ID:{REFERENCE_IDS})",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Detected objects: {len(detected_objects)}",
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