"""
PoseTracker 모듈
================

이 모듈은 YOLO 포즈 모델(`yolo11n-pose.pt`)을 사용해
프레임 단위로 사람의 관절 키포인트를 추론하고,
정규화된 포즈 유사도에 기반해 **프레임 간 일관된 ID**를 부여/유지하는 예제 구현입니다.

Final Task 요약 (노트북 `yolo11_pose.ipynb`에서 구현한 내용 정리):
---------------------------------------------------------------
1. **키포인트 정규화 (normalize_keypoints)**
   - YOLO의 `results[0].keypoints`에서 단일 사람의 키포인트 (x, y, conf)를 가져와:
     - 신뢰도(confidence) 임계값보다 낮은 키포인트를 제거하고,
     - 왼쪽/오른쪽 엉덩이의 중간점(mid-hip)을 원점으로 평행 이동하여 위치를 정규화하고,
     - mid-hip ~ mid-shoulder(몸통 길이)를 1이 되도록 스케일링하여 크기를 정규화합니다.
   - 유효한 키포인트가 너무 적거나, 몸통 길이가 거의 0인 경우는 빈 배열을 반환하여
     이후 유사도 계산 시 `np.inf` 로 취급되도록 합니다.

2. **포즈 유사도 계산 (get_pose_similarity)**
   - 두 명의 사람에 대해 정규화된 키포인트 배열이 있으면,
     단순 유클리드 거리(`np.linalg.norm`)를 사용해 포즈 간 거리를 계산합니다.
   - 배열 크기가 다르거나 비어 있으면 `np.inf` 를 반환하여 "매칭 불가"로 처리합니다.

3. **포즈 기반 ID 할당 (process_frame)**
   - 매 프레임마다 YOLO `model.track(frame, persist=True)` 로 사람/포즈를 추론하고,
     각 사람의 정규화된 포즈를 이전 프레임의 포즈 이력(`person_id_to_pose_history`)과 비교합니다.
   - 가장 유사한(거리 최소) 이력이 있고, 그 거리가 `similarity_threshold` 아래이면
     해당 ID를 재사용합니다. 그렇지 않으면 새로운 ID를 발급합니다.
   - 현재 프레임의 포즈로 포즈 이력을 갱신하여, 시간이 지나도 포즈 변화에 적응하도록 합니다.

4. **시각화 (Custom ID Visualization)**
   - YOLO의 `results[0].plot()` 으로 기본 시각화를 얻은 뒤,
     각 사람의 바운딩 박스 상단에 `cv2.putText` 를 사용해
     `ID: N` 형식의 **커스텀 포즈 기반 ID** 를 녹색으로 표시합니다.

이 구현은 YOLO 내부 트래커 ID와는 독립적인, "포즈 기반 보조 ID" 예제이며,
향후 외형 특징(embedding)과 결합해 재식별(re-id)을 강화하는 등의 확장도 가능합니다.
"""

from ultralytics import YOLO
import cv2
import numpy as np
import json
import os
import time


class PoseTracker:
    def __init__(self, model_path="yolo11n-pose.pt", similarity_threshold=0.5):
        self.model = YOLO(model_path)
        self.similarity_threshold = similarity_threshold
        
        # ID 관리 (카메라별로 관리하거나 전역으로 관리할 수 있음. 여기선 단순화를 위해 개별 인스턴스 관리)
        self.person_id_to_pose_history = {}
        self.next_person_id = 1
        self.frame_index = 0  # 현재까지 처리한 프레임 번호

        # JSON 저장용 로그 (프레임별 ID, 바운딩 박스, 키포인트 기록)
        self.tracking_log = []
        self.preexisting_ids = set()   # 이전 실행에서 이미 존재하던 ID들 (JSON 기준)
        # ID 신뢰도 관리: 연속 매칭된 프레임 수 (여러 프레임을 보고 ID를 확정하기 위함)
        self.id_match_count = {}       # {id: 연속 매칭된 프레임 수}
        self.output_json_path = None   # 추적 로그를 저장/로드할 JSON 경로
        
        # 키포인트 인덱스
        self.LEFT_HIP = 11
        self.RIGHT_HIP = 12
        self.LEFT_SHOULDER = 5
        self.RIGHT_SHOULDER = 6

    # ------------------------------------------------------------------
    # JSON 기반 글로벌 ID 로드 (이전 실행 결과를 불러와 포즈 기반 매칭에 활용)
    # ------------------------------------------------------------------
    def load_global_ids(self, json_path: str):
        """
        이전 실행에서 저장해 둔 포즈 기반 ID JSON을 읽어,
        - person_id_to_pose_history 초기값 (글로벌 ID → 마지막 포즈)
        - preexisting_ids (이전부터 존재하던 ID 집합)
        - next_person_id (기존 ID의 최대값 + 1)
        을 설정합니다.
        """
        self.output_json_path = json_path

        if not os.path.exists(json_path):
            return

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return

        for item in data:
            track_id = int(item.get("track_id", -1))
            kpts_list = item.get("keypoints", [])
            if track_id < 0 or not kpts_list:
                continue
            kpts = np.array(kpts_list, dtype=float)
            normalized = self.normalize_keypoints(kpts)
            if normalized.size == 0:
                continue
            self.person_id_to_pose_history[track_id] = normalized

        if self.person_id_to_pose_history:
            self.preexisting_ids = set(self.person_id_to_pose_history.keys())
            self.next_person_id = max(self.person_id_to_pose_history.keys()) + 1

    def normalize_keypoints(self, person_keypoints, confidence_threshold=0.5):
        """
        단일 사람의 원시 키포인트를 **위치 + 크기 정규화**된 포즈 표현으로 변환합니다.

        - 신뢰도(conf)가 threshold 이상인 키포인트만 사용
        - 엉덩이 중앙(mid-hip)을 원점(0, 0)으로 평행 이동
        - mid-hip ~ mid-shoulder 거리(몸통 길이)를 1로 스케일링

        정규화가 불가능한 경우(유효 키포인트 부족, 몸통 길이 0 등)에는 빈 배열을 반환합니다.
        """
        # (N, 3) -> x, y, conf
        if person_keypoints.shape[1] == 3:
            valid_mask = person_keypoints[:, 2] > confidence_threshold
            valid_kpts_xy = person_keypoints[valid_mask][:, :2]
        else:
            valid_kpts_xy = person_keypoints[:, :2]

        if valid_kpts_xy.shape[0] < 2:
            return np.array([])

        # 1. 엉덩이 중앙 기준 위치 정규화
        kp = person_keypoints
        if (kp[self.LEFT_HIP, 2] > confidence_threshold and kp[self.RIGHT_HIP, 2] > confidence_threshold):
            mid_hip = (kp[self.LEFT_HIP, :2] + kp[self.RIGHT_HIP, :2]) / 2
        elif valid_kpts_xy.shape[0] > 0:
            mid_hip = np.mean(valid_kpts_xy, axis=0)
        else:
            return np.array([])

        normalized_coords = kp[:, :2] - mid_hip

        # 2. 몸통 길이 기준 크기 정규화
        if (kp[self.LEFT_SHOULDER, 2] > confidence_threshold and kp[self.RIGHT_SHOULDER, 2] > confidence_threshold):
            mid_shoulder = (kp[self.LEFT_SHOULDER, :2] + kp[self.RIGHT_SHOULDER, :2]) / 2
            torso_length = np.linalg.norm(mid_shoulder - mid_hip)
        else:
            # 어깨가 없으면 가장 먼 점의 거리 등을 대안으로 쓸 수 있으나 일단 생략
            return np.array([])
            
        if torso_length < 1e-6:
            return np.array([])

        return normalized_coords / torso_length

    def get_pose_similarity(self, kpts1, kpts2):
        """두 정규화된 포즈 벡터 간 유클리드 거리(작을수록 유사)를 계산"""
        if kpts1.size == 0 or kpts2.size == 0 or kpts1.shape != kpts2.shape:
            return np.inf
        return np.linalg.norm(kpts1 - kpts2)

    def process_frame(self, frame):
        """
        단일 프레임을 입력으로 받아:
        1) YOLO 포즈 추론 및 기본 시각화 실행
        2) 정규화된 포즈 유사도 기반으로 커스텀 ID를 부여/업데이트
        3) 프레임에 `ID: N` 텍스트를 그려 넣은 이미지를 반환합니다.
        """
        # 프레임 번호 증가
        self.frame_index += 1

        # persist=True로 YOLO 자체 트래킹 활성화 (기본 트래킹 + 포즈 보정)
        results = self.model.track(frame, persist=True, verbose=False)
        
        current_frame_assigned_ids = {}
        newly_assigned_pose_history = {}
        
        if results[0].keypoints is not None and results[0].keypoints.data.numel() > 0:
            keypoints_data = results[0].keypoints.data.cpu().numpy()
            boxes_data = results[0].boxes.xyxy.cpu().numpy()
            
            # 이번 프레임에서 연속 매칭 횟수를 다시 계산하기 위한 임시 딕셔너리
            new_match_count = {}

            for i, person_kpts in enumerate(keypoints_data):
                normalized_kpts = self.normalize_keypoints(person_kpts)
                bbox = boxes_data[i]
                
                assigned_id = -1
                min_sim = np.inf
                
                # 기존 포즈 이력과 비교
                if normalized_kpts.size > 0:
                    for pid, hist_kpts in self.person_id_to_pose_history.items():
                        sim = self.get_pose_similarity(normalized_kpts, hist_kpts)
                        if sim < min_sim:
                            min_sim = sim
                            assigned_id = pid
                
                # 유사도가 임계값 이내이면 기존 ID 유지, 아니면 새 ID
                # (참고: YOLO track ID가 이미 있다면 그걸 우선하되, 끊겼을 때 포즈로 재연결하는 로직으로 발전 가능)
                if assigned_id != -1 and min_sim < self.similarity_threshold:
                    pass 
                else:
                    assigned_id = self.next_person_id
                    self.next_person_id += 1
                
                current_frame_assigned_ids[i] = assigned_id
                if normalized_kpts.size > 0:
                    newly_assigned_pose_history[assigned_id] = normalized_kpts
                    # 이전 프레임에서 동일 ID가 있었다면 연속 매칭 카운트 +1, 아니면 1로 시작
                    prev_count = self.id_match_count.get(assigned_id, 0)
                    new_match_count[assigned_id] = prev_count + 1

        # 포즈 이력 업데이트
        self.person_id_to_pose_history = newly_assigned_pose_history
        self.id_match_count = new_match_count

        # 시각화
        annotated_frame = results[0].plot()

        # 화면 하단 중앙 안내 텍스트 (quit / reset)
        h, w = annotated_frame.shape[:2]
        info_text = "quit: Q    reset ID: R"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.9
        thickness = 2
        (tw, th), _ = cv2.getTextSize(info_text, font, font_scale, thickness)
        text_x = (w - tw) // 2
        text_y = h - 15  # 약간 위로 올림

        # 텍스트 가독성을 위한 반투명 배경 박스
        bg_margin = 8
        cv2.rectangle(
            annotated_frame,
            (text_x - bg_margin, text_y - th - bg_margin),
            (text_x + tw + bg_margin, text_y + bg_margin // 2),
            (0, 0, 0),
            -1,
        )

        cv2.putText(
            annotated_frame,
            info_text,
            (text_x, text_y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

        # 커스텀 ID 그리기 (YOLO ID 대신 포즈 ID 표시)
        for idx, assigned_id in current_frame_assigned_ids.items():
            bbox = results[0].boxes.xyxy[idx].cpu().numpy()
            x1, y1, x2, y2 = map(int, bbox)
            # 색상 및 라벨 규칙 (BGR):
            # - ID가 막 매칭되기 시작한 1~N프레임 동안  : 빨간색 + "NEW"
            # - N프레임 이상 연속으로 매칭된 안정 구간 : 파란색 + "ID:N"
            consecutive = self.id_match_count.get(assigned_id, 0)
            if consecutive < 30:  # 대략 1초(30FPS 기준) 동안은 NEW 상태로 표시
                color = (0, 0, 255)          # NEW 단계: 빨간색
                label_text = "NEW"
            else:
                color = (255, 0, 0)          # 안정화 단계: 파란색
                label_text = f"ID: {assigned_id}"
            cv2.putText(
                annotated_frame,
                label_text,
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
            )

            # JSON 로그용 기록 추가 (프레임, ID, 바운딩 박스, 키포인트)
            if results[0].keypoints is not None:
                kpts = results[0].keypoints.data[idx].cpu().numpy().tolist()
            else:
                kpts = []
            self.tracking_log.append(
                {
                    "frame": int(self.frame_index),
                    "track_id": int(assigned_id),
                    "bbox": [x1, y1, x2, y2],
                    "keypoints": kpts,
                }
            )

        return annotated_frame

    # ------------------------------------------------------------------
    # JSON 내보내기 유틸리티
    # ------------------------------------------------------------------
    def export_tracking_log(self, json_path: str):
        """
        지금까지 누적된 포즈 기반 ID 트래킹 로그를 JSON 파일로 저장합니다.

        JSON 포맷 예시:
        [
          {
            "frame": 1,
            "track_id": 1,
            "bbox": [x1, y1, x2, y2],
            "keypoints": [[x, y, conf], ...]
          },
          ...
        ]
        """
        dirpath = os.path.dirname(json_path)
        if dirpath and not os.path.isdir(dirpath):
            # 이미 같은 이름의 파일이 존재하는 경우 등 예외 상황은 그냥 건너뜀
            print(f"⚠ JSON 디렉터리를 생성할 수 없습니다: {dirpath}")
            return
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)

        # 기존 로그가 있으면 불러와서 뒤에 이어 붙임 (여러 실행 결과 누적)
        existing = []
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            except Exception:
                existing = []

        combined = existing + self.tracking_log
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(combined, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    """
    단독 실행용 진입점
    ------------------
    - 기본 동영상: D:/git/detectron2/video/movepeople753.mp4
    - `q` 키: 종료
    - `r` 키: JSON 로그 및 ID 상태 초기화 (다음 프레임부터 새 ID로 시작)
    - 필요하면 아래 `video_path` 를 다른 동영상 경로나 0(웹캠)으로 변경해서 사용
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "output")
    output_json = os.path.join(output_dir, "movepeople753_pose_ids.json")

    # 처리할 동영상 경로 (기본값)
    video_path = r"D:/git/detectron2/video/movepeople753.mp4"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 동영상을 열 수 없습니다: {video_path}")
        exit(1)

    tracker = PoseTracker(model_path="yolo11n-pose.pt", similarity_threshold=0.5)
    # 이전 실행의 JSON이 있다면 로드해서 글로벌 ID 매칭에 사용
    tracker.load_global_ids(output_json)

    window_name = "PoseTracker - movepeople753"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    last_reset_time = None  # 최근 reset 시각 (초)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        annotated = tracker.process_frame(frame)

        # R 누른 후 1초 동안 화면 중앙에 "RESET" 표시
        if last_reset_time is not None:
            elapsed = time.time() - last_reset_time
            if elapsed < 1.0:
                h, w = annotated.shape[:2]
                text = "RESET"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.5
                thickness = 3
                (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
                tx = (w - tw) // 2
                ty = h // 2
                cv2.rectangle(
                    annotated,
                    (tx - 20, ty - th - 20),
                    (tx + tw + 20, ty + 20),
                    (0, 0, 255),
                    -1,
                )
                cv2.putText(
                    annotated,
                    text,
                    (tx, ty),
                    font,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                    cv2.LINE_AA,
                )
            else:
                last_reset_time = None
        cv2.imshow(window_name, annotated)

        key = cv2.waitKey(1) & 0xFF
        # 'q' 키로 종료
        if key == ord("q"):
            break
        # 'r' 키로 JSON 및 ID 상태 초기화
        elif key == ord("r"):
            print("🔁 JSON 로그 및 ID 상태를 초기화합니다.")
            # JSON 파일 삭제
            if os.path.exists(output_json):
                try:
                    os.remove(output_json)
                    print(f"🗑 JSON 파일 삭제: {output_json}")
                except Exception as e:
                    print(f"⚠ JSON 파일 삭제 실패: {e}")
            # 트래커 내부 상태 초기화
            tracker.person_id_to_pose_history = {}
            tracker.preexisting_ids = set()
            tracker.id_match_count = {}
            tracker.tracking_log = []
            tracker.next_person_id = 1
            tracker.frame_index = 0
            last_reset_time = time.time()

    cap.release()
    cv2.destroyAllWindows()

    # JSON 로그 저장 (기존 로그 뒤에 이번 실행 결과를 이어 붙임)
    tracker.export_tracking_log(output_json)
    print(f"💾 포즈 기반 ID 로그를 저장했습니다: {output_json}")
