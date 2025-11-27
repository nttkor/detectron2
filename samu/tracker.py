from ultralytics import YOLO
import cv2
import numpy as np

class PoseTracker:
    def __init__(self, model_path="yolo11n-pose.pt", similarity_threshold=0.5):
        self.model = YOLO(model_path)
        self.similarity_threshold = similarity_threshold
        
        # ID 관리 (카메라별로 관리하거나 전역으로 관리할 수 있음. 여기선 단순화를 위해 개별 인스턴스 관리)
        self.person_id_to_pose_history = {}
        self.next_person_id = 1
        
        # 키포인트 인덱스
        self.LEFT_HIP = 11
        self.RIGHT_HIP = 12
        self.LEFT_SHOULDER = 5
        self.RIGHT_SHOULDER = 6

    def normalize_keypoints(self, person_keypoints, confidence_threshold=0.5):
        """키포인트 정규화 (위치 및 크기)"""
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
        if kpts1.size == 0 or kpts2.size == 0 or kpts1.shape != kpts2.shape:
            return np.inf
        return np.linalg.norm(kpts1 - kpts2)

    def process_frame(self, frame):
        """프레임을 받아 추적 결과와 ID가 그려진 이미지를 반환"""
        # persist=True로 YOLO 자체 트래킹 활성화 (기본 트래킹 + 포즈 보정)
        results = self.model.track(frame, persist=True, verbose=False)
        
        current_frame_assigned_ids = {}
        newly_assigned_pose_history = {}
        
        if results[0].keypoints is not None and results[0].keypoints.data.numel() > 0:
            keypoints_data = results[0].keypoints.data.cpu().numpy()
            boxes_data = results[0].boxes.xyxy.cpu().numpy()
            
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

        # 포즈 이력 업데이트
        self.person_id_to_pose_history = newly_assigned_pose_history

        # 시각화
        annotated_frame = results[0].plot()
        
        # 커스텀 ID 그리기 (YOLO ID 대신 포즈 ID 표시)
        for idx, assigned_id in current_frame_assigned_ids.items():
            bbox = results[0].boxes.xyxy[idx].cpu().numpy()
            x1, y1, x2, y2 = map(int, bbox)
            cv2.putText(annotated_frame, f"ID: {assigned_id}", (x1, max(20, y1-10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        return annotated_frame