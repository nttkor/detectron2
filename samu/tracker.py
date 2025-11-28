"""
================================================================================
PoseTracker v4.2 모듈
================================================================================

[개요]
YOLO11 포즈 모델을 사용한 다중 사람 추적 및 재식별(Re-ID) 시스템

[핵심 개념]
- YOLO ID (Y:): 프레임 내 객체 추적용 (DeepSORT 역할, 임시 ID)
- Pose ID (P:): 포즈 기반 재식별용 (장면이 바뀌어도 같은 사람 인식, 영구 ID)

[화면 표시]
- Y:5(0.95) P:1  → YOLO ID=5, 신뢰도=0.95, Pose ID=1 (확정됨)
- Y:8(0.87) NEW(2) → YOLO ID=8, 아직 확정 안됨 (2프레임 매칭)

[매칭 전략]
1. YOLO ID가 이미 Pose ID에 매핑되어 있으면 그대로 사용 (안정성 최우선)
2. 매핑이 없으면 포즈 유사도로 기존 Pose ID 찾기 (Re-ID)
3. 둘 다 실패하면 새 Pose ID 발급

[키 조작]
- Q: 종료
- R: 트래커 초기화 (모든 ID 리셋)
- Space: 카운트다운 스킵

[출력 파일]
- tracking_log_YYYYMMDD_HHMMSS.json: 프레임별 상세 로그
- tracking_report_YYYYMMDD_HHMMSS.txt: PID vs YID 안정성 분석 리포트
================================================================================
"""

# ============================================================================
# 라이브러리 임포트
# ============================================================================
from ultralytics import YOLO                    # YOLO11 포즈 모델
import cv2                                      # OpenCV (영상 처리/시각화)
import numpy as np                              # 수치 연산
import json                                     # JSON 로그 저장
import os                                       # 파일/경로 처리
import time                                     # 시간 측정
import datetime                                 # 타임스탬프 생성
from collections import deque                   # 고정 크기 큐 (이력 관리)


# ============================================================================
# PoseTrackerV4 클래스
# ============================================================================
class PoseTrackerV4:
    """
    포즈 기반 다중 사람 추적 및 재식별 클래스
    
    [주요 기능]
    - YOLO11 포즈 모델로 사람 감지 및 키포인트 추출
    - 포즈 정규화 (위치/크기 불변)
    - 포즈 유사도 기반 ID 매칭 (Re-ID)
    - YOLO ID ↔ Pose ID 매핑 관리
    - 실시간 시각화 및 통계 HUD
    """
    
    # ------------------------------------------------------------------------
    # 초기화
    # ------------------------------------------------------------------------
    def __init__(
        self,
        model_path="yolo11n-pose.pt",           # YOLO 포즈 모델 경로
        pose_similarity_threshold=2.5,          # 포즈 매칭 임계값 (클수록 관대)
        stabilization_frames=3,                 # Pose ID 확정까지 필요한 연속 프레임 수
        max_missing_frames=90,                  # 사라져도 이력 유지하는 프레임 수 (3초@30fps)
        pose_history_size=30,                   # 저장할 포즈 이력 개수 (평균 계산용)
        scale_history_size=15,                  # scale smoothing용 이력 개수
    ):
        """
        트래커 초기화
        
        Args:
            model_path: YOLO 포즈 모델 파일 경로
            pose_similarity_threshold: 포즈 매칭 임계값 (L2 거리, 클수록 관대)
            stabilization_frames: NEW → CONFIRMED 전환에 필요한 연속 매칭 프레임 수
            max_missing_frames: 사람이 사라져도 Pose ID 이력을 유지하는 최대 프레임 수
            pose_history_size: 각 Pose ID당 저장할 정규화 포즈 개수 (평균 포즈 계산용)
            scale_history_size: 포즈 정규화 시 scale smoothing에 사용할 이력 개수
        """
        self.model = YOLO(model_path)                               # YOLO 모델 로드
        self.pose_similarity_threshold = pose_similarity_threshold  # 포즈 매칭 임계값
        self.stabilization_frames = stabilization_frames            # 확정 필요 프레임 수
        self.max_missing_frames = max_missing_frames                # 최대 미출현 허용 프레임
        self.pose_history_size = pose_history_size                  # 포즈 이력 크기
        self.scale_history_size = scale_history_size                # 스케일 이력 크기

        # --- Pose ID 관리 ---
        self.persons = {}                       # {pose_id: 상태 딕셔너리} - 각 Pose ID별 상태
        self.next_pose_id = 1                   # 다음에 발급할 Pose ID 번호
        
        # --- YOLO ID → Pose ID 매핑 ---
        self.yolo_id_to_pose_id = {}            # {yolo_id: pose_id} - 안정성 확보용 매핑 테이블

        # --- 로그 및 통계 ---
        self.tracking_log = []                  # 프레임별 상세 로그 (JSON 저장용)
        self.frame_index = 0                    # 현재까지 처리한 프레임 번호
        self.stats = {}                         # {pose_id: {yolo_ids, frame_count, ...}} - 통계 데이터

        # --- COCO 17 키포인트 인덱스 ---
        self.LEFT_SHOULDER = 5                  # 왼쪽 어깨
        self.RIGHT_SHOULDER = 6                 # 오른쪽 어깨
        self.LEFT_HIP = 11                      # 왼쪽 엉덩이
        self.RIGHT_HIP = 12                     # 오른쪽 엉덩이

    # ------------------------------------------------------------------------
    # 유틸리티 메서드
    # ------------------------------------------------------------------------
    def get_id_color(self, id):
        """
        Pose ID별 고유 색상 생성
        
        Args:
            id: Pose ID 번호
        Returns:
            (B, G, R) 튜플 - OpenCV BGR 색상
        """
        np.random.seed(id)                      # ID를 시드로 사용 → 같은 ID는 항상 같은 색
        color = np.random.randint(50, 255, size=3).tolist()  # 50~255 범위 (너무 어두운 색 방지)
        return tuple(color)

    def reset(self):
        """
        트래커 전체 초기화 (R 키 입력 시 호출)
        
        - 모든 Pose ID 및 매핑 정보 삭제
        - YOLO 모델 재로드 (내부 트래커 리셋)
        - 통계 초기화
        """
        self.persons = {}                       # Pose ID 상태 초기화
        self.yolo_id_to_pose_id = {}            # YOLO→Pose 매핑 초기화
        self.next_pose_id = 1                   # Pose ID 번호 리셋
        self.tracking_log = []                  # 로그 초기화
        self.frame_index = 0                    # 프레임 번호 리셋
        self.stats = {}                         # 통계 초기화
        self.model = YOLO("yolo11n-pose.pt")    # YOLO 모델 재로드 (내부 트래커 리셋)

    # ------------------------------------------------------------------------
    # 포즈 정규화 관련 메서드
    # ------------------------------------------------------------------------
    def compute_robust_scale(self, kp, bbox, confidence_threshold=0.5):
        """
        다중 기준으로 포즈 스케일 계산 (앙상블)
        
        여러 기준의 중앙값을 사용해 outlier에 강한 스케일 추정
        
        Args:
            kp: 키포인트 배열 (17, 3) - [x, y, confidence]
            bbox: 바운딩 박스 [x1, y1, x2, y2]
            confidence_threshold: 유효 키포인트 판단 임계값
        Returns:
            float: 추정된 스케일 값 (정규화 분모로 사용)
        """
        scales = []                             # 각 기준별 스케일 저장
        
        # --- 키포인트 유효성 체크 ---
        left_hip_valid = kp[self.LEFT_HIP, 2] > confidence_threshold
        right_hip_valid = kp[self.RIGHT_HIP, 2] > confidence_threshold
        left_shoulder_valid = kp[self.LEFT_SHOULDER, 2] > confidence_threshold
        right_shoulder_valid = kp[self.RIGHT_SHOULDER, 2] > confidence_threshold
        
        # --- 기준 1: Torso length (가장 신뢰할 수 있음) ---
        if left_hip_valid and right_hip_valid and left_shoulder_valid and right_shoulder_valid:
            mid_hip = (kp[self.LEFT_HIP, :2] + kp[self.RIGHT_HIP, :2]) / 2          # 엉덩이 중앙
            mid_shoulder = (kp[self.LEFT_SHOULDER, :2] + kp[self.RIGHT_SHOULDER, :2]) / 2  # 어깨 중앙
            torso_length = np.linalg.norm(mid_shoulder - mid_hip)                   # 몸통 길이
            if torso_length > 10:               # 최소 10픽셀
                scales.append(torso_length)
        
        # --- 기준 2: Shoulder width ---
        if left_shoulder_valid and right_shoulder_valid:
            shoulder_width = np.linalg.norm(kp[self.LEFT_SHOULDER, :2] - kp[self.RIGHT_SHOULDER, :2])
            if shoulder_width > 10:
                scales.append(shoulder_width * 1.25)  # torso 스케일로 변환 (경험적 비율)
        
        # --- 기준 3: Hip width ---
        if left_hip_valid and right_hip_valid:
            hip_width = np.linalg.norm(kp[self.LEFT_HIP, :2] - kp[self.RIGHT_HIP, :2])
            if hip_width > 10:
                scales.append(hip_width * 1.67)       # torso 스케일로 변환 (경험적 비율)
        
        # --- 기준 4: Bbox height (fallback) ---
        bbox_height = bbox[3] - bbox[1]         # y2 - y1
        if bbox_height > 50:
            scales.append(bbox_height * 0.3)    # torso 스케일로 변환 (경험적 비율)
        
        # --- 앙상블: 중앙값 사용 (outlier에 강함) ---
        if scales:
            return np.median(scales)
        return max(bbox_height * 0.3, 50)       # 모든 기준 실패 시 fallback

    def normalize_keypoints(self, kp, bbox, scale_history, confidence_threshold=0.5):
        """
        포즈 정규화 (위치 + 크기 불변 변환)
        
        1. 중심점 이동: mid-hip 또는 유효 키포인트 평균을 원점으로
        2. 스케일 정규화: smoothed scale로 나누기
        
        Args:
            kp: 키포인트 배열 (17, 3)
            bbox: 바운딩 박스
            scale_history: 스케일 이력 (deque) - smoothing용
            confidence_threshold: 유효 키포인트 판단 임계값
        Returns:
            (normalized_coords, scale_history): 정규화된 좌표 (17, 2), 업데이트된 스케일 이력
        """
        # --- 유효 키포인트 필터링 ---
        if kp.shape[1] == 3:                    # [x, y, conf] 형식
            confs = kp[:, 2]
            valid_mask = confs > confidence_threshold
            valid_kpts_xy = kp[valid_mask][:, :2]
        else:                                   # [x, y] 형식
            valid_kpts_xy = kp[:, :2]
        
        if valid_kpts_xy.shape[0] < 4:          # 최소 4개 키포인트 필요
            return np.array([]), scale_history
        
        # --- 중심점 계산 ---
        left_hip_valid = kp[self.LEFT_HIP, 2] > confidence_threshold
        right_hip_valid = kp[self.RIGHT_HIP, 2] > confidence_threshold
        
        if left_hip_valid and right_hip_valid:
            center = (kp[self.LEFT_HIP, :2] + kp[self.RIGHT_HIP, :2]) / 2  # mid-hip (가장 안정적)
        else:
            center = np.mean(valid_kpts_xy, axis=0)  # 유효 키포인트 평균
        
        # --- 스케일 계산 및 smoothing ---
        current_scale = self.compute_robust_scale(kp, bbox, confidence_threshold)
        scale_history.append(current_scale)     # 이력에 추가
        smoothed_scale = np.mean(scale_history) # 이동 평균
        
        # --- 정규화 ---
        normalized_coords = (kp[:, :2] - center) / smoothed_scale
        return normalized_coords, scale_history

    # ------------------------------------------------------------------------
    # 포즈 매칭 관련 메서드
    # ------------------------------------------------------------------------
    def get_pose_similarity(self, kp1, kp2):
        """
        두 포즈 간 유사도 계산 (L2 거리)
        
        Args:
            kp1, kp2: 정규화된 키포인트 배열 (17, 2)
        Returns:
            float: L2 거리 (작을수록 유사)
        """
        if kp1.size == 0 or kp2.size == 0 or kp1.shape != kp2.shape:
            return np.inf                       # 비교 불가 → 무한대
        return np.linalg.norm(kp1 - kp2)        # 전체 키포인트의 L2 거리

    def get_average_pose(self, pose_history):
        """
        포즈 이력의 평균 포즈 계산
        
        Args:
            pose_history: 정규화된 포즈 리스트 (deque)
        Returns:
            평균 포즈 배열 (17, 2) 또는 빈 배열
        """
        if not pose_history:
            return np.array([])
        poses = [p for p in pose_history if p.size > 0]  # 유효한 포즈만
        if not poses:
            return np.array([])
        return np.mean(poses, axis=0)           # 요소별 평균

    def find_pose_match(self, norm_kp, current_frame):
        """
        포즈 기반으로 기존 Pose ID 찾기 (Re-ID 핵심)
        
        모든 기존 Pose ID와 비교해서 가장 유사한 것 찾기
        
        Args:
            norm_kp: 현재 정규화된 포즈
            current_frame: 현재 프레임 번호
        Returns:
            (best_id, best_score): 매칭된 Pose ID (-1이면 없음), 유사도 점수
        """
        best_id = -1                            # 매칭된 ID (-1: 없음)
        best_score = np.inf                     # 최소 유사도 점수

        for pid, state in self.persons.items():
            # --- 이미 매칭된 ID는 스킵 ---
            if state['matched_this_frame']:
                continue
            
            # --- 너무 오래 사라진 ID는 스킵 ---
            if current_frame - state['last_seen_frame'] > self.max_missing_frames:
                continue

            # --- 평균 포즈와 비교 ---
            avg_pose = self.get_average_pose(state['pose_history'])
            pose_sim = self.get_pose_similarity(norm_kp, avg_pose)

            # --- 임계값 이내 & 최소 점수 갱신 ---
            if pose_sim < self.pose_similarity_threshold and pose_sim < best_score:
                best_score = pose_sim
                best_id = pid

        return best_id, best_score

    # ------------------------------------------------------------------------
    # 메인 프레임 처리 메서드
    # ------------------------------------------------------------------------
    def process_frame(self, frame):
        """
        단일 프레임 처리 (감지 → 매칭 → 시각화)
        
        [처리 흐름]
        1. YOLO 포즈 추론
        2. 각 사람에 대해:
           - YOLO ID 확인
           - 포즈 정규화
           - Pose ID 매칭 (YOLO 매핑 우선 → 포즈 매칭 → 새 ID)
           - 상태 업데이트
        3. 시각화 (bbox, 키포인트, 라벨, HUD)
        
        Args:
            frame: 입력 프레임 (BGR)
        Returns:
            annotated: 시각화된 프레임 (BGR)
        """
        self.frame_index += 1                   # 프레임 번호 증가
        results = self.model.track(frame, persist=True, verbose=False)  # YOLO 추론

        # --- 모든 Pose ID의 매칭 플래그 초기화 ---
        for pid in self.persons:
            self.persons[pid]['matched_this_frame'] = False

        # --- 현재 프레임 결과 저장용 ---
        current_results = {}                    # {idx: (yolo_id, pose_id, match_method, score)}
        kpts_data = None                        # 키포인트 데이터

        # ====================================================================
        # 감지 결과 처리
        # ====================================================================
        if (
            results[0].keypoints is not None
            and results[0].keypoints.data.numel() > 0
        ):
            kpts_data = results[0].keypoints.data.cpu().numpy()  # (N, 17, 3)
            boxes_data = results[0].boxes.xyxy.cpu().numpy()     # (N, 4)
            
            # --- YOLO track_id 가져오기 ---
            yolo_track_ids = None
            if results[0].boxes.id is not None:
                yolo_track_ids = results[0].boxes.id.cpu().numpy().astype(int)

            # --- 각 감지된 사람 처리 ---
            for i, kp in enumerate(kpts_data):
                bbox = boxes_data[i]            # 바운딩 박스
                
                # YOLO ID (없으면 -1)
                yolo_id = yolo_track_ids[i] if yolo_track_ids is not None and i < len(yolo_track_ids) else -1
                
                # --- 포즈 정규화 ---
                temp_scale_history = deque(maxlen=self.scale_history_size)
                norm_kp, _ = self.normalize_keypoints(kp, bbox, temp_scale_history)
                
                if norm_kp.size == 0:           # 정규화 실패 → 스킵
                    continue

                # ============================================================
                # 매칭 로직: YOLO ID 우선 + Pose ID 보조
                # ============================================================
                pose_id = -1                    # 매칭된 Pose ID
                match_method = ""               # 매칭 방법
                score = np.inf                  # 유사도 점수

                # --- 전략 1: YOLO ID가 이미 매핑되어 있으면 그대로 사용 ---
                if yolo_id != -1 and yolo_id in self.yolo_id_to_pose_id:
                    mapped_pid = self.yolo_id_to_pose_id[yolo_id]
                    if mapped_pid in self.persons:  # 아직 유효한 Pose ID인지 확인
                        pose_id = mapped_pid
                        match_method = "YOLO_LINK"  # YOLO 매핑으로 연결
                        score = 0.0             # YOLO ID 신뢰 → score 0
                
                # --- 전략 2: 매핑 없으면 포즈 매칭 시도 ---
                if pose_id == -1:
                    pose_id, score = self.find_pose_match(norm_kp, self.frame_index)
                    
                    if pose_id != -1:
                        match_method = "POSE_MATCH"  # 포즈로 재식별 성공
                    else:
                        # --- 전략 3: 새 Pose ID 발급 ---
                        pose_id = self.next_pose_id
                        self.next_pose_id += 1
                        match_method = "NEW"    # 새로운 사람
                        score = np.inf
                
                # --- YOLO ID → Pose ID 매핑 업데이트 ---
                if yolo_id != -1:
                    self.yolo_id_to_pose_id[yolo_id] = pose_id

                # ============================================================
                # Pose ID 상태 생성/업데이트
                # ============================================================
                if pose_id not in self.persons:
                    self.persons[pose_id] = {
                        'pose_history': deque(maxlen=self.pose_history_size),  # 포즈 이력
                        'scale_history': deque(maxlen=self.scale_history_size),  # 스케일 이력
                        'last_bbox': bbox,          # 마지막 bbox
                        'last_seen_frame': self.frame_index,  # 마지막 출현 프레임
                        'match_count': 0,           # 연속 매칭 횟수
                        'confirmed': False,         # 확정 여부
                        'matched_this_frame': True, # 이번 프레임 매칭 여부
                        'last_yolo_id': yolo_id,    # 마지막 YOLO ID
                    }
                
                # --- 통계 업데이트 ---
                if pose_id not in self.stats:
                    self.stats[pose_id] = {
                        'yolo_ids': set(),          # 연결된 YOLO ID 집합
                        'frame_count': 0,           # 총 출현 프레임 수
                        'start_frame': self.frame_index,  # 첫 출현 프레임
                        'end_frame': self.frame_index     # 마지막 출현 프레임
                    }
                if yolo_id != -1:
                    self.stats[pose_id]['yolo_ids'].add(int(yolo_id))  # YOLO ID 기록
                self.stats[pose_id]['frame_count'] += 1
                self.stats[pose_id]['end_frame'] = self.frame_index

                # --- 상태 업데이트 ---
                state = self.persons[pose_id]
                
                # 포즈 이력 업데이트 (smoothing 적용)
                norm_kp_smoothed, state['scale_history'] = self.normalize_keypoints(
                    kp, bbox, state['scale_history']
                )
                if norm_kp_smoothed.size > 0:
                    state['pose_history'].append(norm_kp_smoothed)
                
                state['last_bbox'] = bbox
                state['last_seen_frame'] = self.frame_index
                state['match_count'] += 1
                state['matched_this_frame'] = True
                state['last_yolo_id'] = yolo_id

                # --- 확정 여부 체크 ---
                if state['match_count'] >= self.stabilization_frames:
                    state['confirmed'] = True

                current_results[i] = (yolo_id, pose_id, match_method, score)

                # --- 디버그 출력 ---
                status = "CONFIRMED" if state['confirmed'] else "NEW"
                print(f"[Frame {self.frame_index}] idx={i}: YOLO={yolo_id}, POSE={pose_id}, method={match_method}, score={score:.3f}, count={state['match_count']}, {status}")

        # ====================================================================
        # 오래된 Pose ID 정리
        # ====================================================================
        ids_to_remove = [pid for pid, state in self.persons.items() 
                         if self.frame_index - state['last_seen_frame'] > self.max_missing_frames]
        for pid in ids_to_remove:
            print(f"[Frame {self.frame_index}] Pose ID {pid} 삭제 (오래됨)")
            del self.persons[pid]

        # ====================================================================
        # 시각화
        # ====================================================================
        annotated = frame.copy()                # 원본 프레임 복사

        for idx, (yolo_id, pose_id, match_method, score) in current_results.items():
            bbox = results[0].boxes.xyxy[idx].cpu().numpy()
            bx1, by1, bx2, by2 = map(int, bbox)

            state = self.persons[pose_id]
            
            # --- 색상 결정: 확정=ID별 고유색, NEW=빨간색 ---
            if state['confirmed']:
                color = self.get_id_color(pose_id)
            else:
                color = (0, 0, 255)              # 빨간색 (BGR)

            # --- 라벨 생성: Y:YOLO_ID(conf) P:POSE_ID ---
            yolo_conf = 0.0
            if kpts_data is not None:
                kp_confs = kpts_data[idx][:, 2]  # 키포인트 신뢰도
                yolo_conf = kp_confs[kp_confs > 0.5].mean() if (kp_confs > 0.5).any() else 0.0

            label_parts = [f"Y:{yolo_id}({yolo_conf:.2f})"]  # YOLO ID + 신뢰도

            if state['confirmed']:
                label_parts.append(f"P:{pose_id}")  # 확정된 Pose ID
            else:
                label_parts.append(f"NEW({state['match_count']})")  # 미확정 (카운트 표시)
            
            label = " ".join(label_parts)

            # --- 보색 계산 (텍스트 가독성) ---
            text_color = (255 - color[0], 255 - color[1], 255 - color[2])

            # --- bbox 그리기 ---
            cv2.rectangle(annotated, (bx1, by1), (bx2, by2), color, 2)

            # --- 라벨 배경 + 텍스트 ---
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
            cv2.rectangle(annotated, (bx1, by1 - th - 10), (bx1 + tw + 6, by1), color, -1)  # 배경
            cv2.putText(annotated, label, (bx1 + 3, by1 - 5), font, font_scale, text_color, thickness, cv2.LINE_AA)

            # --- 키포인트 그리기 ---
            if kpts_data is not None:
                kp = kpts_data[idx]
                for j in range(kp.shape[0]):
                    x, y, conf = kp[j]
                    if conf > 0.5:
                        cv2.circle(annotated, (int(x), int(y)), 4, color, -1)

            # --- 로그 기록 ---
            self.tracking_log.append({
                "frame": self.frame_index,
                "yolo_id": int(yolo_id),
                "pose_id": int(pose_id),
                "bbox": [bx1, by1, bx2, by2],
                "confirmed": state['confirmed'],
                "match_count": state['match_count'],
                "match_method": match_method,
            })

        # ====================================================================
        # 하단 안내 텍스트
        # ====================================================================
        h, w = annotated.shape[:2]
        info_text = "Y:YOLO_ID  P:POSE_ID  |  quit: Q  reset: R"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(info_text, font, 0.7, 2)
        text_x = (w - tw) // 2                  # 중앙 정렬
        text_y = h - 15
        cv2.rectangle(annotated, (text_x - 8, text_y - th - 8), (text_x + tw + 8, text_y + 4), (0, 0, 0), -1)
        cv2.putText(annotated, info_text, (text_x, text_y), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # ====================================================================
        # 상단 HUD (타임코드 + 통계)
        # ====================================================================
        overlay = annotated.copy()              # 투명 배경용 복사본
        hud_h = 70                              # HUD 높이
        cv2.rectangle(overlay, (0, 0), (w, hud_h), (0, 0, 0), -1)
        annotated = cv2.addWeighted(overlay, 0.3, annotated, 0.7, 0)  # 30% 투명

        # --- 시간 계산 (30fps 가정) ---
        seconds = self.frame_index // 30
        frames_mod = self.frame_index % 30
        time_str = time.strftime("%H:%M:%S", time.gmtime(seconds))
        
        # --- 통계 계산 ---
        current_people_count = sum(1 for p in self.persons.values() if p['matched_this_frame'])  # 현재 인원
        cumulative_pose_ids = len(self.stats)   # 총 Pose ID 수
        cumulative_yolo_swaps = sum(len(stat['yolo_ids']) for stat in self.stats.values())  # 총 YOLO ID 변경 수
        
        # --- HUD 텍스트 ---
        stats_text = f"{time_str}:{frames_mod:02d} | NOW: {current_people_count} | CHANGE: YOLO({cumulative_yolo_swaps}) vs POSE({cumulative_pose_ids})"
        
        font_scale = 1.5
        thickness = 3
        (text_w, text_h), _ = cv2.getTextSize(stats_text, font, font_scale, thickness)
        text_x = (w - text_w) // 2              # 중앙 정렬
        text_y = 50

        # 그림자 + 메인 텍스트
        cv2.putText(annotated, stats_text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
        cv2.putText(annotated, stats_text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        return annotated

    # ------------------------------------------------------------------------
    # 내보내기 메서드
    # ------------------------------------------------------------------------
    def export_tracking_log(self, json_path):
        """
        트래킹 로그를 JSON 파일로 저장
        
        Args:
            json_path: 저장할 JSON 파일 경로
        """
        dirpath = os.path.dirname(json_path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.tracking_log, f, ensure_ascii=False, indent=2)
            
    def export_report(self, report_path):
        """
        PID vs YID 안정성 분석 리포트 생성
        
        Pose ID가 YOLO ID보다 얼마나 안정적인지 분석
        
        Args:
            report_path: 저장할 리포트 파일 경로
        """
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = []
        lines.append(f"PoseTracker Analysis Report")
        lines.append(f"Date: {now}")
        lines.append(f"=" * 40)
        lines.append(f"")
        lines.append(f"Total Frames Processed: {self.frame_index}")
        lines.append(f"Total Pose IDs Created: {len(self.stats)}")
        lines.append(f"")
        lines.append(f"ID Stability Analysis (PID vs YID)")
        lines.append(f"-" * 40)
        
        # --- PID별 분석 ---
        for pid, data in self.stats.items():
            yolo_count = len(data['yolo_ids'])
            duration = data['end_frame'] - data['start_frame'] + 1
            
            # 안정성 비율: 1/YOLO개수 (낮을수록 Re-ID 성공)
            stability_ratio = 1.0 / yolo_count if yolo_count > 0 else 1.0
            
            lines.append(f"[Pose ID {pid}]")
            lines.append(f"  - Duration: {duration} frames")
            lines.append(f"  - Associated YOLO IDs ({yolo_count}): {sorted(list(data['yolo_ids']))}")
            
            if yolo_count > 1:
                lines.append(f"  - ⭐ Re-ID Success! Maintained identity across {yolo_count} YOLO tracks.")
                lines.append(f"  - Stability Score: {stability_ratio:.2f} (Lower is better for single ID)")
            else:
                lines.append(f"  - Stable tracking (1:1 match).")
            lines.append(f"")

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        
        print(f"📊 리포트 생성 완료: {report_path}")


# ============================================================================
# 메인 실행부
# ============================================================================
if __name__ == "__main__":
    """
    단독 실행 진입점
    
    [실행 흐름]
    1. 동영상 로드
    2. 트래커 초기화
    3. 카운트다운 (10초, Space로 스킵)
    4. 메인 루프 (프레임 처리 → 시각화 → 키 입력)
    5. 종료 시 로그/리포트 저장
    """
    
    # --- 경로 설정 ---
    video_path = r"D:/git/detectron2/video/movepeople753.mp4"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "output")
    
    # 타임스탬프로 파일명 생성
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_json = os.path.join(output_dir, f"tracking_log_{timestamp}.json")
    output_report = os.path.join(output_dir, f"tracking_report_{timestamp}.txt")
    
    os.makedirs(output_dir, exist_ok=True)

    # --- 동영상 로드 ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 동영상을 열 수 없습니다: {video_path}")
        exit(1)

    # --- 트래커 초기화 ---
    tracker = PoseTrackerV4(
        model_path="yolo11n-pose.pt",
        pose_similarity_threshold=2.5,          # 포즈 매칭 임계값 (관대하게)
        stabilization_frames=3,                 # 빠른 확정 (3프레임)
        max_missing_frames=90,                  # 3초간 이력 유지
        pose_history_size=30,                   # 1초 평균 (30프레임)
        scale_history_size=15,                  # scale smoothing
    )

    # --- 윈도우 생성 ---
    window_name = "PoseTracker v4.2 (YOLO + Pose ID)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # --- 카운트다운 (10초) ---
    print("🎥 녹화 준비! 10초 후에 시작합니다... (스페이스바: 즉시 시작)")
    for i in range(10, 0, -1):
        print(f"⏳ {i}...")
        countdown_img = np.zeros((720, 1280, 3), dtype=np.uint8)
        text = f"Rec Start in {i}"
        sub_text = "Press SPACE to Start Now"
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(text, font, 3, 5)
        tx, ty = (1280 - tw) // 2, (720 + th) // 2
        
        cv2.putText(countdown_img, text, (tx, ty), font, 3, (255, 255, 255), 5, cv2.LINE_AA)
        cv2.putText(countdown_img, sub_text, (400, 600), font, 1, (200, 200, 200), 2, cv2.LINE_AA)
        
        cv2.imshow(window_name, countdown_img)
        
        key = cv2.waitKey(1000)
        if key == 32:                           # Spacebar
            print("🚀 즉시 시작!")
            break
        elif key == ord('q'):
            exit()

    # --- 메인 루프 ---
    last_reset_time = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("🔁 영상 루프 재생")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        annotated = tracker.process_frame(frame)

        # --- RESET 표시 (1초간) ---
        if last_reset_time is not None:
            elapsed = time.time() - last_reset_time
            if elapsed < 1.0:
                h, w = annotated.shape[:2]
                text = "RESET"
                font = cv2.FONT_HERSHEY_SIMPLEX
                (tw, th), _ = cv2.getTextSize(text, font, 1.5, 3)
                tx, ty = (w - tw) // 2, h // 2
                cv2.rectangle(annotated, (tx - 20, ty - th - 20), (tx + tw + 20, ty + 20), (0, 0, 255), -1)
                cv2.putText(annotated, text, (tx, ty), font, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
            else:
                last_reset_time = None

        cv2.imshow(window_name, annotated)

        # --- 키 입력 처리 ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):                     # Q: 종료
            break
        elif key == ord("r"):                   # R: 리셋
            print("🔁 트래커 초기화")
            if os.path.exists(output_json):
                try:
                    os.remove(output_json)
                    print(f"🗑 JSON 삭제: {output_json}")
                except:
                    pass
            tracker.reset()
            last_reset_time = time.time()
            print("✅ 초기화 완료")

    # --- 종료 처리 ---
    cap.release()
    cv2.destroyAllWindows()

    # --- 로그 및 리포트 저장 ---
    tracker.export_tracking_log(output_json)
    tracker.export_report(output_report)
    
    print(f"💾 로그 저장: {output_json}")
    print(f"📊 리포트 저장: {output_report}")
