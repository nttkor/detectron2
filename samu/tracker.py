"""
PoseTracker v4.1 모듈
=====================

목표:
- YOLO ID: 프레임 내 객체 추적 (DeepSORT 역할)
- Pose ID: 재식별 (Re-ID) - 장면이 바뀌어도 같은 사람 인식

화면 표시:
- Y:5(0.95) P:1 형식으로 YOLO ID와 Pose ID 둘 다 표시
- NEW 상태: 빨간색
- CONFIRMED 상태: 파란색
"""

from ultralytics import YOLO
import cv2
import numpy as np
import json
import os
import time
import datetime
from collections import deque


class PoseTrackerV4:
    def __init__(
        self,
        model_path="yolo11n-pose.pt",
        pose_similarity_threshold=2.5,  # 포즈 매칭 임계값 (관대하게 변경 1.5 -> 2.5)
        stabilization_frames=3,         # Pose ID 확정까지 필요한 프레임 수 (빠르게 확정 5 -> 3)
        max_missing_frames=90,          # 사라져도 이력 유지하는 프레임 수 (3초)
        pose_history_size=30,           # 저장할 포즈 이력 개수 (10 -> 30, 1초 평균으로 안정성 강화)
        scale_history_size=15,          # scale smoothing용 이력 개수 (10 -> 15)
    ):
        self.model = YOLO(model_path)
        self.pose_similarity_threshold = pose_similarity_threshold
        self.stabilization_frames = stabilization_frames
        self.max_missing_frames = max_missing_frames
        self.pose_history_size = pose_history_size
        self.scale_history_size = scale_history_size

        # Pose ID별 상태 관리 (영구 식별용)
        self.persons = {}
        self.next_pose_id = 1
        
        # YOLO ID -> Pose ID 매핑 (안정성 확보용)
        self.yolo_id_to_pose_id = {}

        # 로그 저장용
        self.tracking_log = []
        self.frame_index = 0
        
        # 통계용 데이터: {pose_id: {yolo_ids: set(), frame_count: int, start_frame: int, end_frame: int}}
        self.stats = {}

        # 키포인트 인덱스 (COCO 17 keypoints)
        self.LEFT_SHOULDER = 5
        self.RIGHT_SHOULDER = 6
        self.LEFT_HIP = 11
        self.RIGHT_HIP = 12

    def get_id_color(self, id):
        """ID별 고유 색상 생성 (랜덤이지만 ID별로 고정)"""
        np.random.seed(id)
        # 너무 어두운 색은 피해서 생성 (50~255)
        color = np.random.randint(50, 255, size=3).tolist()
        return tuple(color)

    def reset(self):
        """모든 트래킹 상태를 초기화"""
        self.persons = {}
        self.yolo_id_to_pose_id = {}
        self.next_pose_id = 1
        self.tracking_log = []
        self.frame_index = 0
        self.stats = {}
        self.model = YOLO("yolo11n-pose.pt")

    def compute_robust_scale(self, kp, bbox, confidence_threshold=0.5):
        """다중 기준으로 scale 계산"""
        scales = []
        
        left_hip_valid = kp[self.LEFT_HIP, 2] > confidence_threshold
        right_hip_valid = kp[self.RIGHT_HIP, 2] > confidence_threshold
        left_shoulder_valid = kp[self.LEFT_SHOULDER, 2] > confidence_threshold
        right_shoulder_valid = kp[self.RIGHT_SHOULDER, 2] > confidence_threshold
        
        # Torso length
        if left_hip_valid and right_hip_valid and left_shoulder_valid and right_shoulder_valid:
            mid_hip = (kp[self.LEFT_HIP, :2] + kp[self.RIGHT_HIP, :2]) / 2
            mid_shoulder = (kp[self.LEFT_SHOULDER, :2] + kp[self.RIGHT_SHOULDER, :2]) / 2
            torso_length = np.linalg.norm(mid_shoulder - mid_hip)
            if torso_length > 10:
                scales.append(torso_length)
        
        # Shoulder width
        if left_shoulder_valid and right_shoulder_valid:
            shoulder_width = np.linalg.norm(kp[self.LEFT_SHOULDER, :2] - kp[self.RIGHT_SHOULDER, :2])
            if shoulder_width > 10:
                scales.append(shoulder_width * 1.25)
        
        # Hip width
        if left_hip_valid and right_hip_valid:
            hip_width = np.linalg.norm(kp[self.LEFT_HIP, :2] - kp[self.RIGHT_HIP, :2])
            if hip_width > 10:
                scales.append(hip_width * 1.67)
        
        # Bbox height (fallback)
        bbox_height = bbox[3] - bbox[1]
        if bbox_height > 50:
            scales.append(bbox_height * 0.3)
        
        if scales:
            return np.median(scales)
        return max(bbox_height * 0.3, 50)

    def normalize_keypoints(self, kp, bbox, scale_history, confidence_threshold=0.5):
        """포즈 정규화"""
        if kp.shape[1] == 3:
            confs = kp[:, 2]
            valid_mask = confs > confidence_threshold
            valid_kpts_xy = kp[valid_mask][:, :2]
        else:
            valid_kpts_xy = kp[:, :2]
        
        if valid_kpts_xy.shape[0] < 4:
            return np.array([]), scale_history
        
        # 중심점
        left_hip_valid = kp[self.LEFT_HIP, 2] > confidence_threshold
        right_hip_valid = kp[self.RIGHT_HIP, 2] > confidence_threshold
        
        if left_hip_valid and right_hip_valid:
            center = (kp[self.LEFT_HIP, :2] + kp[self.RIGHT_HIP, :2]) / 2
        else:
            center = np.mean(valid_kpts_xy, axis=0)
        
        # Scale (smoothing)
        current_scale = self.compute_robust_scale(kp, bbox, confidence_threshold)
        scale_history.append(current_scale)
        smoothed_scale = np.mean(scale_history)
        
        normalized_coords = (kp[:, :2] - center) / smoothed_scale
        return normalized_coords, scale_history

    def get_pose_similarity(self, kp1, kp2):
        """포즈 유사도 (L2 거리)"""
        if kp1.size == 0 or kp2.size == 0 or kp1.shape != kp2.shape:
            return np.inf
        return np.linalg.norm(kp1 - kp2)

    def get_average_pose(self, pose_history):
        """평균 포즈"""
        if not pose_history:
            return np.array([])
        poses = [p for p in pose_history if p.size > 0]
        if not poses:
            return np.array([])
        return np.mean(poses, axis=0)

    def find_pose_match(self, norm_kp, current_frame):
        """
        포즈 기반으로 기존 Pose ID 찾기
        - 모든 기존 ID와 비교해서 가장 유사한 것 찾기
        - 장면이 바뀌어도 포즈로 재식별
        """
        best_id = -1
        best_score = np.inf

        for pid, state in self.persons.items():
            # 이미 이번 프레임에서 매칭된 ID는 스킵
            if state['matched_this_frame']:
                continue
            
            # 너무 오래 사라진 ID는 스킵
            if current_frame - state['last_seen_frame'] > self.max_missing_frames:
                continue

            # 평균 포즈와 비교
            avg_pose = self.get_average_pose(state['pose_history'])
            pose_sim = self.get_pose_similarity(norm_kp, avg_pose)

            if pose_sim < self.pose_similarity_threshold and pose_sim < best_score:
                best_score = pose_sim
                best_id = pid

        return best_id, best_score

    def process_frame(self, frame):
        """프레임 처리 및 시각화"""
        self.frame_index += 1
        results = self.model.track(frame, persist=True, verbose=False)

        # 모든 Pose ID의 matched_this_frame 초기화
        for pid in self.persons:
            self.persons[pid]['matched_this_frame'] = False

        # 현재 프레임 결과: {idx: (yolo_id, pose_id, match_method, score)}
        current_results = {}
        kpts_data = None

        if (
            results[0].keypoints is not None
            and results[0].keypoints.data.numel() > 0
        ):
            kpts_data = results[0].keypoints.data.cpu().numpy()
            boxes_data = results[0].boxes.xyxy.cpu().numpy()
            
            # YOLO track_id 가져오기
            yolo_track_ids = None
            if results[0].boxes.id is not None:
                yolo_track_ids = results[0].boxes.id.cpu().numpy().astype(int)

            for i, kp in enumerate(kpts_data):
                bbox = boxes_data[i]
                
                # YOLO ID (없으면 -1)
                yolo_id = yolo_track_ids[i] if yolo_track_ids is not None and i < len(yolo_track_ids) else -1
                
                # 포즈 정규화
                temp_scale_history = deque(maxlen=self.scale_history_size)
                norm_kp, _ = self.normalize_keypoints(kp, bbox, temp_scale_history)
                
                if norm_kp.size == 0:
                    continue

                # === 매칭 로직 개선: YOLO ID 우선 + Pose ID 보조 ===
                pose_id = -1
                match_method = ""
                score = np.inf

                # 1. YOLO ID가 있고 이미 매핑되어 있으면 그대로 사용 (안정성 최우선)
                if yolo_id != -1 and yolo_id in self.yolo_id_to_pose_id:
                    mapped_pid = self.yolo_id_to_pose_id[yolo_id]
                    if mapped_pid in self.persons:
                        pose_id = mapped_pid
                        match_method = "YOLO_LINK"
                        score = 0.0  # YOLO ID를 신뢰하므로 score는 0
                
                # 2. 매핑된 게 없으면 포즈 매칭 시도
                if pose_id == -1:
                    pose_id, score = self.find_pose_match(norm_kp, self.frame_index)
                    
                    if pose_id != -1:
                        match_method = "POSE_MATCH"
                    else:
                        # 새 Pose ID 발급
                        pose_id = self.next_pose_id
                        self.next_pose_id += 1
                        match_method = "NEW"
                        score = np.inf
                
                # 3. 매핑 정보 업데이트
                if yolo_id != -1:
                    self.yolo_id_to_pose_id[yolo_id] = pose_id

                # Pose ID 상태 생성/업데이트
                if pose_id not in self.persons:
                    self.persons[pose_id] = {
                        'pose_history': deque(maxlen=self.pose_history_size),
                        'scale_history': deque(maxlen=self.scale_history_size),
                        'last_bbox': bbox,
                        'last_seen_frame': self.frame_index,
                        'match_count': 0,
                        'confirmed': False,
                        'matched_this_frame': True,
                        'last_yolo_id': yolo_id,
                    }
                
                # 통계 업데이트
                if pose_id not in self.stats:
                    self.stats[pose_id] = {
                        'yolo_ids': set(),
                        'frame_count': 0,
                        'start_frame': self.frame_index,
                        'end_frame': self.frame_index
                    }
                if yolo_id != -1:
                    self.stats[pose_id]['yolo_ids'].add(int(yolo_id))
                self.stats[pose_id]['frame_count'] += 1
                self.stats[pose_id]['end_frame'] = self.frame_index

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

                if state['match_count'] >= self.stabilization_frames:
                    state['confirmed'] = True

                current_results[i] = (yolo_id, pose_id, match_method, score)

                # 디버그 출력
                status = "CONFIRMED" if state['confirmed'] else "NEW"
                print(f"[Frame {self.frame_index}] idx={i}: YOLO={yolo_id}, POSE={pose_id}, method={match_method}, score={score:.3f}, count={state['match_count']}, {status}")

        # 오래된 Pose ID 정리
        ids_to_remove = [pid for pid, state in self.persons.items() 
                         if self.frame_index - state['last_seen_frame'] > self.max_missing_frames]
        for pid in ids_to_remove:
            print(f"[Frame {self.frame_index}] Pose ID {pid} 삭제 (오래됨)")
            del self.persons[pid]
            # yolo mapping에서도 정리 (선택사항, 자동 갱신되므로 필수 아님)

        # 시각화
        annotated = frame.copy()

        for idx, (yolo_id, pose_id, match_method, score) in current_results.items():
            bbox = results[0].boxes.xyxy[idx].cpu().numpy()
            bx1, by1, bx2, by2 = map(int, bbox)

            state = self.persons[pose_id]
            
            # 색상: 확정=ID별 고유색, NEW=빨간색
            if state['confirmed']:
                color = self.get_id_color(pose_id)
            else:
                color = (0, 0, 255)  # 빨간색

            # 라벨: Y:YOLO_ID (Conf) P:POSE_ID
            # 1. YOLO ID + 인식률 (항상 표시)
            # YOLO가 주는 conf 점수(keypoints 평균)를 사용하거나, 우리가 계산한 score를 사용
            yolo_conf = 0.0
            if kpts_data is not None:
                # 현재 사람의 키포인트 평균 confidence
                kp_confs = kpts_data[idx][:, 2]
                yolo_conf = kp_confs[kp_confs > 0.5].mean() if (kp_confs > 0.5).any() else 0.0

            label_parts = [f"Y:{yolo_id}({yolo_conf:.2f})"]

            # 2. Pose ID (확정되었을 때만 표시)
            if state['confirmed']:
                label_parts.append(f"P:{pose_id}")
            else:
                # 확정 전에는 카운트만 작게 표시하거나 생략 (여기선 NEW 표시)
                label_parts.append(f"NEW({state['match_count']})")
            
            label = " ".join(label_parts)

            # 보색 계산 (가독성)
            text_color = (255 - color[0], 255 - color[1], 255 - color[2])

            # bbox
            cv2.rectangle(annotated, (bx1, by1), (bx2, by2), color, 2)

            # 라벨 배경 + 텍스트
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
            
            # 배경 박스 (BBox 색상으로 채우기)
            cv2.rectangle(annotated, (bx1, by1 - th - 10), (bx1 + tw + 6, by1), color, -1)
            
            # 텍스트 (보색으로 그리기)
            cv2.putText(annotated, label, (bx1 + 3, by1 - 5), font, font_scale, text_color, thickness, cv2.LINE_AA)

            # 키포인트
            if kpts_data is not None:
                kp = kpts_data[idx]
                for j in range(kp.shape[0]):
                    x, y, conf = kp[j]
                    if conf > 0.5:
                        cv2.circle(annotated, (int(x), int(y)), 4, color, -1)

            # 로그
            self.tracking_log.append({
                "frame": self.frame_index,
                "yolo_id": int(yolo_id),
                "pose_id": int(pose_id),
                "bbox": [bx1, by1, bx2, by2],
                "confirmed": state['confirmed'],
                "match_count": state['match_count'],
                "match_method": match_method,
            })

        # 안내 텍스트 (하단)
        h, w = annotated.shape[:2]
        info_text = "Y:YOLO_ID  P:POSE_ID  |  quit: Q  reset: R"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(info_text, font, 0.7, 2)
        text_x = (w - tw) // 2
        text_y = h - 15
        cv2.rectangle(annotated, (text_x - 8, text_y - th - 8), (text_x + tw + 8, text_y + 4), (0, 0, 0), -1)
        cv2.putText(annotated, info_text, (text_x, text_y), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # === 상단 HUD (Timecode + Stability Stats) ===
        # 투명 배경 (alpha=0.3)
        overlay = annotated.copy()
        hud_h = 70  # HUD 높이 증가 (40 -> 70)
        cv2.rectangle(overlay, (0, 0), (w, hud_h), (0, 0, 0), -1)
        annotated = cv2.addWeighted(overlay, 0.3, annotated, 0.7, 0)

        # 시간 계산 (30fps 가정)
        seconds = self.frame_index // 30
        frames_mod = self.frame_index % 30  # 초 단위 나머지 프레임 (0~29)
        time_str = time.strftime("%H:%M:%S", time.gmtime(seconds))
        
        # 통계 계산
        # 1. 현재 화면에 있는 사람 수 (PID 기준)
        current_people_count = sum(1 for p in self.persons.values() if p['matched_this_frame'])
        
        # 2. 누적 통계 (지금까지 발생한 총 사건 수)
        cumulative_pose_ids = len(self.stats)  # 지금까지 생성된 총 Pose ID 수
        cumulative_yolo_swaps = sum(len(stat['yolo_ids']) for stat in self.stats.values()) # 지금까지 발생한 총 YOLO ID 변경 횟수
        
        # HUD 텍스트
        # 시간:프레임 | 현재인원: N명 | 누적변경: YOLO(N) vs POSE(N)
        stats_text = f"{time_str}:{frames_mod:02d} | NOW: {current_people_count} | CHANGE: YOLO({cumulative_yolo_swaps}) vs POSE({cumulative_pose_ids})"
        
        # 텍스트 크기 및 위치 계산 (중앙 정렬)
        font_scale = 1.5  # 폰트 크기 3배 (0.6 -> 1.5)
        thickness = 3
        (text_w, text_h), _ = cv2.getTextSize(stats_text, font, font_scale, thickness)
        text_x = (w - text_w) // 2  # 화면 중앙 정렬
        text_y = 50  # 상단 여백

        # 텍스트 그리기 (그림자 효과)
        cv2.putText(annotated, stats_text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness + 3, cv2.LINE_AA) # 그림자
        cv2.putText(annotated, stats_text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA) # 메인 (흰색)

        return annotated

    def export_tracking_log(self, json_path):
        """트래킹 로그 저장"""
        dirpath = os.path.dirname(json_path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.tracking_log, f, ensure_ascii=False, indent=2)
            
    def export_report(self, report_path):
        """PID vs YID 통계 리포트 생성"""
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
        
        # PID별 분석
        for pid, data in self.stats.items():
            yolo_count = len(data['yolo_ids'])
            duration = data['end_frame'] - data['start_frame'] + 1
            
            # "강도" 계산: (PID 1개) / (YOLO ID 개수) 
            # 1.0 = 완벽 (YOLO가 1번 바뀔 때 PID도 1번)
            # < 1.0 = PID가 더 강함 (YOLO가 여러 번 바뀌어도 PID는 유지됨 -> Re-ID 성공)
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

        # 파일 저장
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        
        print(f"📊 리포트 생성 완료: {report_path}")


if __name__ == "__main__":
    video_path = r"D:/git/detectron2/video/movepeople753.mp4"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "output")
    
    # 파일명에 날짜/시간 추가
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_json = os.path.join(output_dir, f"tracking_log_{timestamp}.json")
    output_report = os.path.join(output_dir, f"tracking_report_{timestamp}.txt")
    
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 동영상을 열 수 없습니다: {video_path}")
        exit(1)

    tracker = PoseTrackerV4(
        model_path="yolo11n-pose.pt",
        pose_similarity_threshold=2.5,  # 포즈 매칭 임계값 (관대하게 변경 1.5 -> 2.5)
        stabilization_frames=3,         # Pose ID 확정까지 필요한 프레임 수 (빠르게 확정 5 -> 3)
        max_missing_frames=90,          # 사라져도 이력 유지하는 프레임 수 (3초)
        pose_history_size=30,           # 저장할 포즈 이력 개수 (10 -> 30, 1초 평균으로 안정성 강화)
        scale_history_size=15,          # scale smoothing용 이력 개수 (10 -> 15)
    )

    window_name = "PoseTracker v4.1 (YOLO + Pose ID)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # 녹화 준비용 카운트다운 (10초)
    print("🎥 녹화 준비! 10초 후에 시작합니다...")
    for i in range(10, 0, -1):
        print(f"⏳ {i}...")
        # 검은 화면에 카운트다운 표시
        countdown_img = np.zeros((720, 1280, 3), dtype=np.uint8)
        text = f"Rec Start in {i}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(text, font, 3, 5)
        tx, ty = (1280 - tw) // 2, (720 + th) // 2
        cv2.putText(countdown_img, text, (tx, ty), font, 3, (255, 255, 255), 5, cv2.LINE_AA)
        cv2.imshow(window_name, countdown_img)
        cv2.waitKey(1000)

    last_reset_time = None

    while True:
        ret, frame = cap.read()
        if not ret:
            # 동영상이 끝나면 처음으로 되감기 (루프 재생)
            print("🔁 영상 루프 재생")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        annotated = tracker.process_frame(frame)

        # RESET 표시
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

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
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

    cap.release()
    cv2.destroyAllWindows()

    # 종료 시 로그 및 리포트 저장
    tracker.export_tracking_log(output_json)
    tracker.export_report(output_report)
    
    print(f"💾 로그 저장: {output_json}")
    print(f"📊 리포트 저장: {output_report}")
