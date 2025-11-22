"""
================================================================================
Project: OneFormer Panoptic 3D Primitive Visualization (VP-based Line Drawing)
Author: User + Assistant Collaboration
Date: 2024.05.20 (Converted from Pose Estimation to Primitive Visualization)

[프로그램 상세 개요]
이 프로그램은 2D 이미지를 입력받아 Segmentation Mask, Depth 정보를 기반으로
'소실점(Vanishing Point)'을 추정하고, 객체 위에 **3D 와이어프레임(Primitive Line Drawing)**을 그려
이미지의 깊이감과 구조를 시각적으로 강조합니다.

[핵심 로직]
1. AI Model Pipeline: Segmentation (OneFormer) + Depth Estimation (MiDaS).
2. Vanishing Point (VP) Detection: 이미지 내 직선 분석을 통해 주 소실점을 찾습니다.
3. Primitive Drawing:
   - 클래스별로 큐브, 실린더, 평면 등 적절한 2D/3D 형상을 결정합니다.
   - 큐브 형태는 검출된 VP를 향하는 투시 원근법을 적용하여 깊이감을 표현합니다.
   - 모든 선은 **객체의 Segmentation Mask 내부에만** 그려지는 '마스크 제약'을 준수합니다.

[참조 링크 - 삭제됨]
(PCA, Back-Projection 관련 3D 기하학 참조는 Primitive Visualization 로직으로 전환됨에 따라 제거되었습니다.)

[사용 라이브러리]
- PyTorch & Transformers: 딥러닝 모델 로드 및 추론
- OpenCV: 이미지 처리, 그리기, 화면 출력
- NumPy: 행렬 연산
================================================================================
"""

import os              # 파일 및 디렉토리 경로 제어
import glob            # 폴더 내 이미지 파일 검색
import math            # 수학 연산
import torch           # 딥러닝 모델 구동을 위한 PyTorch
import cv2             # 이미지 처리 및 시각화 (OpenCV)
import time            # FPS 계산 및 성능 측정
import numpy as np     # 행렬 연산 및 마스크 처리
from PIL import Image  # 이미지 로딩
from transformers import ( # HuggingFace Transformers 라이브러리
    OneFormerProcessor, 
    OneFormerForUniversalSegmentation, 
    DPTImageProcessor, 
    DPTForDepthEstimation
)

# ============================================================================
# 환경 설정 및 상수 정의
# ============================================================================
IMAGE_DIR = "./images"                                     # 입력 이미지가 위치할 폴더 경로
SEGMENTATION_MODEL = "shi-labs/oneformer_ade20k_swin_large" # 범용 세그멘테이션 모델
DEPTH_MODEL = "Intel/dpt-hybrid-midas"                     # 깊이 추정 모델

# ============================================================================
# 소실점 검출기 (Vanishing Point Detector)
# ============================================================================

class VanishingPointDetector:
    """
    이미지 내의 직선들을 분석하여 주 소실점(Dominant Vanishing Point)을 추정하는 클래스
    """
    def find_vanishing_point(self, image):
        """
        이미지에서 소실점 (vx, vy)를 찾아 반환합니다. 실패 시 이미지 중심을 반환합니다.
        """
        # OpenCV는 BGR을 기본으로 사용
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 1. 엣지 검출 (Canny)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # 2. 직선 검출 (Hough Transform)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)
        
        if lines is None:
            return (w // 2, h // 2)

        filtered_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x1 == x2: continue
            
            slope = (y2 - y1) / (x2 - x1)
            angle = math.degrees(math.atan(slope))
            
            # 대각선 방향의 선들만 수집 (VP를 찾기 위한 주된 단서)
            if 15 < abs(angle) < 75:
                filtered_lines.append((x1, y1, x2, y2, slope))

        if not filtered_lines:
            return (w // 2, h // 2)

        A_matrix = []
        b_vector = []
        
        for x1, y1, x2, y2, m in filtered_lines:
            # Line equation: mx - y = -(y1 - m*x1)
            c = y1 - m * x1
            A_matrix.append([m, -1])
            b_vector.append([-c])
            
        if len(A_matrix) < 2:
            return (w // 2, h // 2)

        try:
            A = np.array(A_matrix)
            b = np.array(b_vector)
            # 최소 제곱법을 이용해 해(x, y) = 소실점을 구함
            vx, vy = np.linalg.lstsq(A, b, rcond=None)[0]
            vx, vy = int(vx), int(vy)
            
            # 소실점이 화면 밖 너무 멀리 있으면 중심점으로 제한
            if not (-2*w < vx < 3*w and -2*h < vy < 3*h):
                 return (w // 2, h // 2)
                 
            return (vx, vy)
        except:
            return (w // 2, h // 2)

# ============================================================================
# 유틸리티 함수
# ============================================================================

def get_color(idx):
    """인덱스 기반 색상 생성"""
    hue = int((idx * 137.5) % 180)
    hsv = np.uint8([[[hue, 255, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return tuple(map(int, bgr))

def get_shape_type_from_class(class_name):
    """클래스 이름에 따른 3D 형상 타입 결정"""
    cn = class_name.lower()
    
    # Outline Only
    if any(x in cn for x in ["window", "paint", "curtain", "mirror", "cushion", "palm", "sky", "poster", "picture"]): 
        return "outline_only"
    
    # Horizontal Plane (수평면)
    if any(x in cn for x in ["floor", "flooring", "ground", "road", "sidewalk", "runway", "ceiling", "carpet", "rug"]): 
        return "horizontal_plane"
    
    # Vertical Plane (수직면)
    if "wall" in cn: 
        return "wall"
    
    # Topographic (산, 언덕 등)
    if any(x in cn for x in ["mountain", "earth", "hill", "rock"]): 
        return "topographic"
    
    # Volumetric Cube (Perspective 적용 대상)
    if any(x in cn for x in ["bed", "table", "desk", "sofa", "couch", "cabinet", "box", "chest", "building", "house", "bridge", "chair", "bench", "seat", "stool", "door", "closet"]): 
        return "cube_perspective"
    
    # Cylinder (사람, 기둥 등)
    if any(x in cn for x in ["person", "people", "pedestrian"]): 
        return "person"
    if any(x in cn for x in ["lamp", "light", "pole", "column", "tree", "pillar"]): 
        return "cylinder_symmetric"
    
    return "outline_only"

def draw_line_in_mask(img, pt1, pt2, mask, color, thickness=1):
    """
    제1원칙: pt1에서 pt2로 가는 선을 마스크 내부에만 그려 파편화합니다.
    """
    x1, y1 = pt1
    x2, y2 = pt2
    
    dist = max(abs(x2-x1), abs(y2-y1))
    if dist == 0: return

    # 선을 따라 픽셀 좌표를 샘플링
    points = np.linspace(0, 1, int(dist) + 1)
    xx = np.int32(x1 + points * (x2 - x1))
    yy = np.int32(y1 + points * (y2 - y1))
    
    h, w = mask.shape
    valid_indices = []
    
    # 선의 픽셀들을 순회하며 마스크 내부에 있는지 확인
    for i, (cx, cy) in enumerate(zip(xx, yy)):
        is_valid = False
        if 0 <= cx < w and 0 <= cy < h:
            if mask[cy, cx]:
                is_valid = True

        if is_valid:
            valid_indices.append(i)
        elif valid_indices:
            # 마스크 경계를 벗어날 때, 그 전에 모은 유효 픽셀들로 선을 그립니다.
            start_idx = valid_indices[0]
            end_idx = valid_indices[-1]
            cv2.line(img, (xx[start_idx], yy[start_idx]), (xx[end_idx], yy[end_idx]), color, thickness)
            valid_indices = []
    
    # 마지막으로 남아있는 유효 픽셀들을 처리
    if valid_indices:
        start_idx = valid_indices[0]
        end_idx = valid_indices[-1]
        cv2.line(img, (xx[start_idx], yy[start_idx]), (xx[end_idx], yy[end_idx]), color, thickness)

# ============================================================================
# Shape Drawing Functions (Primitive Renderer)
# ============================================================================

def draw_outline_only(img, segment_mask, color, thickness=2):
    """객체 마스크의 외곽선만 그립니다."""
    mask_uint8 = (segment_mask.astype(np.uint8) * 255)
    # 외곽선(Contour) 검출
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, color, thickness)

def draw_wall_structure(img, bbox_2d, segment_mask, all_segments_info, seg_map_full, color, thickness=2):
    """벽 구조를 그리고, 인접한 객체와의 경계에 수직선을 추가합니다."""
    draw_outline_only(img, segment_mask, color, thickness)
    x_min, y_min, x_max, y_max = bbox_2d
    
    # 인접한 객체(문, 창문, 가구 등)와의 수직 분할선을 찾기 위한 로직
    # 간단히 해당 벽 마스크 경계에 인접한 다른 마스크의 x좌표를 수집
    split_x_coords = set()
    for info in all_segments_info:
        if info['id'] == segment_mask.mean() or not np.any(seg_map_full == info['id']): continue
        other_mask = (seg_map_full == info['id'])
        
        # 다른 객체의 경계점(y_min 또는 y_max 근처)이 벽의 x 범위 내에 있는지 확인
        y_idxs, x_idxs = np.where(other_mask)
        if len(x_idxs) > 0:
            for x, y in zip(x_idxs, y_idxs):
                if x_min <= x <= x_max and (abs(y - y_min) < 20 or abs(y - y_max) < 20):
                    split_x_coords.add(x)

    # 수집된 x좌표들을 필터링하여 일정한 간격의 수직선만 남김
    sorted_x = sorted(list(split_x_coords))
    filtered_x = []
    if sorted_x:
        last_x = sorted_x[0]
        filtered_x.append(last_x)
        for x in sorted_x[1:]:
            if x - last_x > 30: # 30 픽셀 이상 간격이 있어야 분할선으로 인정
                filtered_x.append(x)
                last_x = x
                
    for x in filtered_x:
        # 벽의 상단(y_min)과 하단(y_max)을 잇는 수직선을 마스크 내부만 통과하여 그림
        draw_line_in_mask(img, (x, y_min), (x, y_max), segment_mask, color, 1)

def draw_topographic(img, segment_mask, depth_map, color, thickness=1):
    """지형/산맥 객체에 깊이 등고선을 그립니다."""
    draw_outline_only(img, segment_mask, color, thickness)
    valid_depths = depth_map[segment_mask]
    if len(valid_depths) == 0: return
    
    depth_min, depth_max = valid_depths.min(), valid_depths.max()
    if depth_max - depth_min < 0.01: return
    
    levels = np.linspace(depth_min, depth_max, 8) # 8단계의 깊이 레벨 생성
    for level in levels[1:-1]:
        # 현재 깊이 레벨과 비슷한 영역만 마스크로 추출 (등고선처럼)
        level_mask = (np.abs(depth_map - level) < (depth_max - depth_min) * 0.05) & segment_mask
        mask_uint8 = (level_mask.astype(np.uint8) * 255)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, color, thickness)

def draw_cube_perspective(img, bbox_2d, segment_mask, color, vanishing_point, thickness=2):
    """
    소실점(VP)을 향한 투시 원근법이 적용된 큐브 와이어프레임을 그립니다.
    """
    x_min, y_min, x_max, y_max = bbox_2d
    vx, vy = vanishing_point  # 검출된 소실점 좌표
    
    # 1. Front Face (앞면) - Bounding Box 자체가 앞면
    front_verts = [
        (x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)
    ]
    # 앞면 그리기
    for i in range(4):
        p1 = front_verts[i]
        p2 = front_verts[(i + 1) % 4]
        draw_line_in_mask(img, p1, p2, segment_mask, color, thickness)
    
    # 2. Back Face (뒷면) 계산 - 소실점을 이용한 투영
    # 깊이 Factor: 객체의 크기에 따라 두께를 0.1~0.3 정도로 설정하여 원근감을 표현
    depth_factor = 0.2
    
    def project_point(px, py, vx, vy, factor):
        # P_back = P_front + (VP - P_front) * factor
        dx = vx - px
        dy = vy - py
        return int(px + dx * factor), int(py + dy * factor)
    
    # Back Face의 꼭짓점 계산
    back_verts = [project_point(px, py, vx, vy, depth_factor) for px, py in front_verts]
    
    # 3. Connecting Edges (연결선) - 깊이 방향
    for i in range(4):
        p_front = front_verts[i]
        p_back = back_verts[i]
        draw_line_in_mask(img, p_front, p_back, segment_mask, color, 1)

    # 4. Back Face 그리기
    for i in range(4):
        p1 = back_verts[i]
        p2 = back_verts[(i + 1) % 4]
        draw_line_in_mask(img, p1, p2, segment_mask, color, 1)

def draw_person_structure(img, bbox_2d, segment_mask, color, thickness=2):
    """사람 객체에 간단한 골격 구조를 그립니다."""
    x_min, y_min, x_max, y_max = bbox_2d
    width = x_max - x_min
    height = y_max - y_min
    
    draw_outline_only(img, segment_mask, color, thickness)

    # 간략화된 인체 비례 사용
    head_h = int(height * 0.15)
    torso_h = int(height * 0.40)
    center_x = (x_min + x_max) // 2
    
    y_neck = y_min + head_h
    y_waist = y_neck + torso_h
    
    # 목선
    draw_line_in_mask(img, (x_min + width//3, y_neck), (x_max - width//3, y_neck), segment_mask, color, 1)
    # 허리선
    draw_line_in_mask(img, (x_min, y_waist), (x_max, y_waist), segment_mask, color, 1)
    # 다리 (중앙 분할)
    draw_line_in_mask(img, (center_x, y_waist), (center_x, y_max), segment_mask, color, 1)
    # 팔 (양쪽)
    arm_offset = width // 4
    draw_line_in_mask(img, (x_min + arm_offset, y_neck), (x_min + arm_offset, y_waist), segment_mask, color, 1)
    draw_line_in_mask(img, (x_max - arm_offset, y_neck), (x_max - arm_offset, y_waist), segment_mask, color, 1)

def draw_cylinder_symmetric(img, bbox_2d, segment_mask, color, thickness=2):
    """기둥/나무 등 원통형 객체에 중심선과 수평선을 그립니다."""
    draw_outline_only(img, segment_mask, color, thickness)
    x_min, y_min, x_max, y_max = bbox_2d
    center_x = (x_min + x_max) // 2
    
    # 중심 수직선
    draw_line_in_mask(img, (center_x, y_min), (center_x, y_max), segment_mask, color, 1)
    
    # 수평 분할선
    steps = 4
    step_h = (y_max - y_min) // steps
    for i in range(1, steps):
        y = y_min + step_h * i
        draw_line_in_mask(img, (x_min, y), (x_max, y), segment_mask, color, 1)

# ============================================================================
# Main Class: 추론 및 시각화 통합 관리자
# ============================================================================

class Panoptic3DVisualizer:
    def __init__(self):
        """모델 로딩 및 하드웨어 설정"""
        print(f"Loading Segmentation Model: {SEGMENTATION_MODEL}...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Segmentation Model 로드
        self.processor = OneFormerProcessor.from_pretrained(SEGMENTATION_MODEL)
        self.model = OneFormerForUniversalSegmentation.from_pretrained(SEGMENTATION_MODEL).to(self.device)
        
        print(f"Loading Depth Model: {DEPTH_MODEL}...")
        # Depth Model 로드
        self.depth_processor = DPTImageProcessor.from_pretrained(DEPTH_MODEL)
        self.depth_model = DPTForDepthEstimation.from_pretrained(DEPTH_MODEL).to(self.device)
        
        # 소실점 검출기 초기화
        self.vp_detector = VanishingPointDetector()
        print("System Ready. Models loaded on:", self.device)

    def infer(self, image_path):
        """AI 추론 (Segmentation + Depth) 수행"""
        image = Image.open(image_path).convert("RGB")
        original_size = image.size 
        W, H = original_size
        
        # --- 1. Panoptic Segmentation ---
        inputs = self.processor(images=image, task_inputs=["panoptic"], return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        panoptic_segmentation = self.processor.post_process_panoptic_segmentation(
            outputs, target_sizes=[original_size[::-1]] # (H, W) 순서
        )[0]
        segments_info = panoptic_segmentation["segments_info"]
        segment_map = panoptic_segmentation["segmentation_map"].cpu().numpy()
        
        # --- 2. Depth Estimation ---
        depth_inputs = self.depth_processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            depth_outputs = self.depth_model(**depth_inputs)
            predicted_depth = depth_outputs.predicted_depth
        
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1), size=original_size[::-1], mode="bicubic", align_corners=False
        )
        depth_map = prediction.squeeze().cpu().numpy()
        
        # Depth Map 정규화 (0~1 범위)
        depth_map_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
        
        return np.array(image), segment_map, segments_info, depth_map_norm

    def visualize(self, image, segment_map, segments_info, depth_map):
        """
        추론 결과를 시각화하고 Primitive Line Drawing을 수행하는 메인 함수
        """
        # 이미지 전처리 (어둡게 + 깊이 맵 오버레이)
        vis_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # OpenCV 출력을 위해 BGR로 변환
        vis_img = cv2.addWeighted(vis_img, 0.3, np.zeros_like(vis_img), 0.7, 0) # 어둡게 처리
        
        depth_vis = (depth_map * 255).astype(np.uint8)
        depth_colormap = cv2.applyColorMap(depth_vis, cv2.COLORMAP_MAGMA)
        vis_img = cv2.addWeighted(vis_img, 0.6, depth_colormap, 0.4, 0) # 깊이맵 오버레이

        # 1. 소실점 검출
        vp = self.vp_detector.find_vanishing_point(vis_img)
        # 소실점 위치에 십자가 표시 (빨간색)
        cv2.drawMarker(vis_img, vp, (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

        # 2. 객체 정보 처리
        id2label = self.model.config.id2label
        enhanced_segments_info = []
        for info in segments_info:
            new_info = info.copy()
            new_info['label_name'] = id2label[info['label_id']]
            enhanced_segments_info.append(new_info)

        # 3. 객체별 Primitive Drawing 수행
        for info in enhanced_segments_info:
            segment_id = info['id']
            label_name = info['label_name']
            
            mask = (segment_map == segment_id)
            if not np.any(mask): continue
            
            # Bounding Box (2D) 계산
            y_indices, x_indices = np.where(mask)
            y_min, y_max = y_indices.min(), y_indices.max()
            x_min, x_max = x_indices.min(), x_indices.max()
            bbox_2d = (x_min, y_min, x_max, y_max)
            
            color = get_color(segment_id)
            shape_type = get_shape_type_from_class(label_name)
            
            # Shape Type에 따라 적절한 그리기 함수 호출
            if shape_type == "outline_only":
                draw_outline_only(vis_img, mask, color)
            elif shape_type == "horizontal_plane":
                draw_outline_only(vis_img, mask, color)
            elif shape_type == "wall":
                # 벽은 다른 모든 객체 정보가 필요
                draw_wall_structure(vis_img, bbox_2d, mask, enhanced_segments_info, segment_map, color)
            elif shape_type == "topographic":
                draw_topographic(vis_img, mask, depth_map, color)
            elif shape_type == "cube_perspective":
                # 큐브는 소실점 정보를 전달
                draw_cube_perspective(vis_img, bbox_2d, mask, color, vp)
            elif shape_type == "person":
                draw_person_structure(vis_img, bbox_2d, mask, color)
            elif shape_type == "cylinder_symmetric":
                draw_cylinder_symmetric(vis_img, bbox_2d, mask, color)
            else:
                draw_outline_only(vis_img, mask, color)

            # 라벨 텍스트 표시
            cy, cx = (y_min + y_max) // 2, (x_min + x_max) // 2
            cv2.putText(vis_img, label_name, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        return vis_img

# ============================================================================
# Entry Point: 프로그램 실행 시작점
# ============================================================================

def main():
    # 이미지 폴더가 없으면 생성하고 안내 메시지 출력
    if not os.path.exists(IMAGE_DIR):
        os.makedirs(IMAGE_DIR)
        print("Please create 'images' folder and add pictures.")
        return

    # 처리할 이미지 파일 목록 검색
    image_files = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.jpg")) + 
                         glob.glob(os.path.join(IMAGE_DIR, "*.png")))
    
    if not image_files:
        print("No images found.")
        return

    # 전체 시스템(모델 로드 등) 초기화
    visualizer = Panoptic3DVisualizer()
    
    idx = 0  # 현재 이미지 인덱스
    while True:
        path = image_files[idx]
        print(f"Processing: {os.path.basename(path)}")
        
        t0 = time.time()  # 시작 시간
        
        # 추론 실행 (Segmentation + Depth)
        img_rgb, seg, info, depth = visualizer.infer(path)
        
        # 시각화 실행 (Primitive Drawing + VP)
        res_bgr = visualizer.visualize(img_rgb, seg, info, depth)
        
        dt = time.time() - t0  # 소요 시간
        
        # FPS(초당 프레임) 표시
        cv2.putText(res_bgr, f"FPS: {1/dt:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 결과 화면 출력
        cv2.imshow("Panoptic 3D Primitive Visualization", res_bgr)
        
        # 사용자 입력 대기 (Q: 종료, A/D: 이미지 이동)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'): break
        elif key == ord('a') or key == 81: idx = (idx - 1) % len(image_files)
        elif key == ord('d') or key == 83: idx = (idx + 1) % len(image_files)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()