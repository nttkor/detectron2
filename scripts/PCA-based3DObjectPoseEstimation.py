"""
================================================================================
Project: OneFormer PCA-based 3D Object Pose Estimation
Author: User + Assistant Collaboration
Date: 2024.05.20 (Restored - Pre-Primitive Line Drawing version)

[프로그램 상세 개요]
이 프로그램은 2D 이미지를 입력받아 Segmentation Mask와 Depth 정보를 기반으로
객체의 3D 위치와 방향을 **PCA(주성분 분석)**를 사용하여 추정하고,
이를 2D 이미지 위에 3D 큐브 형태로 투영하여 객체의 자세(Pose)를 시각화합니다.

[핵심 로직]
1. AI Model Pipeline: Segmentation (OneFormer) + Depth Estimation (MiDaS).
2. Back-Projection: Depth Map과 카메라 파라미터를 사용해 2D 픽셀을 3D 포인트 클라우드로 변환.
3. PCA (Principal Component Analysis): 3D 포인트 클라우드에 PCA를 적용하여
   객체의 주축(Principal Axis, 즉 방향)과 크기를 추정.
4. Projection: 추정된 3D 큐브(Bounding Box)를 다시 2D 이미지 평면에 투영.

[사용 라이브러리]
- PyTorch & Transformers: 딥러닝 모델 로드 및 추론
- OpenCV: 이미지 처리, 그리기, 화면 출력
- NumPy: 행렬 연산
================================================================================
"""

import os
import glob
import torch
import cv2
import time
import numpy as np
from PIL import Image
from transformers import (
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

# 카메라 파라미터 (일반적인 웹캠/모바일폰 가정, 사용 환경에 맞게 조정 필요)
# 깊이 추정 모델은 실제 스케일이 아닌 상대적인 깊이를 제공하므로,
# 이 파라미터는 투영 시 상대적인 원근감 부여에만 사용됨.
FOCAL_LENGTH = 1000.0
CX = 320.0
CY = 240.0

# ============================================================================
# 3D 기하학 및 PCA 관련 유틸리티 함수
# ============================================================================

def get_camera_matrix(W, H, f=FOCAL_LENGTH):
    """
    내부 카메라 파라미터 행렬(K)을 생성합니다.
    """
    cx = W / 2.0  # 이미지 중심을 cx, cy로 설정
    cy = H / 2.0
    return np.array([
        [f, 0, cx],
        [0, f, cy],
        [0, 0, 1]
    ])

def back_project_points(depth_map, mask, K):
    """
    Depth Map과 Segmentation Mask를 사용하여 3D 포인트 클라우드를 생성합니다.
    """
    H, W = depth_map.shape
    
    # 마스크 내의 픽셀 좌표 (u, v)
    v_indices, u_indices = np.where(mask)
    
    if len(v_indices) == 0:
        return None

    # 해당 픽셀들의 깊이 값 (Z)
    Z = depth_map[v_indices, u_indices]
    
    # 픽셀 좌표를 동차 좌표로 변환 (u, v, 1)
    uv_ones = np.stack([u_indices, v_indices, np.ones_like(u_indices)], axis=1).T

    # (u, v, 1) = K * [X/Z, Y/Z, 1]. 즉, Z * K_inv * (u, v, 1) = (X, Y, Z)
    K_inv = np.linalg.inv(K)
    
    # 3D 공간 상의 (X/Z, Y/Z, 1) 좌표 계산
    XYZ_norm = K_inv @ uv_ones
    
    # 실제 3D 좌표 (X, Y, Z) 계산
    # Z 값은 (1, N) 행렬이므로, 각 행에 Z를 곱하기 위해 reshape 후 element-wise 곱셈
    X = XYZ_norm[0, :] * Z
    Y = XYZ_norm[1, :] * Z
    
    # 3D 포인트 클라우드 (N, 3) 형태로 반환
    points_3d = np.stack([X, Y, Z], axis=1)
    
    # 깊이가 너무 멀거나 0인 포인트는 제거 (노이즈 필터링)
    valid_points = points_3d[(points_3d[:, 2] > 0.01) & (points_3d[:, 2] < 100.0)]
    
    return valid_points

def compute_pca_orientation(points_3d):
    """
    3D 포인트 클라우드에 PCA를 적용하여 객체의 중심, 방향, 크기를 추정합니다.
    """
    if points_3d.shape[0] < 10:
        # 포인트 수가 너무 적으면 처리 불가
        return None, None, None

    # 1. 중심점 계산 (Centroid)
    centroid = np.mean(points_3d, axis=0)
    
    # 2. 공분산 행렬 계산
    centered_points = points_3d - centroid
    cov_matrix = np.cov(centered_points, rowvar=False)
    
    # 3. 고유값(Eigenvalues)과 고유벡터(Eigenvectors) 계산
    # 고유벡터는 객체의 주축(방향)을 나타냄
    # 고유값은 각 축을 따라 분산된 정도, 즉 크기에 비례함
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # 고유값에 따라 정렬 (가장 큰 고유값 = 주 방향)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    rotation_matrix = eigenvectors[:, sorted_indices]
    
    # 크기 추정 (고유값의 제곱근에 비례)
    # 3D 바운딩 박스의 절반 크기 (반경)
    size_factors = 2.0 * np.sqrt(eigenvalues) 
    
    # (중심점, 회전 행렬, 큐브 크기) 반환
    return centroid, rotation_matrix, size_factors

def project_mesh_to_image(centroid, rotation_matrix, size_factors, K):
    """
    추정된 3D 큐브 메쉬를 2D 이미지 평면에 투영합니다.
    """
    # 큐브의 상대 좌표 (8개의 꼭짓점)
    half_size = size_factors / 2.0
    cube_vertices_local = np.array([
        [-half_size[0], -half_size[1], -half_size[2]],
        [ half_size[0], -half_size[1], -half_size[2]],
        [ half_size[0],  half_size[1], -half_size[2]],
        [-half_size[0],  half_size[1], -half_size[2]],
        [-half_size[0], -half_size[1],  half_size[2]],
        [ half_size[0], -half_size[1],  half_size[2]],
        [ half_size[0],  half_size[1],  half_size[2]],
        [-half_size[0],  half_size[1],  half_size[2]]
    ])
    
    # 3D 월드 좌표로 변환 (회전 + 이동)
    cube_vertices_world = (rotation_matrix @ cube_vertices_local.T).T + centroid
    
    # 2D 평면으로 투영
    # P_img = K * P_world
    projected_homogeneous = K @ cube_vertices_world.T
    
    # 동차 좌표를 일반 좌표로 변환 (W/Z, H/Z)
    projected_2d = projected_homogeneous[:2, :] / projected_homogeneous[2, :]
    
    return projected_2d.T.astype(int) # (N, 2) 형식의 2D 픽셀 좌표

# 큐브의 12개 모서리 연결 인덱스
CUBE_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # Front face
    (4, 5), (5, 6), (6, 7), (7, 4),  # Back face
    (0, 4), (1, 5), (2, 6), (3, 7)   # Connecting edges
]

# ============================================================================
# Main Class: 추론 및 시각화 통합 관리자
# ============================================================================

class PanopticPoseEstimator:
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
        
        return np.array(image), segment_map, segments_info, depth_map, W, H

    def visualize(self, image, segment_map, segments_info, depth_map, W, H):
        """
        추론 결과를 시각화하고 PCA 기반 3D 포즈 추정을 수행하는 메인 함수
        """
        vis_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # OpenCV 출력을 위해 BGR로 변환
        K = get_camera_matrix(W, H) # 카메라 행렬 생성
        
        id2label = self.model.config.id2label

        # 1. 객체별 PCA 및 투영 수행
        for info in segments_info:
            segment_id = info['id']
            label_name = id2label[info['label_id']]
            
            # 사물(thing) 객체만 처리 (배경/stuff 제외)
            if not info['is_thing']:
                continue

            mask = (segment_map == segment_id)
            if not np.any(mask): continue
            
            # 2. 3D 포인트 클라우드 생성
            points_3d = back_project_points(depth_map, mask, K)
            if points_3d is None or points_3d.shape[0] < 100:
                continue

            # 3. PCA를 이용해 3D 자세 추정
            centroid, rotation_matrix, size_factors = compute_pca_orientation(points_3d)
            if centroid is None:
                continue
            
            # 4. 3D 큐브를 2D 이미지에 투영
            projected_2d_vertices = project_mesh_to_image(centroid, rotation_matrix, size_factors, K)
            
            # 시각화 색상 설정
            hue = int((segment_id * 137.5) % 180)
            hsv = np.uint8([[[hue, 255, 255]]])
            color = tuple(map(int, cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]))
            
            # 5. 투영된 큐브 그리기
            for i, j in CUBE_EDGES:
                pt1 = tuple(projected_2d_vertices[i])
                pt2 = tuple(projected_2d_vertices[j])
                
                # 큐브의 깊이 방향 선은 얇게, 앞면은 두껍게
                thickness = 2 if i in [0, 1, 2, 3] and j in [1, 2, 3, 0] else 1
                
                # 화면 경계를 벗어난 점은 그리지 않음
                if (0 <= pt1[0] < W and 0 <= pt1[1] < H and
                    0 <= pt2[0] < W and 0 <= pt2[1] < H):
                    cv2.line(vis_img, pt1, pt2, color, thickness)
            
            # 객체 라벨 및 중심점 표시
            cv2.putText(vis_img, label_name, (pt1[0], pt1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            # cv2.circle(vis_img, tuple(projected_2d_vertices[0]), 5, (255, 0, 0), -1) # 원점 표시

        return vis_img

# ============================================================================
# Entry Point: 프로그램 실행 시작점
# ============================================================================

def main():
    if not os.path.exists(IMAGE_DIR):
        os.makedirs(IMAGE_DIR)
        print("Please create 'images' folder and add pictures.")
        return

    image_files = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.jpg")) + 
                         glob.glob(os.path.join(IMAGE_DIR, "*.png")))
    
    if not image_files:
        print("No images found.")
        return

    estimator = PanopticPoseEstimator()
    
    idx = 0
    while True:
        path = image_files[idx]
        print(f"Processing: {os.path.basename(path)}")
        
        t0 = time.time()
        
        # 추론 실행 (Segmentation + Depth)
        img_rgb, seg, info, depth, W, H = estimator.infer(path)
        
        # 시각화 실행 (PCA 기반 3D 투영)
        res_bgr = estimator.visualize(img_rgb, seg, info, depth, W, H)
        
        dt = time.time() - t0
        
        # FPS 표시
        cv2.putText(res_bgr, f"FPS: {1/dt:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 결과 화면 출력
        cv2.imshow("PCA-based 3D Pose Estimation", res_bgr)
        
        # 사용자 입력 대기 (Q: 종료, A/D: 이미지 이동)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'): break
        elif key == ord('a') or key == 81: idx = (idx - 1) % len(image_files)
        elif key == ord('d') or key == 83: idx = (idx + 1) % len(image_files)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()