"""
================================================================================
Project: OneFormer Panoptic 3D Object Pose Estimation & Replacement
Author: User + Assistant Collaboration
Date: 2024.05.20

[프로그램 상세 개요]
이 프로그램은 2D 이미지 한 장에서 객체를 인식하고, 그 객체의 3D 공간상 위치와 자세(Pose)를 추정하여
가상의 3D 메쉬(Mesh)로 대체(Replacement)하여 시각화하는 증강현실(AR) 시뮬레이터입니다.

[참조 링크]
1. AI Models:
   - OneFormer (Segmentation): https://huggingface.co/shi-labs/oneformer_ade20k_swin_large
   - MiDaS (Depth): https://huggingface.co/Intel/dpt-hybrid-midas
2. Dataset:
   - ADE20K: https://groups.csail.mit.edu/vision/datasets/ADE20K/
3. Libraries:
   - Hugging Face Transformers: https://huggingface.co/docs/transformers/index
   - PyTorch: https://pytorch.org/docs/stable/index.html
   - OpenCV: https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html
   - NumPy: https://numpy.org/doc/stable/
4. Algorithms:
   - Pinhole Camera Model: https://en.wikipedia.org/wiki/Pinhole_camera_model
   - PCA (Principal Component Analysis): https://en.wikipedia.org/wiki/Principal_component_analysis

[핵심 기술 및 알고리즘]
1. AI Model Pipeline:
   - Panoptic Segmentation (OneFormer): 이미지에서 '무엇이(Class)' '어디에(Mask)' 있는지 픽셀 단위로 식별합니다.
     객체(Thing)와 배경(Stuff)을 모두 인식합니다.
   - Monocular Depth Estimation (MiDaS): 단안 이미지에서 각 픽셀의 상대적인 '깊이(Depth)'를 추론합니다.

2. 3D Geometry Processing (핵심 로직):
   - Pinhole Camera Model (핀홀 카메라 모델):
     2D 이미지 픽셀(u, v)과 예측된 깊이(z)를 결합하여 3D 공간 좌표(x, y, z)로 역투영(Back-projection)합니다.
     이때, 일반적인 카메라 내부 파라미터(Intrinsic Matrix K)를 가정하여 사용합니다.
   
   - PCA (Principal Component Analysis, 주성분 분석):
     역투영된 3D 점군(Point Cloud)의 분산(Variance)을 분석합니다.
     - 고유벡터(Eigenvectors): 점들이 가장 길게 분포된 축을 찾아 객체의 **회전(Rotation)** 정보로 사용합니다.
     - 고유값(Eigenvalues): 데이터의 퍼짐 정도를 분석하여 객체의 **크기(Scale)** 정보로 사용합니다.
     - 평균(Mean): 점들의 중심을 계산하여 객체의 **위치(Translation)** 정보로 사용합니다.

3. Mesh Instancing & Rendering:
   - Asset Library: 외부 3D 파일 없이 코드로 정의된 기본 도형(Cube, Cylinder, Plane)을 사용합니다.
   - Transformation: 위에서 구한 T(위치), R(회전), S(크기) 행렬을 기본 도형에 적용합니다.
   - Projection: 변형된 3D 메쉬를 다시 2D 화면으로 투영하여 그립니다.

[사용 라이브러리]
- PyTorch & Transformers: 딥러닝 모델 로드 및 추론
- OpenCV: 이미지 처리, 그리기, 화면 출력
- NumPy: 고속 행렬 연산 (선형대수)
================================================================================
"""

import os       # 운영체제 기능 (경로, 파일 시스템)
import glob     # 파일 패턴 매칭 (이미지 목록 검색)
import torch    # 딥러닝 프레임워크 (Tensor 연산, GPU 가속)
import cv2      # Computer Vision 라이브러리 (이미지 I/O, 그리기)
import time     # 시간 측정 (FPS 계산)
import numpy as np      # 수치 해석 및 행렬 연산 핵심 라이브러리
from PIL import Image   # 이미지 파일 로딩 (Transformers 입력 호환용)
from transformers import (  # Hugging Face의 사전 학습된 모델 모듈
    OneFormerProcessor,                 # OneFormer 전처리 (이미지 -> 텐서)
    OneFormerForUniversalSegmentation,  # OneFormer 모델 (Segmentation)
    DPTImageProcessor,                  # DPT/MiDaS 전처리
    DPTForDepthEstimation               # DPT 모델 (Depth Estimation)
)

# ============================================================================
# 환경 설정 및 상수 정의
# ============================================================================
IMAGE_DIR = "./images"  # 분석할 이미지가 위치한 디렉토리
SEGMENTATION_MODEL = "shi-labs/oneformer_ade20k_swin_large"  # 사용할 Segmentation 모델 ID (ADE20K 데이터셋: 150 클래스)
DEPTH_MODEL = "Intel/dpt-hybrid-midas"  # 사용할 Depth 모델 ID (MiDaS 기반, 범용성 우수)

# 가상 카메라 파라미터 설정
# 실제 카메라의 정보를 모르므로, 일반적인 스마트폰/웹캠의 화각(약 50~60도)을 가정합니다.
# Scale 1.0은 대략적으로 이미지 너비와 초점거리가 같다는 의미입니다.
FOCAL_LENGTH_SCALE = 1.0  

# ============================================================================
# 3D Mesh Generator (Virtual Asset Library)
# 설명: .obj 파일 로딩 대신, 정점(Vertex)과 간선(Edge) 데이터를 코드로 생성합니다.
# ============================================================================

class MeshManager:
    """
    표준 3D 모델 데이터(Vertices, Edges)를 관리하고 제공하는 클래스.
    모든 모델은 중심이 (0,0,0)이고 크기가 1인 단위(Unit) 크기로 정의됩니다.
    """
    def __init__(self):
        self.meshes = {}            # 생성된 메쉬를 저장할 딕셔너리
        self._init_primitives()     # 초기화 시 기본 도형 생성 함수 호출

    def _init_primitives(self):
        """기본적인 3D 기하학 도형(Primitive) 데이터를 생성합니다."""
        
        # 1. Unit Cube (단위 정육면체)
        # 용도: 의자, 책상, 건물, 자동차 등 박스 형태의 객체
        r = 0.5  # 중심에서 면까지의 거리 (지름이 1이 됨)
        cube_verts = np.array([
            [-r, -r, -r], [r, -r, -r], [r, r, -r], [-r, r, -r],  # 앞면 (z = -0.5) 점 4개
            [-r, -r, r], [r, -r, r], [r, r, r], [-r, r, r]       # 뒷면 (z = 0.5) 점 4개
        ])
        # 와이어프레임을 그리기 위한 점들의 연결 정보 (인덱스 쌍)
        self.meshes['cube'] = {
            'verts': cube_verts,
            'edges': [
                [0,1], [1,2], [2,3], [3,0],  # 앞면 사각형
                [4,5], [5,6], [6,7], [7,4],  # 뒷면 사각형
                [0,4], [1,5], [2,6], [3,7]   # 앞뒤를 잇는 4개의 기둥
            ]
        }

        # 2. Unit Cylinder (단위 원기둥)
        # 용도: 사람, 나무, 기둥, 가로등 등 세로로 긴 원통형 객체
        cyl_verts = []
        segments = 8  # 원을 8각형으로 근사하여 표현 (성능 최적화)
        for y in [-0.5, 0.5]:  # 아래 원(y=-0.5)과 위 원(y=0.5)
            for i in range(segments):
                theta = 2 * np.pi * i / segments  # 각도 계산 (라디안)
                # 원의 방정식: x = r*cos(theta), z = r*sin(theta) (y축이 높이 방향)
                cyl_verts.append([0.5 * np.cos(theta), y, 0.5 * np.sin(theta)])
        
        cyl_edges = []
        for i in range(segments):
            curr = i
            next_i = (i + 1) % segments  # 마지막 점은 첫 번째 점과 연결 (루프)
            cyl_edges.append([curr, next_i])                       # 아랫면 원
            cyl_edges.append([curr + segments, next_i + segments]) # 윗면 원
            cyl_edges.append([curr, curr + segments])              # 위아래를 잇는 수직선
            
        self.meshes['cylinder'] = {
            'verts': np.array(cyl_verts),
            'edges': cyl_edges
        }

        # 3. Flat Plane (평면)
        # 용도: 바닥, 천장, 벽, 도로 등 넓은 면
        plane_verts = np.array([
            [-0.5, 0, -0.5], [0.5, 0, -0.5], [0.5, 0, 0.5], [-0.5, 0, 0.5] # XZ 평면상에 누워있는 사각형
        ])
        self.meshes['plane'] = {
            'verts': plane_verts,
            'edges': [[0,1], [1,2], [2,3], [3,0], [0,2], [1,3]]  # 테두리 + X자 대각선(면임을 강조)
        }

    def get_mesh_by_class(self, class_name):
        """
        AI가 인식한 클래스 이름(문자열)을 분석하여 가장 적절한 3D 메쉬를 반환합니다.
        """
        cn = class_name.lower()
        
        # 원통형 객체 매핑
        if any(x in cn for x in ["person", "pole", "tree", "lamp", "column"]):
            return self.meshes['cylinder']
        # 바닥형/평면 객체 매핑
        elif any(x in cn for x in ["floor", "ceiling", "road", "sidewalk", "ground", "earth", "grass"]):
            return self.meshes['plane']
        # 벽 (일단 평면 반환, 추후 수직 세우기 필요할 수 있음)
        elif "wall" in cn:
            return self.meshes['plane'] 
        # 그 외 대부분의 객체(가구, 사물, 건물)는 큐브(Bounding Box) 형태로 표현
        else:
            return self.meshes['cube']

# ============================================================================
# 3D Geometry Utils (핵심 수학/알고리즘 로직)
# ============================================================================

def back_project_points(u, v, z, K_inv):
    """
    [Back-Projection] 2D 이미지 좌표를 3D 카메라 좌표계로 변환합니다.
    
    Args:
        u, v (array): 이미지 픽셀 좌표 (x, y)
        z (array): 해당 픽셀의 깊이 값 (Depth)
        K_inv (matrix): 카메라 내부 파라미터(Intrinsic)의 역행렬
    
    Returns:
        xyz (Nx3 array): 복원된 3D 점군 (Point Cloud)
    
    원리: P_cam = K_inv * [u, v, 1] * depth
    """
    # 1. 동차 좌표계(Homogeneous Coordinates) 생성: [u, v] -> [u, v, 1]
    uv_ones = np.vstack((u, v, np.ones_like(u)))
    
    # 2. Normalized Image Plane으로 투영 (카메라 매트릭스 역변환)
    xy_norm = K_inv @ uv_ones
    
    # 3. 깊이(z)를 곱하여 실제 3D 스케일로 확장
    xyz = xy_norm * z
    
    return xyz.T  # (N, 3) 형태로 전치하여 반환

def compute_pca_orientation(points_3d):
    """
    [Pose Estimation with PCA] 3D 점들의 분포를 분석하여 객체의 자세를 추정합니다.
    
    원리:
    데이터의 공분산 행렬(Covariance Matrix)을 고유값 분해(Eigendecomposition)하면,
    데이터가 가장 넓게 퍼져있는 축(주성분)들을 찾을 수 있습니다.
    이 축들을 객체의 로컬 좌표축(회전)으로 간주합니다.
    
    Returns:
        center (vec3): 위치 (Translation)
        rotation_matrix (mat3x3): 회전 (Rotation)
        scale (vec3): 크기 (Scale)
    """
    # 점이 너무 적으면 통계적 의미가 없으므로 분석 중단
    if len(points_3d) < 10: return None, None, None

    # 1. 중심점 (Translation) 계산: 모든 점의 평균 위치
    center = np.mean(points_3d, axis=0)
    
    # 2. 중심화 (Centering): 데이터의 중심을 원점(0,0,0)으로 이동
    centered_points = points_3d - center

    # 3. PCA 수행: 공분산 행렬 계산 -> 고유값/고유벡터 산출
    # covariance_matrix: 데이터가 각 축 방향으로 얼마나 함께 변하는지 나타냄
    covariance_matrix = np.cov(centered_points, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # 4. 정렬: 고유값이 큰 순서대로 (가장 긴 축 -> 가장 짧은 축)
    # eigenvectors의 각 열(column)이 주축 벡터가 됨
    sort_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sort_indices]
    eigenvectors = eigenvectors[:, sort_indices]

    # 5. 회전 행렬 (Rotation) 확정
    # PCA로 찾은 주축들을 객체의 회전 행렬로 사용
    rotation_matrix = eigenvectors

    # 6. 크기 (Scale) 추정 (Bounding Box Fitting)
    # 점들을 찾은 로컬 좌표계(주축)로 변환(Projection)하여 AABB(Axis Aligned Bounding Box) 계산
    local_points = centered_points @ rotation_matrix
    min_bound = np.min(local_points, axis=0)
    max_bound = np.max(local_points, axis=0)
    scale = (max_bound - min_bound)  # 각 축 방향의 길이
    
    # 크기가 너무 작아져서 메쉬가 사라지는 것을 방지 (최소 두께 보장)
    scale = np.maximum(scale, 0.1) 

    return center, rotation_matrix, scale

def project_mesh_to_image(verts, edges, T, R, S, K):
    """
    [Forward Projection] 3D 메쉬를 위치시키고 다시 2D 이미지로 투영합니다.
    
    Pipeline:
    Model Space (Unit Mesh) -> [Scale, Rotation, Translation] -> World Space -> [Camera Matrix] -> Image Plane
    """
    # 1. Scale Transform (크기 변환)
    verts_scaled = verts * S
    
    # 2. Rigid Body Transform (회전 및 이동)
    # 공식: P_world = R * P_local + T
    verts_world = (R @ verts_scaled.T).T + T

    # 3. Projection to 2D (카메라 투영)
    # 공식: P_cam = K * P_world
    verts_cam = (K @ verts_world.T).T
    
    # 4. Clipping (클리핑)
    # 카메라 뒤쪽(z < 0)에 있는 점들은 그리면 안 됨. (0.1은 Near Plane 역할)
    valid_mask = verts_cam[:, 2] > 0.1
    
    # 5. Perspective Division (원근 나눗셈)
    # 동차 좌표계에서 유클리드 좌표계로 변환: (x, y, z) -> (x/z, y/z)
    # 멀리 있는 물체는 작게, 가까이 있는 물체는 크게 만드는 핵심 연산
    verts_2d = verts_cam[:, :2] / verts_cam[:, 2:3]
    
    return verts_2d, edges, valid_mask

# ============================================================================
# Main Class: 추론 및 시각화 통합 관리자
# ============================================================================

class Panoptic3DReplacer:
    def __init__(self):
        """모델 로딩 및 하드웨어 설정"""
        print(f"Loading Segmentation Model: {SEGMENTATION_MODEL}...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu" # GPU 가속 확인
        
        # Segmentation Model 로드
        self.seg_processor = OneFormerProcessor.from_pretrained(SEGMENTATION_MODEL)
        self.seg_model = OneFormerForUniversalSegmentation.from_pretrained(SEGMENTATION_MODEL).to(self.device)
        
        print(f"Loading Depth Model: {DEPTH_MODEL}...")
        # Depth Model 로드
        self.depth_processor = DPTImageProcessor.from_pretrained(DEPTH_MODEL)
        self.depth_model = DPTForDepthEstimation.from_pretrained(DEPTH_MODEL).to(self.device)
        
        self.mesh_manager = MeshManager() # 3D 에셋 관리자 초기화
        print("System Ready. Models loaded on:", self.device)

    def get_color(self, idx):
        """객체 ID별로 고유한 색상 생성 (시각적 구분을 위함)"""
        hue = int((idx * 137.5) % 180)       # Golden Angle을 이용해 색상(Hue)을 최대한 멀리 떨어뜨림
        hsv = np.uint8([[[hue, 255, 255]]])  # 채도(S), 명도(V)는 최대로 설정
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0] # OpenCV 사용을 위해 BGR로 변환
        return tuple(map(int, bgr))

    def infer(self, image_path):
        """
        이미지 파일 하나에 대해 AI 추론(Segmentation + Depth)을 수행하는 함수
        """
        image = Image.open(image_path).convert("RGB")
        W, H = image.size
        
        # --- 1. Panoptic Segmentation (객체 인식) ---
        inputs = self.seg_processor(images=image, task_inputs=["panoptic"], return_tensors="pt").to(self.device)
        with torch.no_grad(): # 추론 모드 (Gradient 계산 생략)
            outputs = self.seg_model(**inputs)
        # 모델 출력을 이미지 크기에 맞게 후처리 (Segmentation Map 생성)
        panoptic_res = self.seg_processor.post_process_panoptic_segmentation(outputs, target_sizes=[(H, W)])[0]
        seg_map = panoptic_res["segmentation_map"].cpu().numpy() # 픽셀별 ID 맵
        seg_info = panoptic_res["segments_info"]                 # 객체 메타데이터 리스트

        # --- 2. Depth Estimation (깊이 추정) ---
        d_inputs = self.depth_processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            d_out = self.depth_model(**d_inputs)
            d_pred = d_out.predicted_depth
            
        # Depth 맵을 원본 해상도로 업샘플링 (Bicubic 보간법 사용)
        d_pred = torch.nn.functional.interpolate(
            d_pred.unsqueeze(1), size=(H, W), mode="bicubic", align_corners=False
        ).squeeze().cpu().numpy()
        
        # --- 3. Depth Normalization (단위 변환) ---
        # 모델이 예측한 Depth는 상대적인 값이므로, 이를 가상의 미터(Meter) 단위로 변환
        # 가정: 가장 가까운 곳 0.5m, 가장 먼 곳 10.0m
        depth_min, depth_max = d_pred.min(), d_pred.max()
        d_metric = 0.5 + (d_pred - depth_min) / (depth_max - depth_min) * 10.0
        
        return np.array(image), seg_map, seg_info, d_metric

    def visualize_replacement(self, image, seg_map, seg_info, depth_map):
        """
        추론 결과(Mask, Depth)를 이용하여 3D 메쉬를 배치하고 시각화하는 메인 함수
        """
        vis_img = image.copy()
        # 배경을 어둡게 처리하여 생성된 3D 메쉬가 더 잘 보이도록 함
        vis_img = cv2.addWeighted(vis_img, 0.3, np.zeros_like(vis_img), 0.7, 0)
        
        H, W = image.shape[:2]
        
        # --- Camera Intrinsic Matrix (K) 구성 ---
        # 3D 공간의 점을 2D 이미지로 투영하기 위한 카메라 내부 파라미터 행렬
        focal_length = W * FOCAL_LENGTH_SCALE
        cx, cy = W / 2, H / 2  # 이미지 중심을 광학 중심(Principal Point)으로 가정
        K = np.array([[focal_length, 0, cx], [0, focal_length, cy], [0, 0, 1]])
        K_inv = np.linalg.inv(K) # 역투영을 위해 미리 역행렬 계산

        # 클래스 ID를 이름으로 변환하기 위한 매핑
        id2label = self.seg_model.config.id2label

        # 감지된 모든 객체에 대해 루프 수행
        for info in seg_info:
            seg_id = info['id']              # 객체 고유 ID
            label_id = info['label_id']      # 클래스 ID
            label_name = id2label[label_id]  # 클래스 이름 (예: 'chair')
            
            # 현재 객체의 마스크 추출
            mask = (seg_map == seg_id)
            if not np.any(mask): continue

            # --- Step 1: 3D Point Cloud Sampling ---
            # 마스크 영역의 픽셀 좌표(u, v)와 깊이(z) 추출
            y_idxs, x_idxs = np.where(mask)
            
            # 모든 픽셀을 다 쓰면 PCA 연산이 느려지므로, 최대 1000개만 랜덤 샘플링 (통계적 근사)
            if len(y_idxs) > 1000:
                choice = np.random.choice(len(y_idxs), 1000, replace=False)
                y_idxs = y_idxs[choice]
                x_idxs = x_idxs[choice]
            
            z_vals = depth_map[y_idxs, x_idxs]
            
            # --- Step 2: Back-Projection (2D -> 3D) ---
            # 선택된 픽셀들을 3D 공간상의 점들로 변환
            points_3d = back_project_points(x_idxs, y_idxs, z_vals, K_inv)
            
            # --- Step 3: Pose Estimation (PCA) ---
            # 3D 점들의 분포를 분석하여 객체의 중심, 회전, 크기 계산
            center, rotation, scale = compute_pca_orientation(points_3d)
            if center is None: continue

            # 클래스 이름에 맞는 3D 메쉬(Cube, Cylinder 등) 가져오기
            mesh_data = self.mesh_manager.get_mesh_by_class(label_name)
            mesh_verts = mesh_data['verts']
            mesh_edges = mesh_data['edges']

            # --- 예외 처리: 배경(Stuff) 객체 ---
            # 벽이나 바닥 같은 배경 요소는 회전이 꼬이면 시각적으로 이상하므로 회전 초기화
            is_stuff = info['isthing'] == False
            if is_stuff:
                rotation = np.eye(3)  # 회전 없음 (축 정렬)

            # --- Step 4: Projection & Rendering (3D -> 2D) ---
            # 위치, 회전, 크기를 적용하여 메쉬를 이미지 위에 투영
            verts_2d, edges, valid_mask = project_mesh_to_image(mesh_verts, mesh_edges, center, rotation, scale, K)

            # 색상 결정
            color = self.get_color(seg_id)
            
            # 정점(Vertices) 그리기
            for x, y in verts_2d.astype(int):
                if 0 <= x < W and 0 <= y < H:
                    cv2.circle(vis_img, (x, y), 2, color, -1)
            
            # 간선(Edges - Wireframe) 그리기
            for edge in edges:
                i, j = edge
                # 두 점 모두 카메라 앞에 있고 유효한 경우에만 그림
                if valid_mask[i] and valid_mask[j]:
                    pt1 = tuple(verts_2d[i].astype(int))
                    pt2 = tuple(verts_2d[j].astype(int))
                    # 이미지 범위 내에 있는지 확인 (간단한 클리핑)
                    if 0 <= pt1[0] < W and 0 <= pt1[1] < H and 0 <= pt2[0] < W and 0 <= pt2[1] < H:
                        cv2.line(vis_img, pt1, pt2, color, 2)
            
            # 라벨 텍스트 표시 (메쉬의 중심에)
            cx_2d, cy_2d = verts_2d.mean(axis=0).astype(int)
            if 0 <= cx_2d < W and 0 <= cy_2d < H:
                cv2.putText(vis_img, label_name, (cx_2d, cy_2d), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

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
    replacer = Panoptic3DReplacer()
    
    idx = 0  # 현재 이미지 인덱스
    while True:
        path = image_files[idx]
        print(f"Processing: {os.path.basename(path)}")
        
        t0 = time.time()  # 시작 시간
        
        # 추론 실행 (Segmentation + Depth)
        img, seg, info, depth = replacer.infer(path)
        
        # 시각화 실행 (Pose Estimation + Rendering)
        res = replacer.visualize_replacement(img, seg, info, depth)
        
        dt = time.time() - t0  # 소요 시간
        
        # OpenCV 출력을 위해 RGB -> BGR 변환
        res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
        
        # FPS(초당 프레임) 표시
        cv2.putText(res, f"FPS: {1/dt:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 결과 화면 출력
        cv2.imshow("3D Object Replacement", res)
        
        # 사용자 입력 대기 (Q: 종료, A/D: 이미지 이동)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'): break
        elif key == ord('a') or key == 81: idx = (idx - 1) % len(image_files)
        elif key == ord('d') or key == 83: idx = (idx + 1) % len(image_files)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()