"""
OneFormer Panoptic Segmentation + Depth Estimation - Interactive Visualization Tool
ADE20K 공식 Thing/Stuff 분류 (CSAILVision MIT) + 신뢰도 + Depth 정보 표시

프로그램 개요:
    이 프로그램은 OneFormer 딥러닝 모델을 사용하여 이미지의 Panoptic Segmentation을 수행하고,
    MiDaS 모델을 사용하여 Depth Estimation을 수행한 후, 두 결과를 융합하여 시각화하는 도구입니다.
    
    주요 기능:
    1. 이미지 디렉토리에서 JPG 파일을 자동으로 로드
    2. OneFormer 모델을 사용한 Panoptic Segmentation 추론 수행
    3. MiDaS 모델을 사용한 Monocular Depth Estimation 수행
    4. Thing(객체)과 Stuff(배경)를 구분하여 시각화
    5. 각 세그먼트의 클래스 이름, 신뢰도 점수, 평균 거리 정보 표시
    6. 키보드 입력(A/D 또는 화살표 키)으로 이미지 간 이동
    7. 추론 시간 및 Thing/Stuff 개수 정보 표시
    
    사용 모델:
    - Segmentation: shi-labs/oneformer_ade20k_swin_large
    - Depth: Intel/dpt-hybrid-midas (MiDaS)
    - 데이터셋: ADE20K (150개 클래스)
    - 분류: Thing(객체) / Stuff(배경) 공식 분류 사용
    
    실행 방법:
    - 스크립트 실행 시 첫 번째 이미지가 자동으로 로드됨
    - 'A' 또는 왼쪽 화살표: 이전 이미지
    - 'D' 또는 오른쪽 화살표: 다음 이미지
    - 'Q': 프로그램 종료
    
    시각화 특징:
    - Stuff 영역: 반투명 색상 오버레이
    - Thing 영역: 외곽선으로 표시
    - 텍스트 색상: Thing(노란색), Stuff(흰색)
    - 각 세그먼트 중심에 클래스 이름, 신뢰도, 평균 거리 표시
    - Depth 정보는 상대적 거리 (절대값 아님)
    
"""

import os  # 파일 시스템 경로 조작
import glob  # 파일 패턴 매칭
import math  # 수학 연산
import torch  # PyTorch 딥러닝 프레임워크
import cv2  # OpenCV 이미지 처리
import time  # 시간 측정
import numpy as np  # 수치 연산
from PIL import Image  # PIL 이미지 처리
from transformers import (  # Transformers 라이브러리
    OneFormerProcessor, 
    OneFormerForUniversalSegmentation,
    pipeline
)

# ============================================================================
# 상수 정의
# ============================================================================
IMAGE_DIR = r"D:/git/detectron2/ade20k_consistency/original_ade20k"  # 이미지 디렉토리 경로
SEGMENTATION_MODEL = "shi-labs/oneformer_ade20k_swin_large"  # Segmentation 모델
DEPTH_MODEL = "Intel/dpt-hybrid-midas"  # Depth Estimation 모델 (빠른 버전)
TARGET_HEIGHT = 800  # 시각화 목표 높이

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
        
        Args:
            image (numpy.ndarray): 입력 이미지 (BGR 형식)
        
        Returns:
            tuple: (vx, vy) 소실점 좌표
        """
        h, w = image.shape[:2]  # 이미지 크기
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 그레이스케일 변환
        
        # 1. 엣지 검출 (Canny)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)  # Canny 엣지 검출
        
        # 2. 직선 검출 (Hough Transform)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)  # 직선 검출
        
        if lines is None:  # 직선이 없으면
            return (w // 2, h // 2)  # 이미지 중심 반환

        filtered_lines = []  # 필터링된 직선 리스트
        for line in lines:  # 각 직선에 대해
            x1, y1, x2, y2 = line[0]  # 직선의 두 점
            if x1 == x2:  # 수직선이면 건너뛰기
                continue
            
            slope = (y2 - y1) / (x2 - x1)  # 기울기 계산
            angle = math.degrees(math.atan(slope))  # 각도 계산
            
            # 대각선 방향의 선들만 수집 (VP를 찾기 위한 주된 단서)
            if 15 < abs(angle) < 75:  # 15도~75도 사이의 대각선만
                filtered_lines.append((x1, y1, x2, y2, slope))  # 직선 정보 추가

        if not filtered_lines:  # 필터링된 직선이 없으면
            return (w // 2, h // 2)  # 이미지 중심 반환

        A_matrix = []  # 행렬 A
        b_vector = []  # 벡터 b
        
        for x1, y1, x2, y2, m in filtered_lines:  # 각 직선에 대해
            # Line equation: mx - y = -(y1 - m*x1)
            c = y1 - m * x1  # 상수항
            A_matrix.append([m, -1])  # 행렬 A에 추가
            b_vector.append([-c])  # 벡터 b에 추가
            
        if len(A_matrix) < 2:  # 직선이 2개 미만이면
            return (w // 2, h // 2)  # 이미지 중심 반환

        try:
            A = np.array(A_matrix)  # 행렬 A 생성
            b = np.array(b_vector)  # 벡터 b 생성
            # 최소 제곱법을 이용해 해(x, y) = 소실점을 구함
            vx, vy = np.linalg.lstsq(A, b, rcond=None)[0]  # 최소 제곱법으로 소실점 계산
            vx, vy = int(vx), int(vy)  # 정수로 변환
            
            # 소실점이 화면 밖 너무 멀리 있으면 중심점으로 제한
            if not (-2*w < vx < 3*w and -2*h < vy < 3*h):  # 범위 밖이면
                 return (w // 2, h // 2)  # 이미지 중심 반환
                 
            return (vx, vy)  # 소실점 반환
        except:  # 예외 발생 시
            return (w // 2, h // 2)  # 이미지 중심 반환


# ============================================================================
# 유틸리티 함수
# ============================================================================

def get_color(idx):
    """
    인덱스 기반으로 고유한 색상을 생성합니다.
    
    HSV 색상 공간을 사용하여 각 인덱스마다 서로 다른 색상을 생성하며,
    137.5도 간격으로 색상을 배치하여 시각적으로 구분하기 쉽게 만듭니다.
    
    Args:
        idx (int): 색상을 생성할 인덱스
    
    Returns:
        tuple: (B, G, R) 형식의 정수 튜플 (OpenCV BGR 형식)
    """
    hue = int((idx * 137.5) % 180)  # HSV 색상 공간에서 색상값 계산
    hsv = np.uint8([[[hue, 255, 255]]])  # HSV 배열 생성
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]  # HSV를 BGR로 변환
    return tuple(map(int, bgr))  # 정수 튜플로 반환


def get_shape_type_from_class(class_name):
    """
    클래스 이름을 기반으로 3D 형태를 추정합니다.
    
    Args:
        class_name (str): 클래스 이름 (예: "bed", "wall", "floor")
    
    Returns:
        str: 형태 타입 (상세한 타입 반환)
    """
    cn = class_name.lower()  # 소문자로 변환
    
    # Outline Only (외곽선만)
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
    
    return "outline_only"  # 기본값


def calculate_segment_depth(seg_map, segment_id, depth_map):
    """
    특정 세그먼트의 평균 depth 값을 계산합니다.
    
    Args:
        seg_map (numpy.ndarray): 세그멘테이션 맵
        segment_id (int): 세그먼트 ID
        depth_map (numpy.ndarray): Depth 맵 (이미지와 동일한 크기)
    
    Returns:
        float: 세그먼트 영역의 평균 depth 값 (상대적 거리)
    """
    mask = (seg_map == segment_id)  # 세그먼트 마스크 생성
    if not np.any(mask):  # 마스크가 비어있으면
        return 0.0
    segment_depths = depth_map[mask]  # 세그먼트 영역의 depth 값들
    return float(np.mean(segment_depths))  # 평균 depth 반환


def calculate_segment_depth_range(seg_map, segment_id, depth_map):
    """
    특정 세그먼트의 depth 범위를 계산합니다.
    
    Args:
        seg_map (numpy.ndarray): 세그멘테이션 맵
        segment_id (int): 세그먼트 ID
        depth_map (numpy.ndarray): Depth 맵 (이미지와 동일한 크기)
    
    Returns:
        tuple: (min_depth, max_depth, avg_depth)
    """
    mask = (seg_map == segment_id)  # 세그먼트 마스크 생성
    if not np.any(mask):  # 마스크가 비어있으면
        return 0.0, 0.0, 0.0
    segment_depths = depth_map[mask]  # 세그먼트 영역의 depth 값들
    return float(np.min(segment_depths)), float(np.max(segment_depths)), float(np.mean(segment_depths))  # min, max, avg 반환


def detect_depth_edges_in_segment(segment_mask, depth_map, threshold_ratio=0.15):
    """
    세그먼트 내부의 depth 변화가 큰 곳(edge)을 감지합니다.
    
    Args:
        segment_mask (numpy.ndarray): 세그먼트 마스크 (boolean)
        depth_map (numpy.ndarray): Depth 맵
        threshold_ratio (float): Edge 감지 임계값 비율 (depth 범위의 일정 비율)
    
    Returns:
        tuple: (horizontal_edges, vertical_edges) - 수평/수직 edge 마스크
    """
    # 세그먼트 영역의 depth만 추출
    segment_depth = depth_map.copy()  # Depth 맵 복사
    segment_depth[~segment_mask] = 0  # 세그먼트 외부는 0으로 설정
    
    # Depth 범위 계산
    valid_depths = depth_map[segment_mask]  # 유효한 depth 값들
    if len(valid_depths) == 0:  # 유효한 depth가 없으면
        return np.zeros_like(segment_mask, dtype=bool), np.zeros_like(segment_mask, dtype=bool)  # 빈 마스크 반환
    
    depth_range = valid_depths.max() - valid_depths.min()  # Depth 범위
    threshold = depth_range * threshold_ratio  # Edge 감지 임계값
    
    # Sobel 필터로 gradient 계산
    sobel_x = cv2.Sobel(segment_depth, cv2.CV_64F, 1, 0, ksize=3)  # 수평 방향 gradient
    sobel_y = cv2.Sobel(segment_depth, cv2.CV_64F, 0, 1, ksize=3)  # 수직 방향 gradient
    
    # 절댓값으로 변환
    sobel_x = np.abs(sobel_x)  # 절댓값
    sobel_y = np.abs(sobel_y)  # 절댓값
    
    # 임계값 이상인 곳을 edge로 감지
    horizontal_edges = (sobel_x > threshold) & segment_mask  # 수평 edge (수직 선)
    vertical_edges = (sobel_y > threshold) & segment_mask  # 수직 edge (수평 선)
    
    return horizontal_edges, vertical_edges


def draw_line_in_mask(img, pt1, pt2, mask, color, thickness=2):
    """
    마스크 내부에만 선을 그립니다.
    
    Args:
        img (numpy.ndarray): 이미지
        pt1 (tuple): 시작점 (x, y)
        pt2 (tuple): 끝점 (x, y)
        mask (numpy.ndarray): 마스크 (boolean)
        color (tuple): 색상
        thickness (int): 선 두께
    """
    # 선을 따라 점들을 생성하고 마스크 내부인지 확인
    x1, y1 = pt1  # 시작점
    x2, y2 = pt2  # 끝점
    
    # Bresenham 알고리즘으로 선을 따라 점 생성
    dx = abs(x2 - x1)  # X 차이
    dy = abs(y2 - y1)  # Y 차이
    sx = 1 if x1 < x2 else -1  # X 방향
    sy = 1 if y1 < y2 else -1  # Y 방향
    err = dx - dy  # 오차
    
    x, y = x1, y1  # 현재 위치
    points = []  # 마스크 내부 점 리스트
    
    while True:  # 선을 따라 이동
        if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:  # 범위 내이면
            if mask[y, x]:  # 마스크 내부이면
                points.append((x, y))  # 점 추가
            else:  # 마스크 외부이면
                if len(points) > 1:  # 이전 점들이 있으면
                    # 이전 점들로 선 그리기
                    for i in range(len(points) - 1):  # 각 점 쌍에 대해
                        cv2.line(img, points[i], points[i+1], color, thickness)  # 선 그리기
                    points = []  # 리스트 초기화
        
        if x == x2 and y == y2:  # 끝점에 도달하면
            break  # 종료
        
        e2 = 2 * err  # 오차 2배
        if e2 > -dy:  # Y 방향 이동
            err -= dy  # 오차 조정
            x += sx  # X 이동
        if e2 < dx:  # X 방향 이동
            err += dx  # 오차 조정
            y += sy  # Y 이동
    
    # 마지막 점들로 선 그리기
    if len(points) > 1:  # 점들이 있으면
        for i in range(len(points) - 1):  # 각 점 쌍에 대해
            cv2.line(img, points[i], points[i+1], color, thickness)  # 선 그리기


def draw_cube_structure(img, bbox_2d, depth_map, segment_mask, color, thickness=2):
    """
    큐브 형태의 3D 구조를 그립니다 (마스크 내부에만).
    
    Args:
        img (numpy.ndarray): 이미지 (BGR 형식)
        bbox_2d (tuple): 2D 바운딩 박스 (x_min, y_min, x_max, y_max)
        depth_map (numpy.ndarray): Depth 맵
        segment_mask (numpy.ndarray): 세그먼트 마스크
        color (tuple): 색상 (B, G, R)
        thickness (int): 선 두께
    """
    x_min, y_min, x_max, y_max = bbox_2d  # 2D 바운딩 박스 좌표
    
    # 세그먼트 영역의 depth 범위 계산
    valid_depths = depth_map[segment_mask]  # 유효한 depth 값들
    if len(valid_depths) == 0:  # 유효한 depth가 없으면
        return
    min_depth = valid_depths.min()  # 최소 depth
    max_depth = valid_depths.max()  # 최대 depth
    
    # 앞면 (가까운 면) - 마스크 내부의 경계만 사용
    # 마스크의 외곽선 찾기
    mask_uint8 = (segment_mask.astype(np.uint8) * 255)  # 마스크를 uint8로 변환
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 외곽선 찾기
    
    if len(contours) > 0:  # 외곽선이 있으면
        # 가장 큰 외곽선 사용
        largest_contour = max(contours, key=cv2.contourArea)  # 가장 큰 외곽선
        # 외곽선을 따라 선 그리기
        cv2.drawContours(img, [largest_contour], -1, color, thickness)  # 외곽선 그리기
        
        # Depth edge를 따라 내부 선 그리기
        horizontal_edges, vertical_edges = detect_depth_edges_in_segment(segment_mask, depth_map, threshold_ratio=0.15)  # Edge 감지
        
        # 수평 선 그리기 (마스크 내부에만)
        y_coords, x_coords = np.where(vertical_edges)  # 수직 edge 좌표
        if len(y_coords) > 5:  # 충분한 점이 있으면
            unique_y = np.unique(y_coords)  # 고유한 y 좌표
            for y in unique_y[::3]:  # 일부만 선택
                x_points = x_coords[y_coords == y]  # 해당 y의 x 좌표들
                if len(x_points) > 1:  # 점이 2개 이상이면
                    x_sorted = np.sort(x_points)  # 정렬
                    # 마스크 내부인지 확인하며 선 그리기
                    for i in range(len(x_sorted) - 1):  # 각 점 쌍에 대해
                        if segment_mask[y, x_sorted[i]] and segment_mask[y, x_sorted[i+1]]:  # 둘 다 마스크 내부이면
                            cv2.line(img, (x_sorted[i], y), (x_sorted[i+1], y), color, thickness)  # 선 그리기


def draw_cylinder_structure(img, bbox_2d, depth_map, segment_mask, color, thickness=2):
    """
    실린더 형태의 3D 구조를 그립니다 (마스크 내부에만).
    
    Args:
        img (numpy.ndarray): 이미지 (BGR 형식)
        bbox_2d (tuple): 2D 바운딩 박스 (x_min, y_min, x_max, y_max)
        depth_map (numpy.ndarray): Depth 맵
        segment_mask (numpy.ndarray): 세그먼트 마스크
        color (tuple): 색상 (B, G, R)
        thickness (int): 선 두께
    """
    # 마스크의 외곽선 찾기
    mask_uint8 = (segment_mask.astype(np.uint8) * 255)  # 마스크를 uint8로 변환
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 외곽선 찾기
    
    if len(contours) > 0:  # 외곽선이 있으면
        # 가장 큰 외곽선 사용
        largest_contour = max(contours, key=cv2.contourArea)  # 가장 큰 외곽선
        # 외곽선을 따라 선 그리기
        cv2.drawContours(img, [largest_contour], -1, color, thickness)  # 외곽선 그리기
        
        # 실린더 내부의 수직선 그리기 (마스크 내부에만)
        # Depth edge를 따라 수직선 그리기
        horizontal_edges, _ = detect_depth_edges_in_segment(segment_mask, depth_map, threshold_ratio=0.15)  # 수평 edge만
        
        # 수직선 그리기 (마스크 내부에만)
        y_coords, x_coords = np.where(horizontal_edges)  # 수평 edge 좌표
        if len(y_coords) > 5:  # 충분한 점이 있으면
            unique_x = np.unique(x_coords)  # 고유한 x 좌표
            for x in unique_x[::5]:  # 일부만 선택
                y_points = y_coords[x_coords == x]  # 해당 x의 y 좌표들
                if len(y_points) > 1:  # 점이 2개 이상이면
                    y_sorted = np.sort(y_points)  # 정렬
                    # 마스크 내부인지 확인하며 선 그리기
                    for i in range(len(y_sorted) - 1):  # 각 점 쌍에 대해
                        if segment_mask[y_sorted[i], x] and segment_mask[y_sorted[i+1], x]:  # 둘 다 마스크 내부이면
                            cv2.line(img, (x, y_sorted[i]), (x, y_sorted[i+1]), color, thickness)  # 선 그리기


def draw_plane_structure(img, bbox_2d, depth_map, segment_mask, color, thickness=2):
    """
    평면 형태의 3D 구조를 그립니다 (꺾인 경계를 따라 접힌 종이처럼).
    
    Args:
        img (numpy.ndarray): 이미지 (BGR 형식)
        bbox_2d (tuple): 2D 바운딩 박스 (x_min, y_min, x_max, y_max)
        depth_map (numpy.ndarray): Depth 맵
        segment_mask (numpy.ndarray): 세그먼트 마스크
        color (tuple): 색상 (B, G, R)
        thickness (int): 선 두께
    """
    # 평면의 꺾인 경계(edge)를 찾아서 접힌 종이처럼 표시
    horizontal_edges, vertical_edges = detect_depth_edges_in_segment(segment_mask, depth_map, threshold_ratio=0.15)  # Edge 감지
    
    # 꺾인 경계를 강조하여 그리기 (접힌 종이 효과)
    # 수평 선 그리기 (수직 edge를 따라 - 종이가 수직으로 접힌 부분)
    y_coords, x_coords = np.where(vertical_edges)  # 수직 edge 좌표
    if len(y_coords) > 5:  # 충분한 점이 있으면
        unique_y = np.unique(y_coords)  # 고유한 y 좌표
        for y in unique_y:  # 각 y 좌표에 대해
            x_points = x_coords[y_coords == y]  # 해당 y의 x 좌표들
            if len(x_points) > 1:  # 점이 2개 이상이면
                x_sorted = np.sort(x_points)  # 정렬
                # 연속된 선분으로 그리기
                gaps = np.diff(x_sorted)  # 간격 계산
                gap_threshold = 3  # 간격 임계값
                
                start_idx = 0  # 시작 인덱스
                for i, gap in enumerate(gaps):  # 각 간격에 대해
                    if gap > gap_threshold:  # 간격이 크면
                        # 이전까지의 선 그리기
                        if i > start_idx:  # 선분이 있으면
                            cv2.line(img, (x_sorted[start_idx], y), (x_sorted[i], y), color, thickness)  # 선 그리기
                        start_idx = i + 1  # 다음 시작점
                # 마지막 선분 그리기
                if start_idx < len(x_sorted):  # 남은 선분이 있으면
                    cv2.line(img, (x_sorted[start_idx], y), (x_sorted[-1], y), color, thickness)  # 선 그리기
    
    # 수직 선 그리기 (수평 edge를 따라 - 종이가 수평으로 접힌 부분)
    y_coords, x_coords = np.where(horizontal_edges)  # 수평 edge 좌표
    if len(y_coords) > 5:  # 충분한 점이 있으면
        unique_x = np.unique(x_coords)  # 고유한 x 좌표
        for x in unique_x:  # 각 x 좌표에 대해
            y_points = y_coords[x_coords == x]  # 해당 x의 y 좌표들
            if len(y_points) > 1:  # 점이 2개 이상이면
                y_sorted = np.sort(y_points)  # 정렬
                # 연속된 선분으로 그리기
                gaps = np.diff(y_sorted)  # 간격 계산
                gap_threshold = 3  # 간격 임계값
                
                start_idx = 0  # 시작 인덱스
                for i, gap in enumerate(gaps):  # 각 간격에 대해
                    if gap > gap_threshold:  # 간격이 크면
                        # 이전까지의 선 그리기
                        if i > start_idx:  # 선분이 있으면
                            cv2.line(img, (x, y_sorted[start_idx]), (x, y_sorted[i]), color, thickness)  # 선 그리기
                        start_idx = i + 1  # 다음 시작점
                # 마지막 선분 그리기
                if start_idx < len(y_sorted):  # 남은 선분이 있으면
                    cv2.line(img, (x, y_sorted[start_idx]), (x, y_sorted[-1]), color, thickness)  # 선 그리기


def draw_sphere_structure(img, bbox_2d, depth_map, segment_mask, color, thickness=2):
    """
    구 형태의 3D 구조를 그립니다.
    
    Args:
        img (numpy.ndarray): 이미지 (BGR 형식)
        bbox_2d (tuple): 2D 바운딩 박스 (x_min, y_min, x_max, y_max)
        depth_map (numpy.ndarray): Depth 맵
        segment_mask (numpy.ndarray): 세그먼트 마스크
        color (tuple): 색상 (B, G, R)
        thickness (int): 선 두께
    """
    x_min, y_min, x_max, y_max = bbox_2d  # 2D 바운딩 박스 좌표
    
    # 구의 중심과 반지름
    center_x = (x_min + x_max) / 2  # 중심 X
    center_y = (y_min + y_max) / 2  # 중심 Y
    radius = min((x_max - x_min) / 2, (y_max - y_min) / 2)  # 반지름
    
    # 구의 원형 윤곽선 그리기
    cv2.circle(img, (int(center_x), int(center_y)), int(radius), color, thickness)  # 외곽 원
    
    # 수평선과 수직선으로 구 형태 표현
    cv2.ellipse(img, (int(center_x), int(center_y)), (int(radius), int(radius * 0.5)), 0, 0, 360, color, thickness)  # 수평 타원
    cv2.ellipse(img, (int(center_x), int(center_y)), (int(radius * 0.5), int(radius)), 0, 0, 360, color, thickness)  # 수직 타원


def draw_contour_lines(img, segment_mask, depth_map, color, thickness=1):
    """
    Depth map을 이용해 등고선을 그립니다 (mountain, earth 등).
    
    Args:
        img (numpy.ndarray): 이미지
        segment_mask (numpy.ndarray): 세그먼트 마스크
        depth_map (numpy.ndarray): Depth 맵
        color (tuple): 색상
        thickness (int): 선 두께
    """
    # Depth 값을 구간으로 나누어 등고선 생성
    valid_depths = depth_map[segment_mask]  # 유효한 depth 값들
    if len(valid_depths) == 0:  # 유효한 depth가 없으면
        return
    
    depth_min = valid_depths.min()  # 최소 depth
    depth_max = valid_depths.max()  # 최대 depth
    depth_range = depth_max - depth_min  # Depth 범위
    
    if depth_range == 0:  # 범위가 없으면
        return
    
    # 등고선 레벨 생성 (5-10개 레벨)
    num_levels = 8  # 등고선 레벨 개수
    levels = np.linspace(depth_min, depth_max, num_levels)  # 등고선 레벨
    
    for level in levels[1:-1]:  # 첫 번째와 마지막 제외
        # 해당 depth 레벨의 경계 찾기
        level_mask = (np.abs(depth_map - level) < depth_range * 0.05) & segment_mask  # 레벨 마스크
        if np.any(level_mask):  # 마스크가 있으면
            # 등고선 그리기 (마스크 내부에만)
            y_coords, x_coords = np.where(level_mask)  # 좌표 추출
            if len(y_coords) > 10:  # 충분한 점이 있으면
                # 간단한 등고선 표현 (일부 점만 선택)
                for i in range(0, len(y_coords), max(1, len(y_coords) // 50)):  # 일부만 선택
                    y, x = y_coords[i], x_coords[i]  # 좌표
                    if segment_mask[y, x]:  # 마스크 내부이면
                        cv2.circle(img, (x, y), 1, color, thickness)  # 작은 점으로 표시


def draw_person_structure(img, bbox_2d, depth_map, segment_mask, color, thickness=2):
    """
    사람 구조를 그립니다 (수직 실린더 + 위에 구).
    
    Args:
        img (numpy.ndarray): 이미지
        bbox_2d (tuple): 2D 바운딩 박스
        depth_map (numpy.ndarray): Depth 맵
        segment_mask (numpy.ndarray): 세그먼트 마스크
        color (tuple): 색상
        thickness (int): 선 두께
    """
    x_min, y_min, x_max, y_max = bbox_2d  # 바운딩 박스
    center_x = (x_min + x_max) / 2  # 중심 X
    height = y_max - y_min  # 높이
    
    # 몸통 (하단 70%): 수직 실린더
    body_bottom = int(y_max)  # 몸통 하단
    body_top = int(y_min + height * 0.3)  # 몸통 상단
    body_center_y = (body_bottom + body_top) / 2  # 몸통 중심 Y
    body_radius = min((x_max - x_min) / 2 * 0.4, (body_bottom - body_top) / 2)  # 몸통 반지름
    
    # 실린더 외곽선 (마스크 내부에만)
    mask_uint8 = (segment_mask.astype(np.uint8) * 255)  # 마스크 변환
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 외곽선
    if len(contours) > 0:  # 외곽선이 있으면
        largest_contour = max(contours, key=cv2.contourArea)  # 가장 큰 외곽선
        # 몸통 부분만 필터링
        body_contour = []  # 몸통 외곽선
        for pt in largest_contour:  # 각 점에 대해
            x, y = pt[0][0], pt[0][1]  # 좌표
            if body_top <= y <= body_bottom and segment_mask[y, x]:  # 몸통 영역이고 마스크 내부이면
                body_contour.append([[x, y]])  # 추가
        if len(body_contour) > 0:  # 점이 있으면
            body_contour = np.array(body_contour, dtype=np.int32)  # 배열 변환
            cv2.drawContours(img, [body_contour], -1, color, thickness)  # 외곽선 그리기
    
    # 머리 (상단 30%): 구
    head_center_y = int(y_min + height * 0.15)  # 머리 중심 Y
    head_radius = min((x_max - x_min) / 2 * 0.3, height * 0.15)  # 머리 반지름
    
    # 머리 원 그리기 (마스크 내부에만)
    for angle in range(0, 360, 10):  # 각도별로
        rad = np.radians(angle)  # 라디안 변환
        x = int(center_x + head_radius * np.cos(rad))  # X 좌표
        y = int(head_center_y + head_radius * np.sin(rad))  # Y 좌표
        if 0 <= y < segment_mask.shape[0] and 0 <= x < segment_mask.shape[1]:  # 범위 내이면
            if segment_mask[y, x]:  # 마스크 내부이면
                cv2.circle(img, (x, y), 1, color, thickness)  # 점 그리기


def draw_palm_tree_structure(img, bbox_2d, depth_map, segment_mask, color, thickness=2):
    """
    야자수 구조를 그립니다 (위에 평면 잎 + 아래 실린더).
    
    Args:
        img (numpy.ndarray): 이미지
        bbox_2d (tuple): 2D 바운딩 박스
        depth_map (numpy.ndarray): Depth 맵
        segment_mask (numpy.ndarray): 세그먼트 마스크
        color (tuple): 색상
        thickness (int): 선 두께
    """
    x_min, y_min, x_max, y_max = bbox_2d  # 바운딩 박스
    center_x = (x_min + x_max) / 2  # 중심 X
    height = y_max - y_min  # 높이
    
    # 잎 부분 (상단 40%): 평면
    leaf_bottom = int(y_min + height * 0.4)  # 잎 하단
    leaf_top = int(y_min)  # 잎 상단
    
    # 잎의 외곽선 그리기 (마스크 내부에만)
    mask_uint8 = (segment_mask.astype(np.uint8) * 255)  # 마스크 변환
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 외곽선
    if len(contours) > 0:  # 외곽선이 있으면
        largest_contour = max(contours, key=cv2.contourArea)  # 가장 큰 외곽선
        # 잎 부분만 필터링
        leaf_contour = []  # 잎 외곽선
        for pt in largest_contour:  # 각 점에 대해
            x, y = pt[0][0], pt[0][1]  # 좌표
            if leaf_top <= y <= leaf_bottom and segment_mask[y, x]:  # 잎 영역이고 마스크 내부이면
                leaf_contour.append([[x, y]])  # 추가
        if len(leaf_contour) > 0:  # 점이 있으면
            leaf_contour = np.array(leaf_contour, dtype=np.int32)  # 배열 변환
            cv2.drawContours(img, [leaf_contour], -1, color, thickness)  # 외곽선 그리기
    
    # 중심선 그리기 (마스크 내부에만)
    center_y_start = leaf_bottom  # 시작 Y
    center_y_end = y_max  # 끝 Y
    for y in range(center_y_start, center_y_end, 2):  # 2픽셀 간격
        x = int(center_x)  # 중심 X
        if 0 <= y < segment_mask.shape[0] and 0 <= x < segment_mask.shape[1]:  # 범위 내이면
            if segment_mask[y, x]:  # 마스크 내부이면
                cv2.circle(img, (x, y), 1, color, thickness)  # 점 그리기
    
    # 줄기 부분 (하단 60%): 실린더
    trunk_top = leaf_bottom  # 줄기 상단
    trunk_bottom = y_max  # 줄기 하단
    trunk_radius = min((x_max - x_min) / 2 * 0.2, (trunk_bottom - trunk_top) / 2)  # 줄기 반지름
    
    # 줄기 외곽선 (마스크 내부에만)
    trunk_contour = []  # 줄기 외곽선
    for pt in largest_contour:  # 각 점에 대해
        x, y = pt[0][0], pt[0][1]  # 좌표
        if trunk_top <= y <= trunk_bottom and segment_mask[y, x]:  # 줄기 영역이고 마스크 내부이면
            trunk_contour.append([[x, y]])  # 추가
    if len(trunk_contour) > 0:  # 점이 있으면
        trunk_contour = np.array(trunk_contour, dtype=np.int32)  # 배열 변환
        cv2.drawContours(img, [trunk_contour], -1, color, thickness)  # 외곽선 그리기


def draw_horizontal_plane(img, segment_mask, color, thickness=1):
    """
    수평 평면을 그립니다 (window, floor, ceiling - 외곽선만).
    
    Args:
        img (numpy.ndarray): 이미지
        segment_mask (numpy.ndarray): 세그먼트 마스크
        color (tuple): 색상
        thickness (int): 선 두께
    """
    # 외곽선만 그리기
    mask_uint8 = (segment_mask.astype(np.uint8) * 255)  # 마스크 변환
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 외곽선
    if len(contours) > 0:  # 외곽선이 있으면
        largest_contour = max(contours, key=cv2.contourArea)  # 가장 큰 외곽선
        cv2.drawContours(img, [largest_contour], -1, color, thickness)  # 외곽선 그리기


def draw_wall_structure(img, bbox_2d, depth_map, segment_mask, all_segments_info, seg_map_resized, color, thickness=2, vanishing_point=None):
    """
    벽 구조를 그립니다 (수직 평면, floor/ceiling과 만나는 곳에서 수직선으로 분리).
    
    Args:
        img (numpy.ndarray): 이미지
        bbox_2d (tuple): 2D 바운딩 박스
        depth_map (numpy.ndarray): Depth 맵
        segment_mask (numpy.ndarray): 세그먼트 마스크
        all_segments_info (list): 모든 세그먼트 정보
        seg_map_resized (numpy.ndarray): 리사이즈된 세그멘테이션 맵
        color (tuple): 색상
        thickness (int): 선 두께
    """
    # 외곽선 그리기
    mask_uint8 = (segment_mask.astype(np.uint8) * 255)  # 마스크 변환
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 외곽선
    if len(contours) > 0:  # 외곽선이 있으면
        largest_contour = max(contours, key=cv2.contourArea)  # 가장 큰 외곽선
        cv2.drawContours(img, [largest_contour], -1, color, thickness)  # 외곽선 그리기
    
    # floor와 ceiling 찾기
    floor_segments = []  # floor 세그먼트
    ceiling_segments = []  # ceiling 세그먼트
    
    for seg_info in all_segments_info:  # 각 세그먼트에 대해
        seg_id = seg_info['id']  # 세그먼트 ID
        seg_mask = (seg_map_resized == seg_id)  # 세그먼트 마스크
        if np.any(seg_mask):  # 마스크가 있으면
            y_coords, _ = np.where(seg_mask)  # Y 좌표
            if len(y_coords) > 0:  # 좌표가 있으면
                avg_y = y_coords.mean()  # 평균 Y
                # 벽과 겹치는지 확인
                overlap = np.any(seg_mask & segment_mask)  # 겹침 확인
                if overlap:  # 겹치면
                    # floor는 아래쪽, ceiling은 위쪽
                    wall_y_coords, _ = np.where(segment_mask)  # 벽 Y 좌표
                    if len(wall_y_coords) > 0:  # 좌표가 있으면
                        wall_bottom = wall_y_coords.max()  # 벽 하단
                        wall_top = wall_y_coords.min()  # 벽 상단
                        if avg_y > wall_bottom * 0.8:  # 아래쪽이면
                            floor_segments.append(seg_id)  # floor 추가
                        elif avg_y < wall_top * 1.2:  # 위쪽이면
                            ceiling_segments.append(seg_id)  # ceiling 추가
    
    # floor와 만나는 곳에서 수직선 그리기
    for floor_id in floor_segments:  # 각 floor에 대해
        floor_mask = (seg_map_resized == floor_id)  # floor 마스크
        intersection = floor_mask & segment_mask  # 교집합
        if np.any(intersection):  # 교집합이 있으면
            y_coords, x_coords = np.where(intersection)  # 교집합 좌표
            if len(y_coords) > 0:  # 좌표가 있으면
                intersection_y = int(y_coords.max())  # 교집합 최대 Y (벽 하단)
                unique_x = np.unique(x_coords)  # 고유한 X 좌표
                for x in unique_x[::5]:  # 일부만 선택
                    # 벽 내부로 수직선 그리기
                    wall_y_coords, _ = np.where(segment_mask & (seg_map_resized[:, int(x)] == seg_map_resized[intersection_y, int(x)]))  # 벽 Y 좌표
                    if len(wall_y_coords) > 0:  # 좌표가 있으면
                        wall_top = wall_y_coords.min()  # 벽 상단
                        # 수직선 그리기 (마스크 내부에만)
                        draw_line_in_mask(img, (int(x), intersection_y), (int(x), wall_top), segment_mask, color, thickness)  # 선 그리기
    
    # ceiling과 만나는 곳에서 수직선 그리기
    for ceiling_id in ceiling_segments:  # 각 ceiling에 대해
        ceiling_mask = (seg_map_resized == ceiling_id)  # ceiling 마스크
        intersection = ceiling_mask & segment_mask  # 교집합
        if np.any(intersection):  # 교집합이 있으면
            y_coords, x_coords = np.where(intersection)  # 교집합 좌표
            if len(y_coords) > 0:  # 좌표가 있으면
                intersection_y = int(y_coords.min())  # 교집합 최소 Y (벽 상단)
                unique_x = np.unique(x_coords)  # 고유한 X 좌표
                for x in unique_x[::5]:  # 일부만 선택
                    # 벽 내부로 수직선 그리기
                    wall_y_coords, _ = np.where(segment_mask & (seg_map_resized[:, int(x)] == seg_map_resized[intersection_y, int(x)]))  # 벽 Y 좌표
                    if len(wall_y_coords) > 0:  # 좌표가 있으면
                        wall_bottom = wall_y_coords.max()  # 벽 하단
                        # 수직선 그리기 (마스크 내부에만)
                        draw_line_in_mask(img, (int(x), intersection_y), (int(x), wall_bottom), segment_mask, color, thickness)  # 선 그리기


def draw_table_bed_cube(img, bbox_2d, depth_map, segment_mask, color, thickness=2, vanishing_point=None):
    """
    테이블/침대 구조를 그립니다 (직육면체, 퍼스펙티브 9개 선).
    
    Args:
        img (numpy.ndarray): 이미지
        bbox_2d (tuple): 2D 바운딩 박스
        depth_map (numpy.ndarray): Depth 맵
        segment_mask (numpy.ndarray): 세그먼트 마스크
        color (tuple): 색상
        thickness (int): 선 두께
        vanishing_point (tuple, optional): 소실점 (x, y), None이면 중심 사용
    """
    x_min, y_min, x_max, y_max = bbox_2d  # 바운딩 박스
    
    # Depth 범위로 높이 추정
    valid_depths = depth_map[segment_mask]  # 유효한 depth
    if len(valid_depths) == 0:  # 유효한 depth가 없으면
        return
    min_depth = valid_depths.min()  # 최소 depth
    max_depth = valid_depths.max()  # 최대 depth
    depth_diff = max_depth - min_depth  # Depth 차이
    
    # 앞면 4개 정점 (가까운 면)
    front_bottom_left = (x_min, y_max)  # 왼쪽 아래
    front_bottom_right = (x_max, y_max)  # 오른쪽 아래
    front_top_right = (x_max, y_min)  # 오른쪽 위
    front_top_left = (x_min, y_min)  # 왼쪽 위
    
    # [개선] 실제 소실점 또는 이미지 중심 사용
    if vanishing_point is not None:  # 소실점이 제공되면
        vp_x, vp_y = vanishing_point  # 실제 소실점 사용
    else:  # 소실점이 없으면
        vp_x = (x_min + x_max) / 2  # 이미지 중심 (기본값)
        vp_y = (y_min + y_max) / 2
    
    # [개선] Depth 범위에서 perspective 강도 계산
    if depth_diff > 0:
        # 깊이 차이가 크면 perspective가 더 강함
        # 원근감을 더 강하게 하기 위해 범위를 0.4~0.85로 조정
        perspective_scale = 0.85 - (depth_diff / (max_depth + 1)) * 0.45
        perspective_scale = max(0.4, min(0.85, perspective_scale))  # 범위 제한
    else:
        perspective_scale = 0.65  # 깊이 차이 없으면 더 강한 원근
    
    # [추가 개선] 바운딩 박스 중심에서 소실점까지의 거리로 회전 앙각 계산
    bbox_center_x = (x_min + x_max) / 2
    bbox_center_y = (y_min + y_max) / 2
    dx = vp_x - bbox_center_x
    dy = vp_y - bbox_center_y
    
    # [추가 개선] 상단이 소실점 방향으로 더 많이 축소되는 효과
    # 이는 객체가 지면에 놓여있고 카메라가 위쪽을 향하는 효과를 냄
    top_perspective_scale = perspective_scale * 0.85  # 상단을 더 축소
    bottom_perspective_scale = perspective_scale * 1.0  # 하단은 유지
    
    # 뒷면 4개 정점 (소실점 기반 계산 + 기울기 추가) [개선]
    back_bottom_left = (int(vp_x + (x_min - vp_x) * bottom_perspective_scale), 
                        int(vp_y + (y_max - vp_y) * bottom_perspective_scale))  # 뒷면 왼쪽 아래
    back_bottom_right = (int(vp_x + (x_max - vp_x) * bottom_perspective_scale),
                         int(vp_y + (y_max - vp_y) * bottom_perspective_scale))  # 뒷면 오른쪽 아래
    back_top_right = (int(vp_x + (x_max - vp_x) * top_perspective_scale),
                      int(vp_y + (y_min - vp_y) * top_perspective_scale))  # 뒷면 오른쪽 위 (더 축소)
    back_top_left = (int(vp_x + (x_min - vp_x) * top_perspective_scale),
                     int(vp_y + (y_min - vp_y) * top_perspective_scale))  # 뒷면 왼쪽 위 (더 축소)
    
    # 9개 선 (앞면 4개 + 뒷면 4개 + 연결선 4개 중 1개만)
    front_edges = [  # 앞면 4개 엣지
        (front_bottom_left, front_bottom_right),
        (front_bottom_right, front_top_right),
        (front_top_right, front_top_left),
        (front_top_left, front_bottom_left),
    ]
    
    back_edges = [  # 뒷면 4개 엣지
        (back_bottom_left, back_bottom_right),
        (back_bottom_right, back_top_right),
        (back_top_right, back_top_left),
        (back_top_left, back_bottom_left),
    ]
    
    connecting_edges = [  # 연결선 4개
        (front_bottom_left, back_bottom_left),
        (front_bottom_right, back_bottom_right),
        (front_top_right, back_top_right),
        (front_top_left, back_top_left),
    ]
    
    # 모든 엣지 그리기 (마스크 내부에만)
    for edge in front_edges + back_edges + connecting_edges:  # 모든 엣지에 대해
        pt1, pt2 = edge  # 두 점
        draw_line_in_mask(img, pt1, pt2, segment_mask, color, thickness)  # 마스크 내부에만 선 그리기


def draw_bridge_chair_table(img, bbox_2d, depth_map, segment_mask, color, thickness=2):
    """
    다리/의자/테이블 구조를 그립니다 (수평 큐브 + 수직 큐브 다리).
    
    Args:
        img (numpy.ndarray): 이미지
        bbox_2d (tuple): 2D 바운딩 박스
        depth_map (numpy.ndarray): Depth 맵
        segment_mask (numpy.ndarray): 세그먼트 마스크
        color (tuple): 색상
        thickness (int): 선 두께
    """
    x_min, y_min, x_max, y_max = bbox_2d  # 바운딩 박스
    center_x = (x_min + x_max) / 2  # 중심 X
    
    # 상판 (수평 큐브) - 상단 30%
    top_height = int((y_max - y_min) * 0.3)  # 상판 높이
    top_y_min = y_min  # 상판 상단
    top_y_max = y_min + top_height  # 상판 하단
    
    # 상판 외곽선
    top_mask = segment_mask.copy()  # 마스크 복사
    top_mask[top_y_max:, :] = False  # 하단 제거
    mask_uint8 = (top_mask.astype(np.uint8) * 255)  # 마스크 변환
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 외곽선
    if len(contours) > 0:  # 외곽선이 있으면
        largest_contour = max(contours, key=cv2.contourArea)  # 가장 큰 외곽선
        cv2.drawContours(img, [largest_contour], -1, color, thickness)  # 외곽선 그리기
    
    # 다리 (수직 큐브) - 하단 70%, 좌우 2개
    leg_width = (x_max - x_min) * 0.15  # 다리 너비
    leg_left_x = int(x_min + (x_max - x_min) * 0.2)  # 왼쪽 다리 X
    leg_right_x = int(x_min + (x_max - x_min) * 0.8)  # 오른쪽 다리 X
    leg_top = top_y_max  # 다리 상단
    leg_bottom = y_max  # 다리 하단
    
    # 왼쪽 다리
    leg_left_mask = segment_mask.copy()  # 마스크 복사
    leg_left_mask[:leg_top, :] = False  # 상단 제거
    leg_left_mask[:, :int(leg_left_x - leg_width/2)] = False  # 왼쪽 제거
    leg_left_mask[:, int(leg_left_x + leg_width/2):] = False  # 오른쪽 제거
    if np.any(leg_left_mask):  # 마스크가 있으면
        leg_contours, _ = cv2.findContours((leg_left_mask.astype(np.uint8) * 255), 
                                          cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 외곽선
        if len(leg_contours) > 0:  # 외곽선이 있으면
            cv2.drawContours(img, leg_contours, -1, color, thickness)  # 외곽선 그리기
    
    # 오른쪽 다리
    leg_right_mask = segment_mask.copy()  # 마스크 복사
    leg_right_mask[:leg_top, :] = False  # 상단 제거
    leg_right_mask[:, :int(leg_right_x - leg_width/2)] = False  # 왼쪽 제거
    leg_right_mask[:, int(leg_right_x + leg_width/2):] = False  # 오른쪽 제거
    if np.any(leg_right_mask):  # 마스크가 있으면
        leg_contours, _ = cv2.findContours((leg_right_mask.astype(np.uint8) * 255), 
                                          cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 외곽선
        if len(leg_contours) > 0:  # 외곽선이 있으면
            cv2.drawContours(img, leg_contours, -1, color, thickness)  # 외곽선 그리기


def draw_building_structure(img, bbox_2d, depth_map, segment_mask, img_bgr_resized, color, thickness=2):
    """
    건물 구조를 그립니다 (큐브 외곽선 + 폴리곤, depth/edge 참조).
    
    Args:
        img (numpy.ndarray): 이미지
        bbox_2d (tuple): 2D 바운딩 박스
        depth_map (numpy.ndarray): Depth 맵
        segment_mask (numpy.ndarray): 세그먼트 마스크
        img_bgr_resized (numpy.ndarray): 리사이즈된 원본 이미지
        color (tuple): 색상
        thickness (int): 선 두께
    """
    # 외곽선 그리기
    mask_uint8 = (segment_mask.astype(np.uint8) * 255)  # 마스크 변환
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 외곽선
    if len(contours) > 0:  # 외곽선이 있으면
        largest_contour = max(contours, key=cv2.contourArea)  # 가장 큰 외곽선
        cv2.drawContours(img, [largest_contour], -1, color, thickness)  # 외곽선 그리기
    
    # 이미지 엣지 찾기 (Canny)
    img_gray = cv2.cvtColor(img_bgr_resized, cv2.COLOR_BGR2GRAY)  # 그레이스케일 변환
    edges = cv2.Canny(img_gray, 50, 150)  # Canny 엣지 검출
    edges_in_mask = edges & (segment_mask.astype(np.uint8) * 255)  # 마스크 내부 엣지만
    
    # Depth edge와 이미지 edge 결합
    depth_h_edges, depth_v_edges = detect_depth_edges_in_segment(segment_mask, depth_map, threshold_ratio=0.1)  # Depth edge
    combined_edges = (depth_h_edges | depth_v_edges) & (edges_in_mask > 0)  # 결합된 엣지
    
    # 엣지를 따라 선 그리기 (마스크 내부에만)
    y_coords, x_coords = np.where(combined_edges)  # 엣지 좌표
    if len(y_coords) > 10:  # 충분한 점이 있으면
        # 간단한 폴리곤 표현 (일부 점만 연결)
        points = list(zip(x_coords[::max(1, len(x_coords)//100)], y_coords[::max(1, len(y_coords)//100)]))  # 일부 점만
        for i in range(len(points) - 1):  # 각 점 쌍에 대해
            pt1, pt2 = points[i], points[i+1]  # 두 점
            if segment_mask[pt1[1], pt1[0]] and segment_mask[pt2[1], pt2[0]]:  # 둘 다 마스크 내부이면
                cv2.line(img, pt1, pt2, color, thickness)  # 선 그리기


def draw_lamp_structure(img, bbox_2d, depth_map, segment_mask, color, thickness=2):
    """
    램프 구조를 그립니다 (수직 중심축 중심 원형 대칭).
    
    Args:
        img (numpy.ndarray): 이미지
        bbox_2d (tuple): 2D 바운딩 박스
        depth_map (numpy.ndarray): Depth 맵
        segment_mask (numpy.ndarray): 세그먼트 마스크
        color (tuple): 색상
        thickness (int): 선 두께
    """
    x_min, y_min, x_max, y_max = bbox_2d  # 바운딩 박스
    center_x = (x_min + x_max) / 2  # 중심 X
    center_y = (y_min + y_max) / 2  # 중심 Y
    
    # 수직 중심선 그리기 (마스크 내부에만)
    for y in range(y_min, y_max, 2):  # 2픽셀 간격
        x = int(center_x)  # 중심 X
        if 0 <= y < segment_mask.shape[0] and 0 <= x < segment_mask.shape[1]:  # 범위 내이면
            if segment_mask[y, x]:  # 마스크 내부이면
                cv2.circle(img, (x, y), 1, color, thickness)  # 점 그리기
    
    # 원형 대칭 구조 (수평 원들)
    height = y_max - y_min  # 높이
    num_circles = 5  # 원 개수
    for i in range(num_circles):  # 각 원에 대해
        y = int(y_min + (y_max - y_min) * (i + 1) / (num_circles + 1))  # Y 좌표
        radius = (x_max - x_min) / 2 * (1 - i * 0.1)  # 반지름 (위로 갈수록 작게)
        
        # 원 그리기 (마스크 내부에만)
        for angle in range(0, 360, 15):  # 각도별로
            rad = np.radians(angle)  # 라디안 변환
            x = int(center_x + radius * np.cos(rad))  # X 좌표
            py = int(y + radius * 0.3 * np.sin(rad))  # Y 좌표 (타원)
            if 0 <= py < segment_mask.shape[0] and 0 <= x < segment_mask.shape[1]:  # 범위 내이면
                if segment_mask[py, x]:  # 마스크 내부이면
                    cv2.circle(img, (x, py), 1, color, thickness)  # 점 그리기


def draw_shape_based_3d(img, segment_mask, depth_map, class_name, color, thickness=2, 
                       all_segments_info=None, seg_map_resized=None, img_bgr_resized=None, 
                       vanishing_point=None):
    """
    클래스 이름을 기반으로 적절한 3D 형태를 그립니다.
    
    Args:
        img (numpy.ndarray): 이미지 (BGR 형식)
        segment_mask (numpy.ndarray): 세그먼트 마스크 (boolean)
        depth_map (numpy.ndarray): Depth 맵
        class_name (str): 클래스 이름
        color (tuple): 색상 (B, G, R)
        thickness (int): 선 두께
        all_segments_info (list, optional): 모든 세그먼트 정보 (wall용)
        seg_map_resized (numpy.ndarray, optional): 리사이즈된 세그멘테이션 맵 (wall용)
        img_bgr_resized (numpy.ndarray, optional): 리사이즈된 원본 이미지 (building용)
        vanishing_point (tuple, optional): 소실점 (x, y), [개선] 추가됨
    """
    # 2D 바운딩 박스 계산
    y_coords, x_coords = np.where(segment_mask)  # 마스크 영역의 좌표 추출
    if len(y_coords) == 0 or len(x_coords) == 0:  # 좌표가 없으면
        return
    
    x_min, x_max = int(x_coords.min()), int(x_coords.max())  # X 범위
    y_min, y_max = int(y_coords.min()), int(y_coords.max())  # Y 범위
    bbox_2d = (x_min, y_min, x_max, y_max)  # 2D 바운딩 박스
    
    # 클래스 이름으로 형태 결정
    shape_type = get_shape_type_from_class(class_name)  # 형태 타입 가져오기
    
    # 형태에 따라 그리기
    if shape_type == "person":  # 사람
        draw_person_structure(img, bbox_2d, depth_map, segment_mask, color, thickness)  # 사람 그리기
    elif shape_type == "palm_tree":  # 야자수
        draw_palm_tree_structure(img, bbox_2d, depth_map, segment_mask, color, thickness)  # 야자수 그리기
    elif shape_type == "contour":  # 등고선 (mountain, earth)
        draw_contour_lines(img, segment_mask, depth_map, color, thickness)  # 등고선 그리기
    elif shape_type == "window" or shape_type == "floor" or shape_type == "ceiling":  # 창문, 바닥, 천장
        draw_horizontal_plane(img, segment_mask, color, thickness)  # 수평 평면 그리기
    elif shape_type == "wall":  # 벽
        if all_segments_info is not None and seg_map_resized is not None:  # 필요한 정보가 있으면
            # [개선] 소실점을 전달
            draw_wall_structure(img, bbox_2d, depth_map, segment_mask, all_segments_info, 
                              seg_map_resized, color, thickness, vanishing_point=vanishing_point)  # 벽 그리기
        else:  # 정보가 없으면
            draw_horizontal_plane(img, segment_mask, color, thickness)  # 기본 평면 그리기
    elif shape_type == "bridge" or shape_type == "chair" or shape_type == "table":  # 다리, 의자, 테이블
        draw_bridge_chair_table(img, bbox_2d, depth_map, segment_mask, color, thickness)  # 다리/의자/테이블 그리기
    elif shape_type == "bed":  # 침대
        # [개선] 소실점을 전달하여 더 정확한 원근 표현
        draw_table_bed_cube(img, bbox_2d, depth_map, segment_mask, color, thickness, 
                           vanishing_point=vanishing_point)  # 침대 그리기 (소실점 활용)
    elif shape_type == "building":  # 건물
        if img_bgr_resized is not None:  # 이미지가 있으면
            draw_building_structure(img, bbox_2d, depth_map, segment_mask, img_bgr_resized, color, thickness)  # 건물 그리기
        else:  # 이미지가 없으면
            draw_cube_structure(img, bbox_2d, depth_map, segment_mask, color, thickness)  # 기본 큐브 그리기
    elif shape_type == "lamp":  # 램프
        draw_lamp_structure(img, bbox_2d, depth_map, segment_mask, color, thickness)  # 램프 그리기
    elif shape_type == "paint" or shape_type == "cushion" or shape_type == "simple_plane":  # 페인트, 쿠션, 단순 평면
        draw_horizontal_plane(img, segment_mask, color, thickness)  # 외곽선만 그리기
    else:  # 기본값
        # Depth edge 기반 선 그리기 (마스크 내부에만)
        horizontal_edges, vertical_edges = detect_depth_edges_in_segment(segment_mask, depth_map)  # Edge 감지
        # 간단한 edge 선 그리기
        y_coords, x_coords = np.where(vertical_edges)  # 수직 edge 좌표
        if len(y_coords) > 5:  # 충분한 점이 있으면
            unique_y = np.unique(y_coords)  # 고유한 y 좌표
            for y in unique_y[::5]:  # 일부만 선택
                x_points = x_coords[y_coords == y]  # 해당 y의 x 좌표들
                if len(x_points) > 1:  # 점이 2개 이상이면
                    x_sorted = np.sort(x_points)  # 정렬
                    # 마스크 내부인지 확인하며 선 그리기
                    for i in range(len(x_sorted) - 1):  # 각 점 쌍에 대해
                        if segment_mask[y, x_sorted[i]] and segment_mask[y, x_sorted[i+1]]:  # 둘 다 마스크 내부이면
                            cv2.line(img, (x_sorted[i], y), (x_sorted[i+1], y), color, thickness)  # 선 그리기


# ============================================================================
# 시각화 함수
# ============================================================================



def visualize_cv2_all(img_bgr, seg_map, segments_info, id2label, is_thing_map, 
                     depth_map, filename, seg_time, depth_time, show_depth_mode=False, 
                     show_3d_boxes=False, selected_segment_id=None, vanishing_point=None):
    """
    Panoptic Segmentation과 Depth Estimation 결과를 OpenCV를 사용하여 시각화합니다.
    
    이 함수는 세그멘테이션 결과와 depth 정보를 받아서 Thing과 Stuff를 구분하여 시각화하고,
    각 세그먼트의 클래스 이름, 신뢰도 점수, 평균 거리, 통계 정보를 이미지에 오버레이합니다.
    
    Args:
        img_bgr (numpy.ndarray): 원본 이미지 (BGR 형식)
        seg_map (numpy.ndarray): 세그멘테이션 맵 (각 픽셀의 세그먼트 ID)
        segments_info (list): 각 세그먼트의 정보 딕셔너리 리스트 (id, label_id, score 등)
        id2label (dict): 라벨 ID를 클래스 이름으로 매핑하는 딕셔너리
        is_thing_map (dict): 라벨 ID를 Thing 여부(bool)로 매핑하는 딕셔너리
        depth_map (numpy.ndarray): Depth 맵 (이미지와 동일한 크기)
        filename (str): 현재 처리 중인 이미지 파일명
        seg_time (float): Segmentation 추론에 소요된 시간(초)
        depth_time (float): Depth Estimation 추론에 소요된 시간(초)
        show_depth_mode (bool): True면 Depth Map을 배경으로 사용, False면 원본 이미지 사용
        show_3d_boxes (bool): True면 3D 바운딩 박스 표시
        selected_segment_id (int, optional): 선택된 세그먼트 ID (None이면 모두 표시)
        vanishing_point (tuple, optional): 소실점 (x, y), [개선] 추가됨
    
    Returns:
        numpy.ndarray: 시각화된 이미지 (BGR 형식)
    """
    h, w = img_bgr.shape[:2]  # 이미지 높이와 너비 추출
    target_w = int(w * TARGET_HEIGHT / h)  # 비율 유지하며 목표 너비 계산
    
    # Depth 맵 리사이즈 및 정규화
    depth_resized = cv2.resize(depth_map, (target_w, TARGET_HEIGHT), interpolation=cv2.INTER_LINEAR)  # Depth 맵 리사이즈
    
    # Depth 값을 0-255 범위로 정규화 (가까운 곳이 밝게)
    depth_min = depth_resized.min()  # 최소값
    depth_max = depth_resized.max()  # 최대값
    if depth_max > depth_min:  # 범위가 있으면
        depth_normalized = ((depth_resized - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)  # 정규화
    else:  # 범위가 없으면
        depth_normalized = np.zeros_like(depth_resized, dtype=np.uint8)  # 모두 0
    
    # 배경 이미지 선택 (Depth 모드면 Depth 맵, 아니면 원본 이미지)
    if show_depth_mode:  # Depth 모드면
        resized_orig = cv2.cvtColor(depth_normalized, cv2.COLOR_GRAY2BGR)  # Depth 맵을 BGR로 변환
    else:  # 일반 모드면
        resized_orig = cv2.resize(img_bgr, (target_w, TARGET_HEIGHT), interpolation=cv2.INTER_LINEAR)  # 원본 이미지 리사이즈
    
    overlay_stuff = np.zeros_like(resized_orig, dtype=np.uint8)  # Stuff 오버레이 초기화
    
    inst_info = {s['id']: s for s in segments_info}  # 세그먼트 정보를 딕셔너리로 변환
    unique_ids = np.unique(seg_map)  # 고유 세그먼트 ID 추출
    
    # 선택된 세그먼트만 필터링
    if selected_segment_id is not None:  # 세그먼트가 선택되었으면
        if selected_segment_id in unique_ids:  # 선택된 ID가 있으면
            unique_ids = np.array([selected_segment_id])  # 선택된 ID만 사용
        else:  # 선택된 ID가 없으면
            unique_ids = np.array([])  # 빈 배열
    centroids = {}  # 중심점 저장 딕셔너리 초기화
    segment_depths = {}  # 세그먼트별 depth 저장 딕셔너리 초기화
    
    # 세그멘테이션 맵 리사이즈
    seg_map_resized = cv2.resize(seg_map.astype(np.float32), (target_w, TARGET_HEIGHT), 
                                 interpolation=cv2.INTER_NEAREST).astype(seg_map.dtype)
    
    # Stuff 그리기
    for i, cid in enumerate(unique_ids):  # 각 고유 ID 순회
        if cid not in inst_info:  # 정보가 없으면 건너뛰기
            continue
        info = inst_info[cid]  # 세그먼트 정보 가져오기
        label_id = info['label_id']  # 라벨 ID 추출
        is_thing = is_thing_map.get(label_id, False)  # Thing 여부 확인
        
        if is_thing:  # Thing이면 건너뛰기
            continue

        mask_resized = (seg_map_resized == cid)  # 리사이즈된 마스크 생성
        if not np.any(mask_resized):  # 마스크가 비어있으면 건너뛰기
            continue
        
        b, g, r = get_color(i)  # 색상 가져오기
        overlay_stuff[mask_resized] = (b, g, r)  # Stuff 영역에 색상 적용
        
        # 중심점 및 depth 계산
        y, x = np.where(mask_resized)  # 마스크 영역의 좌표 추출
        if len(y) > 0 and len(x) > 0:  # 좌표가 있으면
            centroids[int(cid)] = (int(x.mean()), int(y.mean()))  # 중심점 계산 및 저장
            segment_depths[int(cid)] = calculate_segment_depth(seg_map_resized, cid, depth_resized)  # Depth 계산

    # Stuff 오버레이 블렌딩 (Depth 모드에서는 더 투명하게)
    if show_depth_mode:  # Depth 모드면
        alpha = 80 / 255.0  # 더 투명하게 설정
    else:  # 일반 모드면
        alpha = 120 / 255.0  # 기본 투명도 설정
    blended = cv2.addWeighted(overlay_stuff, alpha, resized_orig, 1 - alpha, 0)  # Stuff 오버레이와 배경 이미지 블렌딩

    # Thing 그리기
    for i, cid in enumerate(unique_ids):  # 각 고유 ID 순회
        if cid not in inst_info:  # 정보가 없으면 건너뛰기
            continue
            
        info = inst_info[cid]  # 세그먼트 정보 가져오기
        label_id = info['label_id']  # 라벨 ID 추출
        is_thing = is_thing_map.get(label_id, False)  # Thing 여부 확인
        
        if not is_thing:  # Thing이 아니면 건너뛰기
            continue

        mask_resized = (seg_map_resized == cid)  # 리사이즈된 마스크 생성
        if not np.any(mask_resized):  # 마스크가 비어있으면 건너뛰기
            continue
        
        mask_uint8 = (mask_resized.astype(np.uint8) * 255)  # 마스크를 0-255 범위로 변환
        
        b, g, r = get_color(i)  # 색상 가져오기
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 외곽선 찾기
        cv2.drawContours(blended, contours, -1, (b, g, r), 2)  # Thing 외곽선 그리기
        
        # 중심점 및 depth 계산
        y, x = np.where(mask_resized)  # 마스크 영역의 좌표 추출
        if len(y) > 0 and len(x) > 0:  # 좌표가 있으면
            centroids[int(cid)] = (int(x.mean()), int(y.mean()))  # 중심점 계산 및 저장
            segment_depths[int(cid)] = calculate_segment_depth(seg_map_resized, cid, depth_resized)  # Depth 계산

    # 라벨 + 신뢰도 + Depth 텍스트
    font_scale = cv2.getFontScaleFromHeight(cv2.FONT_HERSHEY_SIMPLEX, 12, 1)  # 폰트 크기 계산
    
    for cid, (cx, cy) in centroids.items():  # 각 중심점에 대해
        label_id = inst_info[cid]['label_id']  # 라벨 ID 가져오기
        class_name = id2label.get(label_id, str(label_id)).split(';')[0]  # 클래스 이름 가져오기 (첫 번째만)
        score = inst_info[cid].get('score', 0.0)  # 신뢰도 점수 가져오기
        avg_depth = segment_depths.get(cid, 0.0)  # 평균 depth 가져오기
        
        is_thing = is_thing_map.get(label_id, False)  # Thing 여부 확인
        text_color = (0, 255, 255) if is_thing else (255, 255, 255)  # Thing은 노란색, Stuff는 흰색
        
        # 첫 번째 줄: 클래스 이름
        label_text = f"{class_name}"  # 라벨 텍스트 생성
        cv2.putText(blended, label_text, (cx - 10, cy),  # 텍스트 그리기
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1, cv2.LINE_AA)
        
        # 두 번째 줄: 신뢰도
        score_text = f"[{label_id}] {score:.2f}"  # 신뢰도 텍스트 생성
        cv2.putText(blended, score_text, (cx - 10, cy + 15),  # 신뢰도 텍스트 그리기
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, text_color, 1, cv2.LINE_AA)
        
        # 세 번째 줄: Depth (상대적 거리)
        depth_text = f"D: {avg_depth:.3f}"  # Depth 텍스트 생성
        cv2.putText(blended, depth_text, (cx - 10, cy + 30),  # Depth 텍스트 그리기
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, text_color, 1, cv2.LINE_AA)
    
    # 3D 형태 기반 시각화
    if show_3d_boxes:  # 3D 박스 표시 모드면
        for cid in unique_ids:  # 각 세그먼트에 대해
            if cid not in inst_info:  # 정보가 없으면 건너뛰기
                continue
            
            mask_resized = (seg_map_resized == cid)  # 리사이즈된 마스크 생성
            if not np.any(mask_resized):  # 마스크가 비어있으면 건너뛰기
                continue
            
            # 클래스 이름 가져오기
            label_id = inst_info[cid]['label_id']  # 라벨 ID 가져오기
            class_name = id2label.get(label_id, str(label_id)).split()[0]  # 클래스 이름 가져오기 (첫 번째만)
            
            # 색상 가져오기 (세그먼트 인덱스로)
            segment_idx = list(unique_ids).index(cid)  # 세그먼트 인덱스 찾기
            b, g, r = get_color(segment_idx)  # 색상 가져오기
            
            # 클래스 이름 기반 3D 형태 그리기
            # [개선] 소실점을 draw_shape_based_3d에 전달
            draw_shape_based_3d(blended, mask_resized, depth_resized, class_name, (b, g, r), thickness=2,
                              all_segments_info=segments_info, seg_map_resized=seg_map_resized, 
                              img_bgr_resized=resized_orig, vanishing_point=vanishing_point)  # 소실점 전달

    # 상단 정보
    thing_count = sum(1 for s in segments_info if is_thing_map.get(s['label_id'], False))  # Thing 개수 계산
    stuff_count = len(segments_info) - thing_count  # Stuff 개수 계산
    
    # 모드 표시
    mode_text = "Depth Map Mode" if show_depth_mode else "Segmentation Mode"  # 모드 텍스트
    box_text = " | 3D Boxes ON" if show_3d_boxes else ""  # 3D 박스 상태 텍스트
    select_text = f" | Selected: {selected_segment_id}" if selected_segment_id is not None else ""  # 선택된 세그먼트 텍스트
    info_text = f"{filename} | T:{thing_count} S:{stuff_count} | {mode_text}{box_text}{select_text}"  # 정보 텍스트 생성
    cv2.putText(blended, info_text, (10, 20),  # 상단 왼쪽에 정보 표시
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), 1, cv2.LINE_AA)
    
    # 시간 정보
    time_text = f"Seg: {seg_time:.3f}s | Depth: {depth_time:.3f}s"  # 추론 시간 텍스트 생성
    (tw, th), _ = cv2.getTextSize(time_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)  # 텍스트 크기 계산
    cv2.putText(blended, time_text, (target_w - tw - 10, 20),  # 상단 오른쪽에 시간 표시
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), 1, cv2.LINE_AA)
    
    # Depth 범위 정보 (Depth 모드일 때만)
    if show_depth_mode:  # Depth 모드면
        depth_range_text = f"Range: {depth_min:.3f} ~ {depth_max:.3f} (가까운 곳=밝게)"  # Depth 범위 텍스트 생성
        (tw2, th2), _ = cv2.getTextSize(depth_range_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.7, 1)  # 텍스트 크기 계산
        cv2.putText(blended, depth_range_text, (target_w - tw2 - 10, 45),  # 상단 오른쪽에 범위 표시
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.7, (255, 255, 255), 1, cv2.LINE_AA)

    window_name = "OneFormer + Depth - Panoptic Segmentation"  # 윈도우 이름 설정
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # 윈도우 생성
    cv2.resizeWindow(window_name, target_w, TARGET_HEIGHT)  # 윈도우 크기 조정
    cv2.moveWindow(window_name, 0, 0)  # 윈도우 위치 이동
    cv2.imshow(window_name, blended)  # 이미지 표시
    return blended, seg_map_resized  # 블렌딩된 이미지와 리사이즈된 세그멘테이션 맵 반환


# ============================================================================
# 모델 초기화
# ============================================================================

def initialize_models():
    """
    Segmentation 및 Depth Estimation 모델을 초기화합니다.
    
    Returns:
        tuple: (processor, segmentation_model, depth_estimator, device)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # GPU 사용 가능 여부에 따라 디바이스 선택
    print(f"🔧 디바이스: {device}")  # 디바이스 정보 출력
    
    # Segmentation 모델 로드
    print(f"🔧 Segmentation 모델 로드: {SEGMENTATION_MODEL}")  # 모델 로드 메시지 출력
    processor = OneFormerProcessor.from_pretrained(SEGMENTATION_MODEL)  # 프로세서 로드
    segmentation_model = OneFormerForUniversalSegmentation.from_pretrained(SEGMENTATION_MODEL)  # 모델 로드
    segmentation_model.to(device)  # 모델을 디바이스로 이동
    segmentation_model.eval()  # 평가 모드로 설정
    
    # Depth Estimation 모델 로드
    print(f"🔧 Depth 모델 로드: {DEPTH_MODEL}")  # Depth 모델 로드 메시지 출력
    depth_estimator = pipeline("depth-estimation", model=DEPTH_MODEL, device=0 if device.type == "cuda" else -1)  # Depth 추정 파이프라인 생성
    
    return processor, segmentation_model, depth_estimator, device


# ============================================================================
# 추론 함수
# ============================================================================

def run_inference(idx, processor, segmentation_model, depth_estimator, device, 
                 id2label, is_thing_map, image_files):
    """
    지정된 인덱스의 이미지에 대해 Panoptic Segmentation과 Depth Estimation 추론을 수행합니다.
    
    이 함수는 이미지 파일을 로드하고, OneFormer 모델을 사용하여 세그멘테이션을 수행하고,
    MiDaS 모델을 사용하여 Depth Estimation을 수행한 후, 결과를 시각화 함수로 전달합니다.
    
    Args:
        idx (int): image_files 리스트에서 처리할 이미지의 인덱스
        processor: OneFormer 프로세서
        segmentation_model: OneFormer 모델
        depth_estimator: MiDaS Depth Estimation 파이프라인
        device: PyTorch 디바이스
        id2label (dict): 라벨 ID를 클래스 이름으로 매핑하는 딕셔너리
        is_thing_map (dict): 라벨 ID를 Thing 여부(bool)로 매핑하는 딕셔너리
        image_files (list): 이미지 파일 경로 리스트
    
    출력:
        - 현재 이미지 정보 (인덱스/총 개수, 파일명)
        - Thing/Stuff 개수 디버그 정보
        - 추론 완료 메시지 및 소요 시간
        - 시각화된 결과 이미지 (OpenCV 윈도우)
    """
    img_path = image_files[idx]  # 이미지 경로 가져오기
    filename = os.path.basename(img_path)  # 파일명 추출
    
    img_bgr = cv2.imread(img_path)  # 이미지 읽기 (BGR 형식)
    if img_bgr is None:  # 이미지 로드 실패 시
        print(f"❌ 이미지 로드 실패: {img_path}")  # 에러 메시지 출력
        return None
   
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # BGR을 RGB로 변환
    h, w = img_bgr.shape[:2]  # 이미지 크기 추출
    
    print(f"\n📂 [{idx+1}/{len(image_files)}] {filename}")  # 현재 이미지 정보 출력
    
    # Panoptic Segmentation 추론
    inputs = processor(images=img_rgb, task_inputs=["panoptic"], return_tensors="pt")  # 이미지 전처리
    inputs = {k: v.to(device) for k, v in inputs.items()}  # 입력을 디바이스로 이동
    
    seg_start_time = time.time()  # Segmentation 시작 시간 기록
    with torch.no_grad():  # 그래디언트 계산 비활성화
        seg_outputs = segmentation_model(**inputs)  # 모델 추론 수행
    seg_time = time.time() - seg_start_time  # Segmentation 추론 시간 계산
        
    panoptic_result = processor.post_process_panoptic_segmentation(  # Panoptic 세그멘테이션 후처리
        seg_outputs, target_sizes=[(h, w)])[0]
    
    seg_map = panoptic_result["segmentation"].cpu().numpy()  # 세그멘테이션 맵 추출
    segments_info = panoptic_result["segments_info"]  # 세그먼트 정보 추출
    
    # Depth Estimation 추론
    img_pil = Image.fromarray(img_rgb)  # PIL Image로 변환
    depth_start_time = time.time()  # Depth 시작 시간 기록
    depth_result = depth_estimator(img_pil)  # Depth 추정 수행
    depth_time = time.time() - depth_start_time  # Depth 추론 시간 계산
    
    # Depth 맵을 numpy array로 변환
    depth_map = np.array(depth_result["depth"])  # Depth 맵 추출
    
    thing_count = sum(1 for s in segments_info if is_thing_map.get(s['label_id'], False))  # Thing 개수 계산
    print(f"DEBUG - Thing: {thing_count}, Stuff: {len(segments_info) - thing_count}")  # 디버그 정보 출력
    print(f"✓ Segmentation 완료 ({seg_time:.4f}초)")  # Segmentation 완료 메시지 출력
    print(f"✓ Depth Estimation 완료 ({depth_time:.4f}초)")  # Depth 완료 메시지 출력

    return seg_map, segments_info, depth_map, img_bgr, filename, seg_time, depth_time  # 결과 반환


# ============================================================================
# 메인 함수
# ============================================================================

def main():
    """
    프로그램의 메인 실행 루프입니다.
    
    이 함수는 프로그램의 진입점으로, 모델을 초기화하고 첫 번째 이미지를 자동으로 로드하며,
    키보드 입력을 받아 이미지 간 이동을 처리하는 인터랙티브 루프를 실행합니다.
    
    실행 흐름:
        1. 이미지 파일 목록 로드
        2. 모델 초기화 (Segmentation + Depth)
        3. ADE20K 매핑 데이터 로드
        4. 첫 번째 이미지(인덱스 0) 자동 추론 및 시각화
        5. 사용법 안내 메시지 출력
        6. 무한 루프로 키보드 입력 대기
        7. 입력에 따라 이전/다음 이미지로 이동하거나 프로그램 종료
    
    키보드 입력:
        - 'A' 또는 왼쪽 화살표 (0x250000): 이전 이미지로 이동
        - 'D' 또는 오른쪽 화살표 (0x270000): 다음 이미지로 이동
        - 'S': Depth Map 표시 (밝기로, 가까운 곳=밝게)
        - 'Q': 프로그램 종료
    
    특징:
        - 이미지 인덱스는 순환 구조 (마지막 이미지에서 다음 = 첫 이미지)
        - 각 이미지 이동 시 자동으로 추론 및 시각화 수행
        - 종료 시 모든 OpenCV 윈도우 자동 닫기
    """
    # 이미지 리스트 로드
    image_files = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.jpg")))  # JPG 파일 목록 정렬
    if not image_files:  # 이미지가 없으면
        raise FileNotFoundError(f"'{IMAGE_DIR}'에 이미지가 없습니다.")  # 에러 발생
    print(f"📂 {len(image_files)}개 이미지")  # 이미지 개수 출력
    
    # 모델 초기화
    processor, segmentation_model, depth_estimator, device = initialize_models()  # 모델 초기화
    
    # ADE20K 공식 Thing/Stuff 분류 및 클래스 이름 사용
    from ade20k_thing_stuff_map import ADE20K_THING_STUFF_CLASSES, ADE20K_CLASS_NAMES  # ADE20K 매핑 데이터 import
    
    is_thing_map = ADE20K_THING_STUFF_CLASSES  # Thing/Stuff 분류 맵 설정
    id2label = ADE20K_CLASS_NAMES  # 공식 클래스 이름 사용
    
    thing_count = sum(1 for v in is_thing_map.values() if v)  # Thing 클래스 개수 계산
    stuff_count = sum(1 for v in is_thing_map.values() if not v)  # Stuff 클래스 개수 계산
    print(f"✓ ADE20K 공식 Thing/Stuff 분류 사용 (CSAILVision MIT)")  # 분류 사용 메시지 출력
    print(f"  - Thing: {thing_count}개 클래스")  # Thing 개수 출력
    print(f"  - Stuff: {stuff_count}개 클래스")  # Stuff 개수 출력
    
    # [개선] 소실점 검출기 초기화
    vp_detector = VanishingPointDetector()  # 소실점 검출기 인스턴스 생성
    print("✓ VanishingPointDetector 활성화됨")  # 활성화 메시지 출력
    
    # 첫 번째 이미지 추론
    cur_idx = 0  # 현재 이미지 인덱스 초기화
    show_depth_mode = False  # Depth 모드 상태 초기화 (False = Segmentation 모드)
    show_3d_boxes = False  # 3D 박스 표시 상태 초기화
    selected_segment_id = None  # 선택된 세그먼트 ID 초기화
    
    # 현재 추론 결과 저장 변수
    current_seg_map = None  # 현재 세그멘테이션 맵
    current_seg_map_resized = None  # 현재 리사이즈된 세그멘테이션 맵
    current_segments_info = None  # 현재 세그먼트 정보
    current_depth_map = None  # 현재 Depth 맵
    current_img_bgr = None  # 현재 이미지
    current_filename = None  # 현재 파일명
    current_seg_time = 0.0  # 현재 Segmentation 시간
    current_depth_time = 0.0  # 현재 Depth 시간
    
    # 마우스 콜백 함수 (전역 변수 접근을 위해 클로저 사용)
    def mouse_callback(event, x, y, flags, param):
        """마우스 클릭 시 세그먼트 선택"""
        nonlocal selected_segment_id, current_seg_map_resized, current_segments_info, current_seg_map
        if event == cv2.EVENT_RBUTTONDOWN:  # 오른쪽 버튼 클릭
            selected_segment_id = None  # 선택 해제 (모든 인스턴스 표시)
            print("🖱️ 모든 인스턴스 표시")  # 모든 인스턴스 표시 메시지 출력
            # 시각화 업데이트
            if current_seg_map is not None:  # 세그멘테이션 맵이 있으면
                # [개선] 소실점 검출
                vanishing_point = vp_detector.find_vanishing_point(current_img_bgr)  # 소실점 검출
                _, current_seg_map_resized = visualize_cv2_all(current_img_bgr, current_seg_map, current_segments_info, id2label, 
                                         is_thing_map, current_depth_map, current_filename, 
                                         current_seg_time, current_depth_time, show_depth_mode, 
                                         show_3d_boxes, selected_segment_id, vanishing_point=vanishing_point)  # 소실점 전달
        elif event == cv2.EVENT_LBUTTONDOWN:  # 왼쪽 버튼 클릭
            if current_seg_map_resized is not None:  # 세그멘테이션 맵이 있으면
                # 클릭한 위치의 세그먼트 ID 찾기
                if 0 <= y < current_seg_map_resized.shape[0] and 0 <= x < current_seg_map_resized.shape[1]:  # 범위 내이면
                    clicked_id = int(current_seg_map_resized[y, x])  # 클릭한 위치의 세그먼트 ID
                    
                    if clicked_id == 0:  # 배경이면
                        selected_segment_id = None  # 선택 해제
                        print("🖱️ 선택 해제")  # 선택 해제 메시지 출력
                    else:  # 세그먼트가 있으면
                        # 클릭한 위치 주변의 모든 세그먼트 찾기 (겹쳐있는 경우)
                        # 주변 영역에서 세그먼트 찾기
                        search_radius = 5  # 검색 반경
                        y_min = max(0, y - search_radius)  # Y 최소값
                        y_max = min(current_seg_map_resized.shape[0], y + search_radius + 1)  # Y 최대값
                        x_min = max(0, x - search_radius)  # X 최소값
                        x_max = min(current_seg_map_resized.shape[1], x + search_radius + 1)  # X 최대값
                        
                        nearby_segments = set()  # 주변 세그먼트 집합
                        for py in range(y_min, y_max):  # Y 범위
                            for px in range(x_min, x_max):  # X 범위
                                seg_id = int(current_seg_map_resized[py, px])  # 세그먼트 ID
                                if seg_id != 0:  # 배경이 아니면
                                    nearby_segments.add(seg_id)  # 세그먼트 추가
                        
                        if len(nearby_segments) > 0:  # 주변 세그먼트가 있으면
                            # 각 세그먼트의 면적 계산
                            segment_areas = []  # 세그먼트 면적 리스트
                            for seg_id in nearby_segments:  # 각 세그먼트에 대해
                                mask = (current_seg_map_resized == seg_id)  # 세그먼트 마스크
                                area = np.sum(mask)  # 면적 계산
                                segment_areas.append((seg_id, area))  # ID와 면적 추가
                            
                            # 면적이 작은 것 우선 정렬
                            segment_areas.sort(key=lambda x: x[1])  # 면적 기준 정렬
                            selected_segment_id = segment_areas[0][0]  # 가장 작은 것 선택
                            
                            # 세그먼트 정보 출력
                            for seg_info in current_segments_info:  # 각 세그먼트에 대해
                                if seg_info['id'] == selected_segment_id:  # 선택된 ID와 같으면
                                    label_id = seg_info.get('label_id', 'N/A')  # 라벨 ID
                                    score = seg_info.get('score', 0.0)  # 신뢰도
                                    area = segment_areas[0][1]  # 면적
                                    print(f"🖱️ 세그먼트 선택: ID={selected_segment_id}, Label={label_id}, Score={score:.2f}, Area={area}")  # 선택 메시지 출력
                                    break  # 찾았으면 종료
                        else:  # 주변 세그먼트가 없으면
                            selected_segment_id = clicked_id  # 클릭한 ID 선택
                            print(f"🖱️ 세그먼트 선택: ID={clicked_id}")  # 선택 메시지 출력
                    
                    # 시각화 업데이트
                    if current_seg_map is not None:  # 세그멘테이션 맵이 있으면
                        # [개선] 소실점 검출
                        vanishing_point = vp_detector.find_vanishing_point(current_img_bgr)  # 소실점 검출
                        _, current_seg_map_resized = visualize_cv2_all(current_img_bgr, current_seg_map, current_segments_info, id2label, 
                                         is_thing_map, current_depth_map, current_filename, 
                                         current_seg_time, current_depth_time, show_depth_mode, 
                                         show_3d_boxes, selected_segment_id, vanishing_point=vanishing_point)  # 소실점 전달
    
    # 마우스 콜백 등록
    window_name = "OneFormer + Depth - Panoptic Segmentation"  # 윈도우 이름
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # 윈도우 생성
    cv2.setMouseCallback(window_name, mouse_callback)  # 마우스 콜백 등록
    
    # 첫 번째 이미지 추론
    result = run_inference(cur_idx, processor, segmentation_model, depth_estimator, device,
                 id2label, is_thing_map, image_files)  # 첫 번째 이미지 추론
    if result is not None:  # 결과가 있으면
        current_seg_map, current_segments_info, current_depth_map, current_img_bgr, \
        current_filename, current_seg_time, current_depth_time = result  # 결과 저장
        # 현재 모드에 맞게 시각화
        # [개선] 소실점 검출
        vanishing_point = vp_detector.find_vanishing_point(current_img_bgr)  # 소실점 검출
        _, current_seg_map_resized = visualize_cv2_all(current_img_bgr, current_seg_map, current_segments_info, id2label, 
                         is_thing_map, current_depth_map, current_filename, 
                         current_seg_time, current_depth_time, show_depth_mode, 
                         show_3d_boxes, selected_segment_id, vanishing_point=vanishing_point)  # 소실점 전달

    print("\n키: A/← (이전), D/→ (다음), S (모드 전환), E (3D 박스), Q (종료)")  # 사용법 안내 출력
    print("마우스: 왼쪽 클릭 (세그먼트 선택), 우클릭 (모든 인스턴스 표시)")  # 마우스 사용법 안내 출력

    while True:  # 무한 루프
        key = cv2.waitKey(0) & 0xFF  # 키 입력 대기
        if key == ord('a') or key == 0x250000:  # 'a' 또는 왼쪽 화살표 키
            cur_idx = (cur_idx - 1) % len(image_files)  # 이전 이미지로 이동
            result = run_inference(cur_idx, processor, segmentation_model, depth_estimator, device,
                         id2label, is_thing_map, image_files)  # 추론 수행
            if result is not None:  # 결과가 있으면
                current_seg_map, current_segments_info, current_depth_map, current_img_bgr, \
                current_filename, current_seg_time, current_depth_time = result  # 결과 저장
                selected_segment_id = None  # 이미지 변경 시 선택 해제
                # [개선] 소실점 검출
                vanishing_point = vp_detector.find_vanishing_point(current_img_bgr)  # 소실점 검출
                # 현재 모드에 맞게 시각화
                _, current_seg_map_resized = visualize_cv2_all(current_img_bgr, current_seg_map, current_segments_info, id2label, 
                                 is_thing_map, current_depth_map, current_filename, 
                                 current_seg_time, current_depth_time, show_depth_mode, 
                                 show_3d_boxes, selected_segment_id, vanishing_point=vanishing_point)  # 소실점 전달
        elif key == ord('d') or key == 0x270000:  # 'd' 또는 오른쪽 화살표 키
            cur_idx = (cur_idx + 1) % len(image_files)  # 다음 이미지로 이동
            result = run_inference(cur_idx, processor, segmentation_model, depth_estimator, device,
                         id2label, is_thing_map, image_files)  # 추론 수행
            if result is not None:  # 결과가 있으면
                current_seg_map, current_segments_info, current_depth_map, current_img_bgr, \
                current_filename, current_seg_time, current_depth_time = result  # 결과 저장
                selected_segment_id = None  # 이미지 변경 시 선택 해제
                # [개선] 소실점 검출
                vanishing_point = vp_detector.find_vanishing_point(current_img_bgr)  # 소실점 검출
                # 현재 모드에 맞게 시각화
                _, current_seg_map_resized = visualize_cv2_all(current_img_bgr, current_seg_map, current_segments_info, id2label, 
                                 is_thing_map, current_depth_map, current_filename, 
                                 current_seg_time, current_depth_time, show_depth_mode, 
                                 show_3d_boxes, selected_segment_id, vanishing_point=vanishing_point)  # 소실점 전달
        elif key == ord('s'):  # 's' 키 - 모드 전환
            show_depth_mode = not show_depth_mode  # 모드 토글
            if current_seg_map is not None:  # 추론 결과가 있으면
                mode_name = "Depth Map 모드" if show_depth_mode else "Segmentation 모드"  # 모드 이름
                print(f"🔄 모드 전환: {mode_name}")  # 모드 전환 메시지 출력
                # [개선] 소실점 검출
                vanishing_point = vp_detector.find_vanishing_point(current_img_bgr)  # 소실점 검출
                # 현재 모드에 맞게 시각화
                _, current_seg_map_resized = visualize_cv2_all(current_img_bgr, current_seg_map, current_segments_info, id2label, 
                                 is_thing_map, current_depth_map, current_filename, 
                                 current_seg_time, current_depth_time, show_depth_mode, 
                                 show_3d_boxes, selected_segment_id, vanishing_point=vanishing_point)  # 소실점 전달
            else:  # 추론 결과가 없으면
                print("⚠️ 먼저 이미지를 로드하세요.")  # 경고 메시지 출력
        elif key == ord('e'):  # 'e' 키 - 3D 박스 토글
            show_3d_boxes = not show_3d_boxes  # 3D 박스 표시 토글
            if current_seg_map is not None:  # 추론 결과가 있으면
                box_status = "ON" if show_3d_boxes else "OFF"  # 박스 상태
                print(f"📦 3D 박스: {box_status}")  # 박스 상태 메시지 출력
                # [개선] 소실점 검출
                vanishing_point = vp_detector.find_vanishing_point(current_img_bgr)  # 소실점 검출
                # 현재 모드에 맞게 시각화
                _, current_seg_map_resized = visualize_cv2_all(current_img_bgr, current_seg_map, current_segments_info, id2label, 
                                 is_thing_map, current_depth_map, current_filename, 
                                 current_seg_time, current_depth_time, show_depth_mode, 
                                 show_3d_boxes, selected_segment_id, vanishing_point=vanishing_point)  # 소실점 전달
            else:  # 추론 결과가 없으면
                print("⚠️ 먼저 이미지를 로드하세요.")  # 경고 메시지 출력
        elif key == ord('q'):  # 'q' 키
            print("\n👋 종료")  # 종료 메시지 출력
            break  # 루프 종료
    cv2.destroyAllWindows()  # 모든 윈도우 닫기


if __name__ == "__main__":  # 스크립트가 직접 실행될 때
    main()  # 메인 함수 실행
