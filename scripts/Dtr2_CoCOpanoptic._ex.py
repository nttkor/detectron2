# 코드 구조 분석
# 1. 프로그램 목적
# Detectron2의 COCO Panoptic Segmentation 모델로 이미지를 추론하고, 4가지 시각화 모드로 결과를 표시하는 인터랙티브 뷰어입니다.
# 2. 주요 기능
# 시각화 모드 (4가지)
# 모드 0: Panoptic Segmentation — Visualizer.draw_panoptic_seg_predictions() 사용 (Thing + Stuff 모두 표시)
# 모드 1: Instance Segmentation — Visualizer.draw_instance_predictions() 사용 (Thing만 표시, instances가 없으면 panoptic_seg 사용)
# 모드 2: Semantic Segmentation — Visualizer.draw_sem_seg() 사용 (sem_seg가 있으면 사용, 없으면 panoptic_seg에서 추출)
# 모드 3: Contour Visualization — 원본 이미지에 각 세그먼트별 윤곽선과 클래스명 표시 (Thing + Stuff 모두 표시, 클래스별 다른 색상)
# 인터랙티브 조작
# A/←: 이전 이미지
# D/→: 다음 이미지
# S: 시각화 모드 전환 (이미지 유지, 모드만 변경)
# Q/ESC: 종료
# 3. 코드 구조
# 상수 및 설정
# IMAGE_DIR = "..."              # 이미지 디렉토리 경로
# MODEL_YAML = "..."             # 모델 설정 파일
# TARGET_WIN_HEIGHT = 800        # 윈도우 높이 고정값
# SCALE_THRESHOLD = 512          # Visualizer 스케일 조정 기준
# VIZ_MODES = [...]              # 시각화 모드 이름 리스트 (4개)
# 핵심 함수들
# setup_model()
# Detectron2 모델 로드
# Predictor와 메타데이터 초기화
# GPU/CPU 자동 선택
# get_image_files()
# 지정된 디렉토리에서 JPG 이미지 파일 목록 수집 (JPG만 지원)
# visualize_output_by_mode()
# 모드에 따라 시각화 수행
# 모드 0: Visualizer.draw_panoptic_seg_predictions() 사용
# 모드 1: Visualizer.draw_instance_predictions() 사용 (instances가 없으면 panoptic_seg 사용)
# 모드 2: Visualizer.draw_sem_seg() 사용 (sem_seg shape 변환 처리: (C,H,W) -> (H,W))
# 모드 3 (Contour): 각 세그먼트별로 윤곽선과 클래스명을 원본 이미지에 그리기
# overlay_window_info()
# 좌측: 파일 정보
# 중앙: 현재 모드 이름
# 우측: 추론 시간
# 폰트: cv2.getFontScaleFromHeight()로 12px 높이 고정, 두께 1, 고정값 사용
# show_result_in_window()
# 윈도우 높이를 800px로 고정
# 너비는 원본 비율 유지
# 오버레이는 imshow 전에 처리
# 4. 메인 실행 흐름
# 1. 모델 및 이미지 파일 로드
# 2. 초기 이미지 추론 수행
# 3. 무한 루프:
#    - 현재 모드로 시각화
#    - 정보 오버레이 추가 (imshow 전에 처리)
#    - 윈도우에 표시
#    - 키 입력 대기
#    - 입력에 따라:
#      * A/D: 이미지 변경 (추론 재수행)
#      * S: 모드만 변경 (추론 결과 재사용)
#      * Q: 종료
# 5. 최적화 포인트
# 추론 결과 재사용: S 키로 모드 전환 시 추론을 다시 하지 않고 저장된 current_outputs 재사용
# 추론 시간 저장: current_inference_time에 저장하여 모드 전환 시에도 동일한 시간 표시
# 메모리 관리: .copy()로 메모리 연속성 보장
# 6. 특징
# 윈도우 크기 고정: 높이 800px, 비율 유지
# 4가지 시각화 모드: Panoptic, Instance, Semantic, Contour
# 효율적인 모드 전환: 이미지 변경 없이 모드만 전환 가능
# 안전한 종료: 윈도우 X 버튼 감지 및 키 입력 처리
# 7. 작업 내역
# - 초기 구현: Detectron2 Tutorial 기반 panoptic segmentation 추론 및 시각화
# - 모드 확장: 3가지 모드에서 4가지 모드로 확장 (Panoptic, Instance, Semantic, Contour)
# - Instance Segmentation 모드: outputs["instances"] 사용, 없으면 panoptic_seg 사용
# - Semantic Segmentation 모드: outputs["sem_seg"] 사용, shape 변환 처리 ((C,H,W) -> (H,W) via argmax)
# - Contour 모드 개선:
#   * 전체 이진 마스크 대신 각 세그먼트별로 윤곽선을 그려 모든 세그먼트가 정확히 표시되도록 개선
#   * 클래스별 다른 색상 적용 (category_id 기반 HSV 색상 생성)
#   * 각 세그먼트 중심점에 클래스명 표시 추가
#   * Thing은 파란색 배경, Stuff는 검정색 배경으로 구분
# - 이미지 로딩: JPG 파일만 읽도록 필터링 (get_image_files 함수)
# - 폰트 설정 개선:
#   * cv2.getFontScaleFromHeight() 사용하여 12px 높이 고정
#   * 폰트 두께 1로 설정
#   * 이미지 해상도와 무관하게 고정값 사용
#   * cv2.LINE_AA 안티앨리어싱 적용
# - 오버레이 처리 순서 개선: 추론 → 시각화 → 오버레이 → imshow 순서로 명확히 분리
# - 오버레이 함수: 원본 이미지를 수정하지 않고 복사본에 텍스트 추가
# 이 코드는 추론 결과를 재사용해 모드 전환을 빠르게 하고, 사용자 인터페이스를 제공합니다.
# ====================================================
# 필수 라이브러리 임포트 (Import Libraries)
# ====================================================

# OS 및 파일 시스템 관리 라이브러리 (경로, 파일 목록 처리)
import os
import glob
import time

# 계산 및 배열 처리 라이브러리 (이미지 데이터를 NumPy 배열로 처리)
import numpy as np

# 컴퓨터 비전 처리 라이브러리 (이미지 로드, 창 관리, 윤곽선, 텍스트 오버레이 등)
import cv2

# PyTorch (딥러닝 모델의 기본 프레임워크 및 GPU/CPU 연산 관리)
import torch

# Detectron2 프레임워크 관련 클래스 임포트
from detectron2.config import get_cfg                     # Detectron2 설정(Configuration) 관리
from detectron2 import model_zoo                         # Detectron2 모델 설정 파일 및 가중치 경로 제공
from detectron2.engine import DefaultPredictor           # 기본 모델 추론기 클래스 (추론 실행)
from detectron2.utils.logger import setup_logger         # Detectron2 로깅 시스템 설정
from detectron2.data import MetadataCatalog              # 데이터셋의 클래스 이름, 색상 등 메타데이터 접근
from detectron2.utils.visualizer import Visualizer, ColorMode  # 추론 결과를 이미지 위에 시각화 (인스턴스/팬옵틱)

# ====================================================
# 1. 상수 및 설정 정의 (Configuration)
# ====================================================
# ⚠️ 경로 수정 필요: 본인의 이미지 경로로 변경하세요.
IMAGE_DIR = r"D:/git/detectron2/ade20k_consistency/original_ade20k" 
MODEL_YAML = "COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml" 
WINDOW_NAME = "Detectron2 Panoptic Viewer"               # OpenCV 윈도우 이름
TARGET_WIN_HEIGHT = 800                                 # [요구사항] 윈도우 창 높이 고정값 (px)
SCALE_THRESHOLD = 512                                   # Visualizer 스케일 조정 기준 높이 (px)
VIZ_MODES = ["Panoptic Seg.", "Instance Seg.", "Semantic Seg.", "Contour Viz."] # 시각화 모드 이름 리스트

# 텍스트 오버레이 스타일 설정 (기준 높이 800px 기준)
FONT = cv2.FONT_HERSHEY_SIMPLEX
BASE_FONT_SCALE = 1.0        # 기준 높이(800px)에서의 폰트 크기
BASE_FONT_THICKNESS = 2      # 기준 높이(800px)에서의 폰트 두께 (더 두껍게)
BASE_Y_POS = 50              # 기준 높이(800px)에서의 Y 위치
BASE_SHADOW_OFFSET = 1       # 기준 높이(800px)에서의 그림자 오프셋
TEXT_COLOR = (255, 255, 255) # White (BGR)
SHADOW_COLOR = (0, 0, 0)     # Black Shadow
CONTOUR_COLOR = (0, 255, 255) # Yellow (BGR) - 윤곽선 색상
CONTOUR_THICKNESS = 2        # 윤곽선 두께

# ====================================================
# 2. 유틸리티 함수
# ====================================================

def setup_model():
    """Detectron2 모델 설정 및 Predictor 초기화"""
    print("\n⚙️  모델을 로드하고 있습니다...")
    setup_logger() # Detectron2 로거 설정
    
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(MODEL_YAML)) # 모델 설정 파일 로드
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(MODEL_YAML) # 가중치 파일 다운로드 및 로드
    
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5             # 인스턴스 추론 시 임계값 설정
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # 사용 가능한 장치 설정 (GPU 또는 CPU)
    
    predictor = DefaultPredictor(cfg)                       # DefaultPredictor 객체 생성
    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])  # 데이터셋의 메타데이터 로드
    
    print(f"✅ 모델 로드 완료 (Device: {cfg.MODEL.DEVICE})")
    return predictor, metadata

def get_image_files(img_dir):
    """지정된 디렉토리에서 JPG 이미지 파일 리스트를 가져옵니다."""
    files = glob.glob(os.path.join(img_dir, "*.jpg"))  # JPG 파일만
    
    files = sorted(files) # 파일 목록을 알파벳 순으로 정렬
    if not files:
        raise FileNotFoundError(f"❌ '{img_dir}' 경로에 JPG 이미지가 없습니다.")
    
    print(f"📂 총 {len(files)}개의 JPG 이미지를 찾았습니다.")
    return files

def visualize_output_by_mode(original_img_bgr, outputs, metadata, viz_scale, viz_mode):
    """
    지정된 모드에 따라 추론 결과를 시각화한 이미지를 생성합니다.
    모든 출력 이미지는 .copy()를 통해 메모리 연속성을 보장합니다.
    3. 주요 매개변수 및 사용자 지정 (Key Parameters and Customization)
    Visualizer를 인스턴스화하거나 사용할 때 가장 중요한 인수는 다음과 같습니다.
    🌟 생성자 (The Constructor)
    Python
    Visualizer(img_rgb, metadata=None, scale=1.0, instance_mode=ColorMode.IMAGE)
    img_rgb: 이미지 ndarray입니다. 중요: 반드시 RGB 형식인지 확인해야 합니다. OpenCV로 이미지를 로드했다면, img[:, :, ::-1]을 사용하여 채널 순서를 반전시키십시오.
    metadata: 클래스 이름(예: "person", "car") 및 색상 팔레트가 포함됩니다.
    scale: 시각화 결과의 크기를 조정합니다. 라벨(Label)이 너무 작아 읽기 어려울 경우, scale=1.2 또는 1.5를 사용하십시오.
    instance_mode:
    from detectron2.utils.visualizer import ColorMode를 사용하여 가져옵니다.
    ColorMode.IMAGE: (기본값) 마스크에 무작위 색상을 선택합니다.
    ColorMode.SEGMENTATION: 배경을 그레이스케일로 변환하여, 색상이 입혀진 마스크가 더욱 두드러지게 만듭니다.

    🎨 그리기 메서드 (Drawing Methods)
    draw_instance_predictions 외에도 다음을 사용할 수 있습니다.
    draw_dataset_dict(dic): 데이터셋 딕셔너리에서 정답(Ground Truth) 주석을 그립니다.
    draw_sem_seg(sem_seg): 의미론적 분할 (Semantic Segmentation) 결과를 그립니다.
    draw_panoptic_seg_predictions(...): 팬옵틱 분할 (Panoptic Segmentation) 출력에 특화된 메서드입니다.

    4. 흔히 발생하는 문제 (Common Pitfalls)
    색상 채널 불일치 (Color Channel Mismatch): 이미지가 파란색/주황색으로 반전되어 보인다면, Visualizer에 이미지를 전달할 때 BGR을 RGB로 뒤집는 것을 잊었기 때문입니다.
    CPU 대 GPU (CPU vs GPU): draw_instance_predictions 메서드는 인스턴스가 CPU 메모리에 있어야 합니다. 시각화하기 전에 항상 출력 인스턴스에 **.to("cpu")**를 호출해야 합니다.
    특정 ColorMode를 구성하거나 사용자 지정 데이터셋을 시각화하는 데 도움이 필요하신가요?
    """
    mode_name = VIZ_MODES[viz_mode]
    
    if viz_mode == 3:  # 🌟 Contour Visualization 모드 (오리지널 이미지 위에 윤곽선만)
        # 1. 원본 이미지 복사본 (copy()로 메모리 연속성 확보)
        contour_img_bgr = original_img_bgr.copy()
        
        # 2. Panoptic Segmentation 마스크 추출 (CPU로 이동 후 NumPy 변환)
        panoptic_seg_tensor = outputs["panoptic_seg"][0] 
        panoptic_seg = panoptic_seg_tensor.to("cpu").numpy()
        segments_info = outputs["panoptic_seg"][1]
        
        # 3. 세그먼트 정보를 딕셔너리로 변환 (id -> segment_info)
        seg_info_dict = {seg['id']: seg for seg in segments_info}
        
        # 4. category_id 기반 색상 생성 함수
        def get_color_from_category_id(category_id):
            """category_id를 기반으로 고유한 색상을 생성합니다."""
            # HSV 색상 공간을 사용하여 각 category_id마다 서로 다른 색상 생성
            hue = int((category_id * 137.5) % 180)  # 137.5도 간격으로 색상 배치
            hsv = np.uint8([[[hue, 255, 255]]])
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
            return tuple(map(int, bgr))  # (B, G, R) 형식
        
        # 5. 클래스 이름 가져오기 함수
        def get_class_name_from_segment(seg_info, metadata):
            """세그먼트 정보에서 클래스 이름을 가져옵니다."""
            cat_id = seg_info.get('category_id', -1)
            is_thing = seg_info.get('isthing', False)
            
            class_name = None
            if is_thing and hasattr(metadata, 'thing_classes'):
                if hasattr(metadata, 'thing_class_id'):
                    try:
                        idx = metadata.thing_class_id.index(cat_id)
                        class_name = metadata.thing_classes[idx]
                    except (ValueError, AttributeError):
                        class_name = None
                else:
                    if cat_id == 0 and len(metadata.thing_classes) > 0:
                        class_name = metadata.thing_classes[0]
                    elif 1 <= cat_id <= len(metadata.thing_classes):
                        class_name = metadata.thing_classes[cat_id - 1]
            elif not is_thing and hasattr(metadata, 'stuff_classes'):
                if hasattr(metadata, 'stuff_class_id'):
                    try:
                        idx = metadata.stuff_class_id.index(cat_id)
                        class_name = metadata.stuff_classes[idx]
                    except (ValueError, AttributeError):
                        class_name = None
                else:
                    if 0 <= cat_id < len(metadata.stuff_classes):
                        class_name = metadata.stuff_classes[cat_id]
            
            return class_name if class_name else f"id_{cat_id}"
        
        # 6. 각 세그먼트별로 윤곽선 그리기 및 클래스명 표시 (모든 Thing과 Stuff 포함, 클래스별 다른 색상)
        unique_ids = np.unique(panoptic_seg)
        font_scale = cv2.getFontScaleFromHeight(FONT, 12, 1)
        font_thickness = 1
        
        for seg_id in unique_ids:
            if seg_id == 0:  # 배경(0)은 건너뛰기
                continue
            
            # 세그먼트 정보 가져오기
            seg_info = seg_info_dict.get(seg_id, {})
            category_id = seg_info.get('category_id', seg_id)
            contour_color = get_color_from_category_id(category_id)
            
            # 각 세그먼트별 마스크 생성
            mask = (panoptic_seg == seg_id).astype(np.uint8) * 255
            # 윤곽선 찾기
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # 윤곽선 그리기 (클래스별 다른 색상)
            cv2.drawContours(contour_img_bgr, contours, -1, contour_color, CONTOUR_THICKNESS)
            
            # 중심점 계산 및 클래스명 표시
            if contours:
                # 가장 큰 윤곽선의 중심점 계산
                largest_contour = max(contours, key=cv2.contourArea)
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # 클래스 이름 가져오기
                    class_name = get_class_name_from_segment(seg_info, metadata)
                    
                    # category_id와 score 가져오기
                    cat_id = seg_info.get('category_id', category_id)
                    score = seg_info.get('score', None)
                    
                    # id:정확도 텍스트 생성
                    if score is not None and score > 0:
                        id_score_text = f"id:{cat_id} {score:.2f}"
                    else:
                        id_score_text = f"id:{cat_id}"
                    
                    # Thing 여부 확인하여 배경 색상 결정
                    is_thing = seg_info.get('isthing', False)
                    bg_color = (255, 0, 0) if is_thing else (0, 0, 0)  # Thing: 파란색, Stuff: 검정색
                    
                    # 텍스트 크기 계산 (두 줄 모두 고려)
                    (text_w1, text_h1), _ = cv2.getTextSize(class_name, FONT, font_scale, font_thickness)
                    (text_w2, text_h2), _ = cv2.getTextSize(id_score_text, FONT, font_scale, font_thickness)
                    max_text_w = max(text_w1, text_w2)
                    total_text_h = text_h1 + text_h2 + 5  # 두 줄 간격 5px
                    
                    # 텍스트 배경 (반투명 사각형)
                    bg_y1 = cy - total_text_h - 2
                    bg_y2 = cy + 2
                    cv2.rectangle(contour_img_bgr, (cx - max_text_w//2 - 2, bg_y1), 
                                (cx + max_text_w//2 + 2, bg_y2), bg_color, -1)
                    cv2.rectangle(contour_img_bgr, (cx - max_text_w//2 - 2, bg_y1), 
                                (cx + max_text_w//2 + 2, bg_y2), contour_color, 1)
                    
                    # 첫 번째 줄: 클래스명 텍스트 표시
                    cv2.putText(contour_img_bgr, class_name, (cx - text_w1//2, cy - text_h2 - 5), 
                              FONT, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
                    
                    # 두 번째 줄: id:정확도 텍스트 표시
                    cv2.putText(contour_img_bgr, id_score_text, (cx - text_w2//2, cy), 
                              FONT, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
        
        return contour_img_bgr, mode_name
        
    # Visualizer를 사용하는 모드들
    if viz_mode == 0:  # Panoptic Segmentation
        panoptic_seg, segments_info = outputs["panoptic_seg"]
        v = Visualizer(original_img_bgr[:, :, ::-1], metadata, scale=viz_scale, instance_mode=ColorMode.IMAGE)
        out = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)
        
    elif viz_mode == 1: # Instance Segmentation
        # outputs["instances"] 사용 (Thing만 포함)
        if "instances" in outputs:
            v = Visualizer(original_img_bgr[:, :, ::-1], metadata, scale=viz_scale, instance_mode=ColorMode.IMAGE)
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        else:
            # instances가 없으면 panoptic_seg 사용
            panoptic_seg, segments_info = outputs["panoptic_seg"]
            v = Visualizer(original_img_bgr[:, :, ::-1], metadata, scale=viz_scale, instance_mode=ColorMode.IMAGE)
            out = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)
        
    elif viz_mode == 2: # Semantic Segmentation
        # outputs["sem_seg"] 사용
        if "sem_seg" in outputs:
            sem_seg_tensor = outputs["sem_seg"].to("cpu")
            # sem_seg가 (num_classes, H, W) 형태인 경우 argmax로 (H, W)로 변환
            if len(sem_seg_tensor.shape) == 3:
                sem_seg_tensor = sem_seg_tensor.argmax(dim=0)
            v = Visualizer(original_img_bgr[:, :, ::-1], metadata, scale=viz_scale, instance_mode=ColorMode.IMAGE)
            out = v.draw_sem_seg(sem_seg_tensor)
        else:
            # sem_seg가 없으면 panoptic_seg에서 semantic 부분 추출
            panoptic_seg, segments_info = outputs["panoptic_seg"]
            # panoptic_seg를 semantic segmentation으로 변환 (category_id만 사용)
            panoptic_seg_np = panoptic_seg.to("cpu").numpy()
            seg_info_dict = {seg['id']: seg for seg in segments_info}
            sem_seg = np.zeros_like(panoptic_seg_np, dtype=np.uint8)
            for seg_id in np.unique(panoptic_seg_np):
                if seg_id == 0:
                    continue
                if seg_id in seg_info_dict:
                    cat_id = seg_info_dict[seg_id].get('category_id', 0)
                    sem_seg[panoptic_seg_np == seg_id] = cat_id
            v = Visualizer(original_img_bgr[:, :, ::-1], metadata, scale=viz_scale, instance_mode=ColorMode.IMAGE)
            out = v.draw_sem_seg(torch.from_numpy(sem_seg))
        
    # Visualizer 결과 이미지를 BGR로 변환하며, copy()를 통해 메모리 연속성 확보
    return out.get_image()[:, :, ::-1].copy(), mode_name

def overlay_window_info(img, file_info, mode_name, inference_time):
    """
    이미지에 파일 정보, 모드, 추론 시간을 오버레이하여 표시합니다.
    원본 이미지를 수정하지 않고 복사본에 오버레이를 추가합니다.
    폰트는 마지막에 추가되므로 해상도나 스케일 계산 없이 고정값을 사용합니다.
    """
    # 원본 이미지 복사 (BGR 형식 보장)
    img_copy = img.copy()
    h, w = img_copy.shape[:2]
    
    # 폰트 설정 고정값 (이미지 해상도나 스케일과 무관)
    font_thickness = 1
    # 원하는 픽셀 높이(12px)로 font_scale 계산
    font_scale = cv2.getFontScaleFromHeight(FONT, 12, font_thickness)
    y_pos = 30
    margin_x = 10
    
    # 1. 좌측 상단: 파일 정보
    text_info = file_info
    cv2.putText(img_copy, text_info, (margin_x, y_pos), FONT, font_scale, TEXT_COLOR, font_thickness, cv2.LINE_AA)

    # 2. 중앙 상단: 모드 이름
    text_mode = f"MODE: {mode_name}"
    (text_w, _), _ = cv2.getTextSize(text_mode, FONT, font_scale, font_thickness)
    center_x = (w - text_w) // 2
    cv2.putText(img_copy, text_mode, (center_x, y_pos), FONT, font_scale, TEXT_COLOR, font_thickness, cv2.LINE_AA)

    # 3. 우측 상단: 추론 시간
    text_time = f"Time: {inference_time:.4f}s"
    (text_w, _), _ = cv2.getTextSize(text_time, FONT, font_scale, font_thickness)
    right_x = w - text_w - margin_x
    cv2.putText(img_copy, text_time, (right_x, y_pos), FONT, font_scale, TEXT_COLOR, font_thickness, cv2.LINE_AA)
    
    return img_copy

def show_result_in_window(output_img, file_info=None, mode_name=None, inference_time=None):
    """
    윈도우 창 크기만 높이 800px로 조절하여 이미지를 출력합니다.
    오버레이 정보가 제공되면 imshow 전에 처리합니다.
    """
    if output_img is None:
        return

    # 오버레이 정보가 제공되면 imshow 전에 처리
    if file_info is not None and mode_name is not None and inference_time is not None:
        output_img = overlay_window_info(output_img, file_info, mode_name, inference_time)

    h, w = output_img.shape[:2]
    
    # 비율 유지하며 윈도우 너비 계산 (높이 800px 기준)
    aspect_ratio = w / h
    target_width = int(TARGET_WIN_HEIGHT * aspect_ratio)

    # 윈도우 설정 및 크기 조절
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, target_width, TARGET_WIN_HEIGHT)
    
    cv2.imshow(WINDOW_NAME, output_img)

# ====================================================
# 3. 메인 실행 함수
# ====================================================

def main():
    """프로그램의 메인 실행 루프입니다."""
    try:
        # 모델 및 파일 로드
        predictor, metadata = setup_model() # [정상 작동 확인]
        image_files = get_image_files(IMAGE_DIR)
    except Exception as e:
        print(f"\n⛔ 초기화 오류: {e}")
        return

    idx = 0                       # 현재 이미지 인덱스
    viz_mode = 0                  # 현재 시각화 모드 (0: Panoptic Seg.)
    total_imgs = len(image_files) # 전체 이미지 개수
    
    # [수정된 변수] 추론 결과, 원본 이미지, 추론 시간을 저장할 변수
    current_outputs = None      # Detectron2 추론 결과
    current_original_img = None # 현재 불러온 원본 이미지 (BGR)
    current_inference_time = 0.0 # 현재 이미지의 추론 시간 (초)

    # 사용자 안내 메시지 출력
    print("\n" + "="*50)
    print("🎮 조작 방법:")
    print("   [A]/[←]: 이전 이미지, [D]/[→]: 다음 이미지")
    print(f"   [S]: 시각화 모드 전환 (현재: {VIZ_MODES[viz_mode]})")
    print("   [Q]/[ESC]/[X 버튼]: 종료")
    print("="*50)

    # OpenCV 윈도우 생성 (루프 외부에서 한 번만 생성)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL) 

    # 이미지 로드 및 추론 함수 (추론 시간 저장)
    def load_and_infer_current_image():
        nonlocal current_outputs, current_original_img, current_inference_time 
        
        current_img_path = image_files[idx]
        filename = os.path.basename(current_img_path)
        
        print(f"\n📂 [{idx+1}/{total_imgs}] {filename} 로드 중...")
        current_original_img = cv2.imread(current_img_path)
        if current_original_img is None:
            print(f"❌ 이미지 로드 실패: {filename}.")
            return False # 로드 실패 시
            
        # 추론 수행 및 시간 측정
        start_time = time.time()
        current_outputs = predictor(current_original_img)
        current_inference_time = time.time() - start_time # 추론 시간 저장
        
        print(f"   ✓ 추론 완료 (Time: {current_inference_time:.4f}s)")
        
        # 디버그: panoptic_seg와 segments_info 출력
        if "panoptic_seg" in current_outputs:
            panoptic_seg, segments_info = current_outputs["panoptic_seg"]
            print(f"\n[DEBUG] panoptic_seg:")
            print(f"  - type: {type(panoptic_seg)}")
            print(f"  - shape: {panoptic_seg.shape if hasattr(panoptic_seg, 'shape') else 'N/A'}")
            print(f"  - dtype: {panoptic_seg.dtype if hasattr(panoptic_seg, 'dtype') else 'N/A'}")
            print(f"  - device: {panoptic_seg.device if hasattr(panoptic_seg, 'device') else 'N/A'}")
            print(f"  - unique values count: {len(torch.unique(panoptic_seg)) if hasattr(torch, 'unique') else 'N/A'}")
            
            print(f"\n[DEBUG] segments_info:")
            print(f"  - type: {type(segments_info)}")
            print(f"  - length: {len(segments_info)}")
            if len(segments_info) > 0:
                print(f"  - sample (first 3):")
                for i, seg in enumerate(segments_info[:3]):
                    print(f"    [{i}] {seg}")
                if len(segments_info) > 3:
                    print(f"    ... (total {len(segments_info)} segments)")
        
        return True # 로드 및 추론 성공 시

    # 초기 이미지 로드 및 추론 시도
    if not load_and_infer_current_image():
        print("초기 이미지 로드 및 추론에 실패했습니다. 프로그램을 종료합니다.")
        cv2.destroyAllWindows()
        return

    # 메인 루프
    while True:
        # 현재 파일 정보 구성
        filename = os.path.basename(image_files[idx])
        file_info = f"[{idx+1}/{total_imgs}] {filename}"

        # Visualizer 배율 동적 계산
        h_ori, _ = current_original_img.shape[:2]
        viz_scale = SCALE_THRESHOLD / h_ori if h_ori <= SCALE_THRESHOLD else 1.0

        # 처리 순서: 1) 추론 (이미 완료) → 2) 시각화 → 3) 오버레이 → 4) 표시
        # 1. 시각화 이미지 생성 (저장된 current_outputs와 current_original_img 사용)
        final_img_viz, mode_name_display = visualize_output_by_mode(
            current_original_img, current_outputs, metadata, viz_scale, viz_mode
        )
        
        # 2. 윈도우에 결과 표시 (오버레이는 show_result_in_window 내부에서 imshow 전에 처리)
        #    오버레이는 시각화된 이미지의 복사본에 텍스트만 추가합니다.
        if final_img_viz is not None:
            show_result_in_window(final_img_viz, file_info, mode_name_display, current_inference_time)
            
        # 4. 윈도우 닫기(X) 버튼 감지 및 키 입력 처리
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            print("\n👋 윈도우 닫기(X) 버튼으로 인해 프로그램을 종료합니다.")
            break
        
        key = cv2.waitKey(0) & 0xFF  # 키 입력 받기 (하위 8비트만 사용)

        # S 키: 시각화 모드 전환 (모드만 변경하고 즉시 재렌더링)
        if key == ord('s'):
            viz_mode = (viz_mode + 1) % len(VIZ_MODES)
            print(f"\n✨ 모드 전환: {VIZ_MODES[viz_mode]}")
            continue  # 루프 처음으로 돌아가서 현재 모드로 재렌더링

        # Q, ESC 종료 로직
        elif key == 27 or key == ord('q'):
            print("\n👋 키 입력으로 프로그램을 종료합니다.")
            break
            
        # A, D 또는 화살표 키: 다음/이전 파일 로드 및 추론 (마지막 모드 유지)
        
        # A 또는 Left Arrow (이전)
        elif key == ord('a') or key == 0x250000:  # 0x250000 = 왼쪽 화살표
            idx = (idx - 1 + total_imgs) % total_imgs
            if not load_and_infer_current_image():  # 새 이미지 로드 및 추론
                idx = (idx + 1) % total_imgs  # 실패 시 인덱스 되돌림
            # 추론 완료 후 루프 처음으로 돌아가서 현재 viz_mode로 표시
            continue
            
        # D 또는 Right Arrow (다음)
        elif key == ord('d') or key == 0x270000:  # 0x270000 = 오른쪽 화살표
            idx = (idx + 1) % total_imgs
            if not load_and_infer_current_image():  # 새 이미지 로드 및 추론
                idx = (idx - 1 + total_imgs) % total_imgs  # 실패 시 인덱스 되돌림
            # 추론 완료 후 루프 처음으로 돌아가서 현재 viz_mode로 표시
            continue
            
        # 다른 키 입력은 무시하고 다시 대기
        else:
            continue
            
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()