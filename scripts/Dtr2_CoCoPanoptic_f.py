"""
Detectron2 Panoptic Segmentation - Interactive Visualization Tool
COCO Panoptic Segmentation (panoptic_fpn_R_50_3x) + 신뢰도 표시

프로그램 개요:
    이 프로그램은 Detectron2의 Panoptic FPN 모델을 사용하여 이미지의 Panoptic Segmentation을 수행하고
    인터랙티브하게 결과를 시각화하는 도구입니다.
    
    주요 기능:
    1. 이미지 디렉토리에서 JPG 파일을 자동으로 로드
    2. Detectron2 Panoptic FPN 모델을 사용한 Panoptic Segmentation 추론 수행
    3. Thing(객체)과 Stuff(배경)를 구분하여 시각화
    4. 각 세그먼트의 클래스 이름과 신뢰도 점수 표시
    5. 키보드 입력(A/D 또는 화살표 키)으로 이미지 간 이동
    6. 추론 시간 및 Thing/Stuff 개수 정보 표시
    
    사용 모델:
    - 모델: COCO-PanopticSegmentation/panoptic_fpn_R_101_3x
    - 데이터셋: COCO (133개 클래스: 80 Thing + 53 Stuff)
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
    - 각 세그먼트 중심에 클래스 이름과 신뢰도 표시

================================================================================
리팩토링 작업 내역 (OneFormer → Detectron2)
================================================================================

1. 모델 변경:
   - 기존: OneFormer (shi-labs/oneformer_ade20k_swin_large)
   - 변경: Detectron2 Panoptic FPN (COCO-PanopticSegmentation/panoptic_fpn_R_50_3x)
   - 이유: Detectron2 기본 모델 사용 요청

2. 데이터셋 변경:
   - 기존: ADE20K (150개 클래스)
   - 변경: COCO (133개 클래스: 80 Thing + 53 Stuff)
   - 메타데이터: MetadataCatalog를 사용하여 COCO 데이터셋 정보 로드

3. 추론 방식 변경:
   - 기존: OneFormerProcessor + OneFormerForUniversalSegmentation
     * processor로 이미지 전처리
     * model(**inputs)로 추론
     * processor.post_process_panoptic_segmentation()로 후처리
   - 변경: Detectron2 DefaultPredictor
     * DefaultPredictor(img_bgr)로 직접 추론
     * outputs["panoptic_seg"]에서 (panoptic_seg, segments_info) 튜플 반환
     * panoptic_seg는 torch.Tensor, segments_info는 리스트

4. 출력 형식 변경:
   - 기존: OneFormer 형식
     * panoptic_result["segmentation"]: numpy 배열
     * panoptic_result["segments_info"]: 리스트 (label_id 사용)
   - 변경: Detectron2 형식
     * panoptic_seg: torch.Tensor (CPU로 변환 후 numpy 배열로 변환)
     * segments_info: 리스트 (category_id, isthing 필드 사용)

5. 클래스 이름 매핑 개선:
   - 문제: 초기에는 category_id가 인덱스로 표시됨 (예: "0", "1", "2")
   - 해결: Dtr2_panoptic._ex.py 방식으로 단순화
     a) 복잡한 id2label 딕셔너리 생성 제거
     b) 메타데이터에서 직접 클래스 이름 가져오기
        * isthing 필드를 먼저 확인하여 Thing/Stuff 구분
        * Thing: thing_class_id에서 category_id의 인덱스를 찾아 thing_classes에서 가져오기
        * Stuff: stuff_class_id에서 category_id의 인덱스를 찾아 stuff_classes에서 가져오기
        * thing_class_id/stuff_class_id가 없으면 인덱스 기반 매핑 (category_id - 1 for Thing, category_id for Stuff)
     c) category_id: 0은 Thing인 경우 "person" (thing_classes[0])으로 매핑
   - 결과: category_id가 실제 클래스 이름으로 표시됨 (예: "person", "car", "sky")

6. Thing/Stuff 구분 방식:
   - 기존: ADE20K의 is_thing_map 딕셔너리 사용
   - 변경: COCO segments_info의 'isthing' 필드 직접 사용
     * segments_info의 각 항목에 'isthing' 필드 포함
     * True면 Thing, False면 Stuff

7. 시각화 함수 수정:
   - visualize_cv2_all 함수에 metadata 파라미터 추가
   - category_id를 클래스 이름으로 변환하는 로직 개선
   - COCO 형식의 segments_info 처리 (category_id, isthing 필드)

8. 메타데이터 처리:
   - COCO 데이터셋 메타데이터 자동 로드
   - 데이터셋이 등록되지 않은 경우 cfg에서 가져오기
   - stuff_classes, thing_classes, stuff_class_id, thing_class_id 모두 처리

================================================================================
추가 작업 내역
================================================================================

9. Stuff 세그먼트 신뢰도 표시 문제 해결:
   - 문제: Stuff 세그먼트의 신뢰도가 0.00으로 표시됨
   - 원인: COCO panoptic segmentation에서 Stuff는 semantic segmentation으로 처리되어
           신뢰도 점수가 없거나 0임 (Thing만 instance segmentation으로 신뢰도 점수 제공)
   - 해결:
     a) score 기본값을 0.0에서 None으로 변경
     b) Thing인 경우: 신뢰도가 있고 0보다 크면 "[category_id]0.95" 형식으로 표시
     c) Stuff인 경우: 신뢰도 대신 클래스 ID만 "[category_id]" 형식으로 표시
   - 결과: Stuff는 신뢰도 없이 클래스 ID만 표시, Thing은 신뢰도 점수 표시

10. 클래스 이름 매핑 단순화 (Dtr2_panoptic._ex.py 방식 적용):
    - 문제: 복잡한 id2label 딕셔너리 생성으로 인한 매핑 오류 (category_id: 0이 "things"로 표시됨)
    - 원인: id2label 딕셔너리에서 Thing과 Stuff가 같은 ID를 사용할 수 있어 충돌 발생
    - 해결: Dtr2_panoptic._ex.py 방식으로 단순화
      a) id2label, is_thing_map 딕셔너리 생성 제거
      b) visualize_cv2_all 함수 단순화
         * id2label, is_thing_map 파라미터 제거
         * 메타데이터에서 직접 클래스 이름 가져오기
         * isthing 필드를 먼저 확인하여 Thing/Stuff 구분
      c) 추론 방식: panoptic_seg, segments_info = predictor(im)["panoptic_seg"] (Tutorial 방식)
    - 결과: 코드가 단순해지고 클래스 이름 매핑이 정확해짐
    - 참고: COCO panoptic segmentation 공식 사이트
            * GitHub: https://github.com/cocodataset/panopticapi
            * 공식 웹사이트: https://cocodataset.org/#panoptic-2021

================================================================================
기술적 세부 사항
================================================================================

- COCO Panoptic Segmentation의 category_id는 COCO의 원본 카테고리 ID를 사용
  * stuff_class_id와 thing_class_id를 통해 원본 ID를 클래스 이름에 매핑
  * Stuff: stuff_class_id의 값들이 COCO 원본 ID
  * Thing: thing_class_id의 값들이 COCO 원본 ID
- segments_info의 'isthing' 필드로 Thing/Stuff 구분 (가장 정확한 방법)
- 클래스 이름 매핑 실패 시 메타데이터에서 직접 가져오는 fallback 로직 구현
- Stuff 세그먼트는 신뢰도 점수가 없으므로 클래스 ID만 표시
- Thing 세그먼트는 신뢰도 점수가 있으면 함께 표시

================================================================================
id / class name 불일치 문제 정리 (2025-11-27)
================================================================================

1. 문제 현상
   - ADE20K 이미지에 COCO panoptic 모델(panoptic_fpn_R_101_3x)을 적용했을 때,
     사람이 보기엔 bed, chair, table, car처럼 보이는데
     라벨은 potted plant, cake, bed, bicycle 등 COCO panoptic 이름으로 표시되었다.
   - 특히 COCO 80-클래스 detection 기준으로 기억하고 있는 id 표(예: 56=chair, 59=bed, 60=dining table)와
     COCO Panoptic 133-클래스 id 표가 달라서, id는 맞는데 이름이 이상하게 느껴지는 문제가 있었다.

2. 원인 분석
   - Detectron2 panoptic 모델은 COCO Panoptic(133 클래스)용 category_id를 사용한다.
     * 예: 56=cake, 59=potted plant, 60=bed, 61=dining table, 24=backpack, 2=bicycle ...
   - 이 스크립트와 `_ex` 모두 `outputs["panoptic_seg"]`의 segments_info.category_id와
     MetadataCatalog의 thing_class_id / stuff_class_id를 그대로 사용하므로,
     "모델 기준"으로는 id→이름 매핑이 정확하다.
   - 사용자가 기대한 것은 COCO-80 detection 스타일의 이름(id=56→chair, 60→table, 2→car, 24→backpack, 61→toilet 등)이라
     두 개의 id/name 맵이 섞여 보인 것이 문제의 본질이었다.

3. 해결 방법
   - 추론 id(category_id)와 isthing 값은 그대로 유지하고, 시각화에 사용되는 "표시용 클래스 이름"만 별도 맵으로 보정한다.
   - 공통 함수 `get_class_name_from_segment(seg_info, metadata)`에서:
     a) 먼저 MetadataCatalog를 사용해 COCO Panoptic 공식 이름을 계산
        - Thing: thing_class_id / thing_classes
        - Stuff: stuff_class_id / stuff_classes
     b) 이름이 없으면 `id_<category_id>` 형태로 fallback
     c) 마지막에 `CUSTOM_CLASS_NAME_MAP[category_id]`가 있으면 그 값으로 한 번 더 오버라이드
   - 이렇게 하면:
     - id와 isthing은 항상 모델 출력 그대로 유지
     - 이름만 사용자 정의 표에 맞게 "chair", "table", "car", "backpack", "toilet" 등으로 보정된다.

4. 현재 매핑 규칙 요약
   - id는 항상 segments_info["category_id"] (COCO Panoptic 기준)를 그대로 표시한다.
   - 이름은 다음 우선순위로 결정된다.
     1) MetadataCatalog 기반 COCO Panoptic 공식 이름
     2) 없으면 "id_<category_id>"
     3) 마지막으로 CUSTOM_CLASS_NAME_MAP에 등록된 이름(사용자 정의 COCO-80 스타일 이름)으로 오버라이드
"""

import os  # 파일 시스템 경로 조작
import glob  # 파일 패턴 매칭
import cv2  # OpenCV 이미지 처리
import time  # 시간 측정
import numpy as np  # 수치 연산

# Detectron2 관련 임포트
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.logger import setup_logger
from detectron2.data import MetadataCatalog

# COCO → 사용자 정의 클래스 이름 오버라이드 맵
# 사용자가 원하는 COCO-80 스타일 이름으로 표시하기 위한 매핑
# (category_id: 원하는 표시 이름)
CUSTOM_CLASS_NAME_MAP = {
    24: "backpack",
    60: "table",
    56: "chair",
    61: "toilet",
    2: "car",
}


def get_class_name_from_segment(seg_info, metadata):
    """
    Dtr2_CoCOpanoptic._ex.py에서 사용한 것과 동일한 방식으로
    COCO panoptic의 segment 정보에서 클래스 이름을 가져옵니다.

    Args:
        seg_info (dict): segments_info의 개별 세그먼트 딕셔너리
        metadata: Detectron2 MetadataCatalog 객체

    Returns:
        str: 클래스 이름 (매핑 실패 시 "id_<category_id>" 형식)
    """
    cat_id = seg_info.get("category_id", -1)
    is_thing = seg_info.get("isthing", False)

    class_name = None
    if is_thing and hasattr(metadata, "thing_classes"):
        if hasattr(metadata, "thing_class_id"):
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
    elif (not is_thing) and hasattr(metadata, "stuff_classes"):
        if hasattr(metadata, "stuff_class_id"):
            try:
                idx = metadata.stuff_class_id.index(cat_id)
                class_name = metadata.stuff_classes[idx]
            except (ValueError, AttributeError):
                class_name = None
        else:
            if 0 <= cat_id < len(metadata.stuff_classes):
                class_name = metadata.stuff_classes[cat_id]

    # 기본 COCO 이름이 없으면 id_표기 사용
    class_name = class_name if class_name else f"id_{cat_id}"
    # 최종적으로 사용자 정의 이름으로 한 번 더 오버라이드
    return CUSTOM_CLASS_NAME_MAP.get(cat_id, class_name)


def visualize_cv2_all(img_bgr, seg_map, segments_info, filename, inference_time, metadata):
    """
    Panoptic Segmentation 결과를 OpenCV를 사용하여 시각화합니다.
    
    이 함수는 세그멘테이션 결과를 받아서 Thing과 Stuff를 구분하여 시각화하고,
    각 세그먼트의 클래스 이름, 신뢰도 점수, 통계 정보를 이미지에 오버레이합니다.
    
    Args:
        img_bgr (numpy.ndarray): 원본 이미지 (BGR 형식)
        seg_map (numpy.ndarray): 세그멘테이션 맵 (각 픽셀의 세그먼트 ID)
        segments_info (list): 각 세그먼트의 정보 딕셔너리 리스트 (id, category_id, isthing, score 등)
        filename (str): 현재 처리 중인 이미지 파일명
        inference_time (float): 추론에 소요된 시간(초)
        metadata: Detectron2 MetadataCatalog 객체 (클래스 이름 가져오기용)
    
    Returns:
        numpy.ndarray: 시각화된 이미지 (BGR 형식)
    
    처리 과정:
        1. 이미지를 800px 높이로 리사이즈 (비율 유지)
        2. Stuff 영역을 반투명 색상 오버레이로 표시
        3. Thing 영역을 외곽선으로 표시
        4. 각 세그먼트 중심에 클래스 이름과 신뢰도 점수 텍스트 추가
        5. 상단에 파일명, Thing/Stuff 개수, 추론 시간 정보 표시
        6. OpenCV 윈도우에 결과 이미지 표시
    """
    h, w = img_bgr.shape[:2]  # 이미지 높이와 너비 추출
    target_h = 800  # 목표 높이 설정
    target_w = int(w * target_h / h)  # 비율 유지하며 목표 너비 계산
    resized_orig = cv2.resize(img_bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)  # 원본 이미지 리사이즈
    
    overlay_stuff = np.zeros_like(resized_orig, dtype=np.uint8)  # Stuff 오버레이 초기화
    
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

    inst_info = {s["id"]: s for s in segments_info}  # 세그먼트 정보를 딕셔너리로 변환
    unique_ids = np.unique(seg_map)  # 고유 세그먼트 ID 추출
    
    # Stuff 그리기 (반투명 오버레이만 적용 – 라벨은 아래에서 contour 기반으로 한 번에 처리)
    for i, cid in enumerate(unique_ids):
        if cid not in inst_info:
            continue
        info = inst_info[cid]
        is_thing = info.get("isthing", False)
        if is_thing:
            continue

        mask = seg_map == cid
        if not np.any(mask):
            continue

        mask_resized = cv2.resize(
            mask.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST
        )

        b, g, r = get_color(i)
        overlay_stuff[mask_resized > 0] = (b, g, r)

    alpha = 120 / 255.0  # 투명도 설정
    blended = cv2.addWeighted(overlay_stuff, alpha, resized_orig, 1 - alpha, 0)  # Stuff 오버레이와 원본 이미지 블렌딩

    # Thing 그리기 (윤곽선만 – 라벨은 아래에서 contour 기반으로 한 번에 처리)
    for i, cid in enumerate(unique_ids):
        if cid not in inst_info:
            continue

        info = inst_info[cid]
        is_thing = info.get("isthing", False)
        if not is_thing:
            continue

        mask = seg_map == cid
        if not np.any(mask):
            continue

        mask_resized = cv2.resize(
            mask.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST
        )
        mask_resized = (mask_resized * 255).astype(np.uint8)

        b, g, r = get_color(i)
        contours, _ = cv2.findContours(
            mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(blended, contours, -1, (b, g, r), 2)

    # 라벨 + 신뢰도 텍스트 (Dtr2_CoCOpanoptic._ex.py Contour 모드와 동일한 스타일)
    font_scale = cv2.getFontScaleFromHeight(cv2.FONT_HERSHEY_SIMPLEX, 12, 1)
    font_thickness = 1

    for i, cid in enumerate(unique_ids):
        if cid not in inst_info:
            continue
        info = inst_info[cid]

        mask = seg_map == cid
        if not np.any(mask):
            continue

        mask_resized = cv2.resize(
            mask.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST
        )
        mask_resized = (mask_resized * 255).astype(np.uint8)

        contours, _ = cv2.findContours(
            mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue

        # 가장 큰 윤곽선의 중심점 계산
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # 클래스/ID/점수 정보
        cat_id = info.get("category_id", info.get("label_id", -1))
        is_thing = info.get("isthing", False)
        class_name = get_class_name_from_segment(info, metadata)
        score = info.get("score", None)

        # 디버깅 출력 (예제와 유사)
        print(
            f"[DEBUG] cid={cid}, category_id={cat_id}, isthing={is_thing}, class_name='{class_name}'"
        )

        if score is not None and score > 0:
            id_score_text = f"id:{cat_id} {score:.2f}"
        else:
            id_score_text = f"id:{cat_id}"

        # Thing/Stuff에 따라 배경색 결정
        bg_color = (255, 0, 0) if is_thing else (0, 0, 0)
        contour_color = get_color(i)

        # 텍스트 크기 계산
        (text_w1, text_h1), _ = cv2.getTextSize(
            class_name, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
        )
        (text_w2, text_h2), _ = cv2.getTextSize(
            id_score_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
        )
        max_text_w = max(text_w1, text_w2)
        total_text_h = text_h1 + text_h2 + 5

        # 텍스트 배경 박스
        bg_y1 = cy - total_text_h - 2
        bg_y2 = cy + 2
        cv2.rectangle(
            blended,
            (cx - max_text_w // 2 - 2, bg_y1),
            (cx + max_text_w // 2 + 2, bg_y2),
            bg_color,
            -1,
        )
        cv2.rectangle(
            blended,
            (cx - max_text_w // 2 - 2, bg_y1),
            (cx + max_text_w // 2 + 2, bg_y2),
            contour_color,
            1,
        )

        # 첫 줄: 클래스 이름
        cv2.putText(
            blended,
            class_name,
            (cx - text_w1 // 2, cy - text_h2 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            font_thickness,
            cv2.LINE_AA,
        )

        # 둘째 줄: id:score
        cv2.putText(
            blended,
            id_score_text,
            (cx - text_w2 // 2, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            font_thickness,
            cv2.LINE_AA,
        )

    # 상단 정보 (Tutorial 방식: segments_info의 'isthing' 필드 직접 사용)
    thing_count = sum(1 for s in segments_info if s.get('isthing', False))  # Thing 개수 계산
    stuff_count = len(segments_info) - thing_count  # Stuff 개수 계산
    
    info_text = f"{filename} | T:{thing_count} S:{stuff_count}"  # 정보 텍스트 생성
    cv2.putText(blended, info_text, (10, 20),  # 상단 왼쪽에 정보 표시
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), 1, cv2.LINE_AA)
    
    time_text = f"Inference: {inference_time:.4f}s"  # 추론 시간 텍스트 생성
    (tw, th), _ = cv2.getTextSize(time_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)  # 텍스트 크기 계산
    cv2.putText(blended, time_text, (target_w - tw - 10, 20),  # 상단 오른쪽에 시간 표시
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), 1, cv2.LINE_AA)

    window_name = "Detectron2 - Panoptic Segmentation"  # 윈도우 이름 설정
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # 윈도우 생성
    cv2.resizeWindow(window_name, target_w, target_h)  # 윈도우 크기 조정
    cv2.moveWindow(window_name, 0, 0)  # 윈도우 위치 이동
    cv2.imshow(window_name, blended)  # 이미지 표시
    return blended  # 블렌딩된 이미지 반환

# Detectron2 로거 설정
setup_logger()

# 이미지 리스트
IMAGE_DIR = r"D:/git/detectron2/ade20k_consistency/original_ade20k"  # 이미지 디렉토리 경로
image_files = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.jpg")))  # JPG 파일 목록 정렬
if not image_files:  # 이미지가 없으면
    raise FileNotFoundError(f"'{IMAGE_DIR}'에 이미지가 없습니다.")  # 에러 발생
print(f"📂 {len(image_files)}개 이미지")  # 이미지 개수 출력

# Detectron2 설정 및 모델 로드 (예제 스크립트와 동일한 R_101_3x 모델 사용)
config_file = "COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"
print(f"🔧 모델 로드: {config_file}")  # 모델 로드 메시지 출력

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(config_file))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # 추론 시 임계값 설정
predictor = DefaultPredictor(cfg)

# COCO 데이터셋 메타데이터 (Dtr2_CoCOpanoptic._ex.py의 setup_model과 동일한 방식)
metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])


def run_inference(idx):
    """
    지정된 인덱스의 이미지에 대해 Panoptic Segmentation 추론을 수행합니다.
    
    이 함수는 이미지 파일을 로드하고, Detectron2 모델을 사용하여 세그멘테이션을 수행한 후,
    결과를 시각화 함수로 전달합니다.
    
    Args:
        idx (int): image_files 리스트에서 처리할 이미지의 인덱스
    
    처리 과정:
        1. 이미지 파일 경로 가져오기 및 로드
        2. Detectron2 DefaultPredictor로 추론 수행
        3. 추론 시간 측정
        4. Panoptic Segmentation 결과 추출
        5. Thing/Stuff 개수 계산 및 디버그 정보 출력
        6. 시각화 함수 호출
    
    출력:
        - 현재 이미지 정보 (인덱스/총 개수, 파일명)
        - Thing/Stuff 개수 디버그 정보
        - 추론 완료 메시지 및 소요 시간
        - 시각화된 결과 이미지 (OpenCV 윈도우)
    
    예외:
        이미지 로드 실패 시 에러 메시지 출력 후 함수 종료
    """
    img_path = image_files[idx]  # 이미지 경로 가져오기
    filename = os.path.basename(img_path)  # 파일명 추출
    
    img_bgr = cv2.imread(img_path)  # 이미지 읽기 (BGR 형식)
    if img_bgr is None:  # 이미지 로드 실패 시
        print(f"❌ 이미지 로드 실패: {img_path}")  # 에러 메시지 출력
        return
    
    print(f"\n📂 [{idx+1}/{len(image_files)}] {filename}")  # 현재 이미지 정보 출력
    
    start_time = time.time()  # 시작 시간 기록
    outputs = predictor(img_bgr)  # Detectron2 모델 추론 수행
    inference_time = time.time() - start_time  # 추론 시간 계산
    
    # Panoptic segmentation 결과 추출 (Tutorial 방식)
    # Tutorial: panoptic_seg, segments_info = predictor(im)["panoptic_seg"]
    panoptic_seg, segments_info = outputs["panoptic_seg"]
    seg_map = panoptic_seg.cpu().numpy()  # 세그멘테이션 맵을 numpy 배열로 변환
    
    # COCO 형식의 segments_info는 'isthing' 필드를 포함
    # 'isthing' 필드를 사용하여 Thing/Stuff 구분
    thing_count = sum(1 for s in segments_info if s.get('isthing', False))  # Thing 개수 계산
    print(f"DEBUG - Thing: {thing_count}, Stuff: {len(segments_info) - thing_count}")  # 디버그 정보 출력
    
    # 디버깅: segments_info 전체를 id:name 형식으로 출력 (예제 스크립트와 동일한 방식)
    if len(segments_info) > 0:
        print(f"\n[DEBUG] segments_info (id:name):")
        print(f"  - type: {type(segments_info)}")
        print(f"  - length: {len(segments_info)}")
        for i, seg in enumerate(segments_info):
            cat_id = seg.get("category_id", seg.get("label_id", -1))
            is_thing = seg.get("isthing", False)
            class_name = get_class_name_from_segment(seg, metadata)
            print(f"    [{i}] id:{cat_id} name:{class_name} (isthing={is_thing})")

    print(f"✓ 추론 완료 ({inference_time:.4f}초)")  # 완료 메시지 출력
    
    # 시각화 함수 호출 (Dtr2_panoptic._ex.py 방식: 메타데이터만 전달)
    visualize_cv2_all(img_bgr, seg_map, segments_info, filename, inference_time, metadata)  # 결과 시각화

def main():
    """
    프로그램의 메인 실행 루프입니다.
    
    이 함수는 프로그램의 진입점으로, 첫 번째 이미지를 자동으로 로드하고
    키보드 입력을 받아 이미지 간 이동을 처리하는 인터랙티브 루프를 실행합니다.
    
    실행 흐름:
        1. 첫 번째 이미지(인덱스 0) 자동 추론 및 시각화
        2. 사용법 안내 메시지 출력
        3. 무한 루프로 키보드 입력 대기
        4. 입력에 따라 이전/다음 이미지로 이동하거나 프로그램 종료
    
    키보드 입력:
        - 'A' 또는 왼쪽 화살표 (0x250000): 이전 이미지로 이동
        - 'D' 또는 오른쪽 화살표 (0x270000): 다음 이미지로 이동
        - 'Q': 프로그램 종료
    
    특징:
        - 이미지 인덱스는 순환 구조 (마지막 이미지에서 다음 = 첫 이미지)
        - 각 이미지 이동 시 자동으로 추론 및 시각화 수행
        - 종료 시 모든 OpenCV 윈도우 자동 닫기
    """
    cur_idx = 0  # 현재 이미지 인덱스 초기화
    run_inference(cur_idx)  # 첫 번째 이미지 추론

    print("\n키: A/← (이전), D/→ (다음), Q (종료)")  # 사용법 안내 출력

    while True:  # 무한 루프
        key = cv2.waitKey(0) & 0xFF  # 키 입력 대기
        if key == ord('a') or key == 0x250000:  # 'a' 또는 왼쪽 화살표 키
            cur_idx = (cur_idx - 1) % len(image_files)  # 이전 이미지로 이동
            run_inference(cur_idx)  # 추론 수행
        elif key == ord('d') or key == 0x270000:  # 'd' 또는 오른쪽 화살표 키
            cur_idx = (cur_idx + 1) % len(image_files)  # 다음 이미지로 이동
            run_inference(cur_idx)  # 추론 수행
        elif key == ord('q'):  # 'q' 키
            print("\n👋 종료")  # 종료 메시지 출력
            break  # 루프 종료
    cv2.destroyAllWindows()  # 모든 윈도우 닫기

if __name__ == "__main__":  # 스크립트가 직접 실행될 때
    main()  # 메인 함수 실행
