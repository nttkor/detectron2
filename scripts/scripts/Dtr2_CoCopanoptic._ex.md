# Detectron2 COCO Panoptic Segmentation 분석 문서

## 📋 목차
1. [프로젝트 개요](#프로젝트-개요)
2. [Detectron2 Panoptic Segmentation 기본 개념](#detectron2-panoptic-segmentation-기본-개념)
3. [출력 데이터 구조 분석](#출력-데이터-구조-분석)
4. [시각화 모드 구현](#시각화-모드-구현)
5. [주요 문제 해결 과정](#주요-문제-해결-과정)
6. [기술적 세부사항](#기술적-세부사항)
7. [최종 구현 특징](#최종-구현-특징)

---

## 프로젝트 개요

### 목적
Detectron2의 COCO Panoptic Segmentation 모델을 사용하여 이미지를 추론하고, 4가지 시각화 모드로 결과를 표시하는 인터랙티브 뷰어 개발

### 사용 모델
- **모델**: `COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml`
- **데이터셋**: COCO (133개 클래스: 80 Thing + 53 Stuff)
- **프레임워크**: Detectron2

### 주요 기능
- 4가지 시각화 모드 제공
- 인터랙티브 이미지 탐색 (A/D 키)
- 실시간 모드 전환 (S 키)
- 추론 결과 재사용으로 빠른 모드 전환

---

## Detectron2 Panoptic Segmentation 기본 개념

### Panoptic Segmentation이란?
Panoptic Segmentation은 **Semantic Segmentation**과 **Instance Segmentation**을 통합한 작업입니다.

- **Semantic Segmentation**: 픽셀 단위로 클래스 분류 (예: "하늘", "도로")
- **Instance Segmentation**: 객체 단위로 분할 및 분류 (예: "사람1", "사람2")
- **Panoptic Segmentation**: 위 두 가지를 통합하여 모든 픽셀을 Thing 또는 Stuff로 분류

### Thing vs Stuff
- **Thing**: 셀 수 있는 객체 (예: person, car, bicycle)
  - 각 인스턴스가 고유한 ID를 가짐
  - 신뢰도 점수(score) 제공
  - Bounding box 정보 포함 가능

- **Stuff**: 형태가 없는 영역 (예: sky, road, wall)
  - 인스턴스 구분 없음
  - 신뢰도 점수 없음
  - Bounding box 정보 없음

---

## 출력 데이터 구조 분석

### 1. 추론 결과 구조

```python
outputs = predictor(img_bgr)
# outputs는 딕셔너리 형태:
# {
#     "panoptic_seg": (panoptic_seg, segments_info),
#     "instances": Instances 객체 (선택적),
#     "sem_seg": torch.Tensor (선택적)
# }
```

### 2. panoptic_seg 구조

**타입**: `torch.Tensor`  
**Shape**: `(H, W)` - 이미지 높이 × 너비  
**Dtype**: `torch.int32`  
**Device**: `cuda:0` (GPU 사용 시)

**내용**:
- 각 픽셀에 세그먼트 ID가 저장됨
- `panoptic_seg[y, x] = seg_id` → 해당 픽셀이 `seg_id` 세그먼트에 속함
- `0`은 배경을 의미

**예시**:
```python
# panoptic_seg 예시 (5×5 이미지)
# [[0, 0, 1, 1, 1],   # 0: 배경, 1: 세그먼트 ID 1
#  [0, 0, 1, 1, 2],   # 2: 세그먼트 ID 2
#  [3, 3, 3, 2, 2],   # 3: 세그먼트 ID 3
#  [3, 3, 3, 0, 0],   # ...
#  [0, 0, 0, 0, 0]]
```

**중요**: `panoptic_seg`는 **마스크 픽셀 데이터**를 포함합니다. Visualizer 함수들은 이 텐서에서 마스크를 추출하여 사용합니다.

### 3. segments_info 구조

**타입**: `list` (딕셔너리 리스트)  
**길이**: 세그먼트 개수 (배경 제외)

**각 세그먼트 정보 (딕셔너리)**:

#### Thing 세그먼트
```python
{
    'id': 1,                    # panoptic_seg의 세그먼트 ID
    'isthing': True,            # Thing 여부
    'category_id': 0,           # COCO 원본 카테고리 ID
    'score': 0.9961739182472229, # 신뢰도 점수 (0~1)
    'instance_id': 0            # 인스턴스 ID
}
```

#### Stuff 세그먼트
```python
{
    'id': 1,                    # panoptic_seg의 세그먼트 ID
    'isthing': False,           # Thing 여부
    'category_id': 37,          # COCO 원본 카테고리 ID
    'area': 138266              # 영역 크기 (픽셀 수)
}
```

**중요 필드 설명**:
- `id`: `panoptic_seg`에서 사용하는 세그먼트 ID
- `category_id`: COCO 데이터셋의 원본 카테고리 ID (0~132)
- `isthing`: Thing/Stuff 구분 (가장 정확한 방법)
- `score`: Thing만 가지고 있음 (Stuff는 없음)
- `area`: Stuff만 가지고 있음 (Thing은 없음)

**⚠️ 주의사항**:
- `segments_info`에는 **bbox 정보가 없습니다**
- `segments_info`에는 **마스크 픽셀 데이터가 없습니다** (메타정보만)
- 실제 마스크는 `panoptic_seg`에서 추출: `mask = (panoptic_seg == seg_id)`

### 4. 실제 출력 예시

```
[DEBUG] panoptic_seg:
  - type: <class 'torch.Tensor'>
  - shape: torch.Size([2200, 1650])
  - dtype: torch.int32
  - device: cuda:0
  - unique values count: 3

[DEBUG] segments_info:
  - type: <class 'list'>
  - length: 3
  - sample (first 3):
    [0] {'id': 1, 'isthing': False, 'category_id': 37, 'area': 138266}
    [1] {'id': 2, 'isthing': False, 'category_id': 40, 'area': 1192278}
    [2] {'id': 3, 'isthing': False, 'category_id': 50, 'area': 2299456}
```

---

## 시각화 모드 구현

### 모드 0: Panoptic Segmentation
**함수**: `Visualizer.draw_panoptic_seg_predictions()`

**특징**:
- Thing과 Stuff 모두 표시
- 색상 오버레이로 마스크 표시
- 클래스명 자동 표시
- **Bbox 없음** (마스크만)

**코드**:
```python
panoptic_seg, segments_info = outputs["panoptic_seg"]
v = Visualizer(original_img_bgr[:, :, ::-1], metadata, scale=viz_scale, instance_mode=ColorMode.IMAGE)
out = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)
```

### 모드 1: Instance Segmentation
**함수**: `Visualizer.draw_instance_predictions()`

**특징**:
- Thing만 표시 (instances 객체 사용)
- **Bbox 포함** (instances.pred_boxes)
- 신뢰도 점수 표시
- instances가 없으면 panoptic_seg 사용 (fallback)

**코드**:
```python
if "instances" in outputs:
    v = Visualizer(original_img_bgr[:, :, ::-1], metadata, scale=viz_scale, instance_mode=ColorMode.IMAGE)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
else:
    # fallback: panoptic_seg 사용
    panoptic_seg, segments_info = outputs["panoptic_seg"]
    v = Visualizer(original_img_bgr[:, :, ::-1], metadata, scale=viz_scale, instance_mode=ColorMode.IMAGE)
    out = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)
```

**차이점**:
- `draw_instance_predictions()`: **bbox를 그림** (instances.pred_boxes 사용)
- `draw_panoptic_seg_predictions()`: **bbox를 그리지 않음** (마스크만)

### 모드 2: Semantic Segmentation
**함수**: `Visualizer.draw_sem_seg()`

**특징**:
- 픽셀 단위 클래스 분류만 표시
- 인스턴스 구분 없음
- Bbox 없음

**코드**:
```python
if "sem_seg" in outputs:
    sem_seg_tensor = outputs["sem_seg"].to("cpu")
    # sem_seg가 (num_classes, H, W) 형태인 경우 argmax로 (H, W)로 변환
    if len(sem_seg_tensor.shape) == 3:
        sem_seg_tensor = sem_seg_tensor.argmax(dim=0)
    v = Visualizer(original_img_bgr[:, :, ::-1], metadata, scale=viz_scale, instance_mode=ColorMode.IMAGE)
    out = v.draw_sem_seg(sem_seg_tensor)
else:
    # fallback: panoptic_seg에서 semantic 부분 추출
    # category_id만 사용하여 semantic segmentation 생성
    ...
```

**Shape 변환**:
- 입력: `(num_classes, H, W)` - 각 클래스별 확률/점수
- 변환: `argmax(dim=0)` → `(H, W)` - 각 픽셀의 클래스 ID

### 모드 3: Contour Visualization
**구현**: 직접 구현 (Detectron2 함수 사용 안 함)

**특징**:
- 원본 이미지 유지
- 각 세그먼트별 윤곽선만 표시
- 클래스별 다른 색상 (category_id 기반)
- 클래스명과 id:정확도 표시
- Thing은 파란색 배경, Stuff는 검정색 배경

**구현 과정**:
1. `panoptic_seg`에서 각 세그먼트 ID 추출
2. 각 세그먼트별로 마스크 생성: `mask = (panoptic_seg == seg_id)`
3. `cv2.findContours()`로 윤곽선 찾기
4. `cv2.drawContours()`로 윤곽선 그리기 (클래스별 색상)
5. 중심점 계산 및 클래스명 표시

**색상 생성**:
```python
def get_color_from_category_id(category_id):
    """category_id를 기반으로 고유한 색상을 생성합니다."""
    hue = int((category_id * 137.5) % 180)  # HSV 색상 공간, 137.5도 간격
    hsv = np.uint8([[[hue, 255, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return tuple(map(int, bgr))  # (B, G, R) 형식
```

---

## 주요 문제 해결 과정

### 문제 1: 클래스명이 인덱스로 표시됨

**증상**: 클래스명이 "class_13", "class_11"처럼 인덱스로 표시됨

**원인**:
- COCO의 `category_id`는 원본 카테고리 ID (0~132)
- 단순 인덱스 매핑으로는 올바른 클래스명을 찾을 수 없음
- Thing과 Stuff가 서로 다른 ID 체계를 사용

**해결**:
1. `isthing` 필드로 Thing/Stuff 구분
2. `thing_class_id` / `stuff_class_id`를 사용하여 원본 ID → 인덱스 매핑
3. 매핑된 인덱스로 `thing_classes` / `stuff_classes`에서 클래스명 가져오기

**코드**:
```python
def get_class_name_from_segment(seg_info, metadata):
    cat_id = seg_info.get('category_id', -1)
    is_thing = seg_info.get('isthing', False)
    
    if is_thing and hasattr(metadata, 'thing_classes'):
        if hasattr(metadata, 'thing_class_id'):
            try:
                idx = metadata.thing_class_id.index(cat_id)
                class_name = metadata.thing_classes[idx]
            except (ValueError, AttributeError):
                class_name = None
        else:
            # fallback: 인덱스 기반 매핑
            if cat_id == 0 and len(metadata.thing_classes) > 0:
                class_name = metadata.thing_classes[0]  # "person"
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
            # fallback: 인덱스 기반 매핑
            if 0 <= cat_id < len(metadata.stuff_classes):
                class_name = metadata.stuff_classes[cat_id]
    
    return class_name if class_name else f"id_{cat_id}"
```

### 문제 2: Stuff 세그먼트 신뢰도가 0.00으로 표시됨

**증상**: Stuff 세그먼트의 신뢰도가 항상 0.00으로 표시됨

**원인**:
- COCO panoptic segmentation에서 Stuff는 semantic segmentation으로 처리
- Stuff는 신뢰도 점수가 없음 (Thing만 instance segmentation으로 신뢰도 제공)

**해결**:
- Thing인 경우: 신뢰도가 있고 0보다 크면 표시
- Stuff인 경우: 신뢰도 대신 클래스 ID만 표시

### 문제 3: Instance/Contour 모드에서 모든 세그먼트가 표시되지 않음

**증상**: Instance 모드에서 Thing만 표시되고, Contour 모드에서 일부 세그먼트가 누락됨

**원인**:
- Instance 모드: `outputs["instances"]`는 Thing만 포함
- Contour 모드: 전체 이진 마스크로 처리하여 내부 경계 손실

**해결**:
- Instance 모드: `instances`가 없으면 `panoptic_seg` 사용 (fallback)
- Contour 모드: 각 세그먼트별로 개별적으로 윤곽선 그리기

**코드** (Contour 모드):
```python
# 각 세그먼트별로 윤곽선 그리기
unique_ids = np.unique(panoptic_seg)
for seg_id in unique_ids:
    if seg_id == 0:  # 배경 제외
        continue
    # 각 세그먼트별 마스크 생성
    mask = (panoptic_seg == seg_id).astype(np.uint8) * 255
    # 윤곽선 찾기
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 윤곽선 그리기 (클래스별 다른 색상)
    cv2.drawContours(contour_img_bgr, contours, -1, contour_color, CONTOUR_THICKNESS)
```

### 문제 4: 폰트 크기가 이미지마다 들쭉날쭉함

**증상**: 이미지 크기에 따라 폰트 크기가 달라져 가독성 저하

**원인**: 이미지 크기에 비례하여 폰트 크기를 계산함

**해결**: 
- `cv2.getFontScaleFromHeight()` 사용하여 픽셀 높이 기준으로 고정
- 이미지 해상도와 무관하게 고정값 사용

**코드**:
```python
font_thickness = 1
font_scale = cv2.getFontScaleFromHeight(FONT, 12, font_thickness)  # 12px 높이 고정
```

### 문제 5: 오버레이가 추론 전에 처리되는 것처럼 보임

**증상**: 폰트가 깨져 보임

**원인**: 오버레이 처리 순서가 불명확함

**해결**: 처리 순서를 명확히 분리
1. 추론 (원본 이미지)
2. 시각화 (추론 결과)
3. 오버레이 (시각화된 이미지의 복사본)
4. 표시 (imshow)

**코드**:
```python
def overlay_window_info(img, file_info, mode_name, inference_time):
    """원본 이미지를 수정하지 않고 복사본에 오버레이를 추가합니다."""
    img_copy = img.copy()  # 복사본 생성
    # ... 텍스트 추가 ...
    return img_copy
```

---

## 기술적 세부사항

### 1. COCO 카테고리 ID 시스템

COCO Panoptic Segmentation은 **원본 COCO 카테고리 ID**를 사용합니다.

- Thing: `thing_class_id` 리스트에 원본 ID 저장 (예: [1, 2, 3, ...])
- Stuff: `stuff_class_id` 리스트에 원본 ID 저장 (예: [92, 93, 94, ...])
- `category_id`는 이 원본 ID를 직접 사용

**매핑 과정**:
```python
# Thing 예시
category_id = 0  # COCO 원본 ID
idx = metadata.thing_class_id.index(0)  # thing_class_id에서 인덱스 찾기
class_name = metadata.thing_classes[idx]  # "person"
```

### 2. Visualizer 함수들의 동작 방식

#### draw_panoptic_seg_predictions()
```python
# 내부 동작 (의사코드)
for seg_info in segments_info:
    seg_id = seg_info['id']
    # panoptic_seg에서 마스크 추출
    mask = (panoptic_seg == seg_id)
    # segments_info에서 메타정보 사용
    category_id = seg_info['category_id']
    class_name = metadata.get_class_name(category_id)
    # 마스크 그리기 (bbox 없음)
    draw_mask(mask, class_name, color)
```

#### draw_instance_predictions()
```python
# 내부 동작 (의사코드)
for instance in instances:
    # instances 객체에서 직접 가져오기
    mask = instance.pred_masks
    bbox = instance.pred_boxes  # bbox 포함!
    class_name = metadata.get_class_name(instance.pred_classes)
    score = instance.scores
    # bbox와 마스크 모두 그리기
    draw_bbox(bbox)
    draw_mask(mask, class_name, score, color)
```

### 3. 마스크에서 Bbox 계산

`segments_info`에 bbox가 없으므로, 필요시 마스크에서 계산:

```python
mask = (panoptic_seg == seg_id)
y_indices, x_indices = np.where(mask)
if len(y_indices) > 0:
    x_min, x_max = x_indices.min(), x_indices.max()
    y_min, y_max = y_indices.min(), y_indices.max()
    bbox = [x_min, y_min, x_max, y_max]
```

### 4. ColorMode의 역할

Detectron2 Visualizer는 `ColorMode`로 시각화 스타일을 제어합니다:

- `ColorMode.IMAGE`: 원본 이미지 위에 랜덤 색상 마스크 오버레이
- `ColorMode.SEGMENTATION`: 배경을 grayscale로 변환하여 마스크가 더 돋보이게

**현재 구현**: 모든 모드에서 `ColorMode.IMAGE` 사용 (일관성 유지)

---

## 최종 구현 특징

### 1. 4가지 시각화 모드

| 모드 | 함수 | Thing | Stuff | Bbox | 특징 |
|------|------|-------|-------|------|------|
| 0: Panoptic | `draw_panoptic_seg_predictions()` | ✅ | ✅ | ❌ | 전체 panoptic 결과 |
| 1: Instance | `draw_instance_predictions()` | ✅ | ❌ | ✅ | Thing만, bbox 포함 |
| 2: Semantic | `draw_sem_seg()` | ✅ | ✅ | ❌ | 픽셀 단위 분류만 |
| 3: Contour | 직접 구현 | ✅ | ✅ | ❌ | 원본 + 윤곽선 |

### 2. 효율적인 모드 전환

- 추론 결과를 `current_outputs`에 저장
- S 키로 모드 전환 시 추론 재수행 없이 저장된 결과 재사용
- 빠른 모드 전환 가능

### 3. 클래스명 매핑 정확도

- `isthing` 필드로 Thing/Stuff 구분
- `thing_class_id` / `stuff_class_id`로 정확한 매핑
- Fallback 로직으로 안정성 확보

### 4. Contour 모드의 고급 기능

- 클래스별 다른 색상 (HSV 색상 공간 활용)
- 클래스명 + id:정확도 표시
- Thing/Stuff 배경 색상 구분
- 원본 이미지 보존

### 5. 사용자 경험

- 고정 폰트 크기 (12px)
- 안티앨리어싱 적용
- 윈도우 크기 고정 (높이 800px)
- 인터랙티브 조작 (A/D/S/Q 키)

---

## 핵심 인사이트

### 1. panoptic_seg vs segments_info

- **panoptic_seg**: 마스크 픽셀 데이터 (H×W 텐서)
- **segments_info**: 메타정보만 (리스트)
- Visualizer 함수들은 `panoptic_seg`에서 마스크를 추출하여 사용

### 2. Bbox 정보

- `segments_info`에는 **bbox가 없음**
- `outputs["instances"]`에만 bbox 포함 (`pred_boxes`)
- 필요시 마스크에서 계산 가능

### 3. Thing vs Stuff 차이

- Thing: `score`, `instance_id` 포함
- Stuff: `area` 포함, `score` 없음
- `isthing` 필드로 구분 (가장 정확)

### 4. 클래스명 매핑

- COCO는 원본 카테고리 ID 사용
- `thing_class_id` / `stuff_class_id`로 매핑 필요
- 단순 인덱스 매핑으로는 불가능

---

## 참고 자료

- [Detectron2 공식 문서](https://detectron2.readthedocs.io/)
- [COCO Panoptic Segmentation 공식 사이트](https://cocodataset.org/#panoptic-2021)
- [COCO Panoptic API GitHub](https://github.com/cocodataset/panopticapi)
- Detectron2 Tutorial: `Dtr2_Tutorial_panoptic.ipynb`

---

## 작업 일지

### 초기 구현
- Detectron2 Tutorial 기반으로 panoptic segmentation 추론 및 시각화 구현
- 기본적인 이미지 로딩 및 추론 파이프라인 구축

### 모드 확장
- 3가지 모드에서 4가지 모드로 확장
- Visualizer의 4가지 drawing 메서드 활용

### 클래스명 매핑 개선
- 복잡한 id2label 딕셔너리 제거
- 메타데이터에서 직접 클래스명 가져오기
- thing_class_id / stuff_class_id 활용

### Contour 모드 개선
- 각 세그먼트별 윤곽선 그리기
- 클래스별 색상 적용
- 클래스명 및 id:정확도 표시

### 폰트 및 오버레이 개선
- 고정 폰트 크기 적용
- 오버레이 처리 순서 명확화
- 안티앨리어싱 적용

---

**작성일**: 2024년  
**작성자**: AI Assistant  
**프로젝트**: Detectron2 COCO Panoptic Segmentation Viewer

