# OneFormerTyny_f_depth_protopy.py - Depth Map 생성 원리 및 소실점 분석

## 📋 목차
1. [프로그램 개요](#프로그램-개요)
2. [Depth Map 생성 원리](#depth-map-생성-원리)
3. [현재 코드의 소실점 구현](#현재-코드의-소실점-구현)
4. [Wall 깊이 패턴 분석](#wall-깊이-패턴-분석)
5. [Perspective 변환의 실제 사용](#perspective-변환의-실제-사용)
6. [개선 제안](#개선-제안)

---

## 프로그램 개요

### 사용 모델
```python
SEGMENTATION_MODEL = "shi-labs/oneformer_ade20k_swin_large"  # Panoptic Segmentation
DEPTH_MODEL = "Intel/dpt-hybrid-midas"                       # Depth Estimation
```

이 프로그램은 두 개의 독립적인 딥러닝 모델을 사용합니다:

1. **OneFormer (Panoptic Segmentation)**
   - 이미지를 semantic class로 분할
   - 각 픽셀에 class label 할당
   - Thing(객체) vs Stuff(배경) 구분

2. **MiDaS (Depth Estimation)**
   - 단일 RGB 이미지에서 depth map을 직접 예측
   - 각 픽셀의 상대적 거리 값 반환
   - 절대값이 아닌 **상대적 거리만 제공**

---

## Depth Map 생성 원리

### 핵심: MiDaS 신경망

**MiDaS (Monocular Depth Estimation)는 단일 이미지에서 depth를 추정하는 사전학습된 깊은 신경망입니다.**

```python
# 코드의 depth 사용 예시 (OneFormerTyny_f_depth_protopy.py)
def calculate_segment_depth(seg_map, segment_id, depth_map):
    """특정 세그먼트의 평균 depth 값을 계산합니다."""
    mask = (seg_map == segment_id)
    if not np.any(mask):
        return 0.0
    segment_depths = depth_map[mask]  # 세그먼트 영역의 depth 값들
    return float(np.mean(segment_depths))  # 평균 depth 반환
```

### MiDaS의 동작 원리

#### 1. **Input: RGB 이미지**
```
원본 이미지 (1920x1080)
    ↓
MiDaS 전처리 (정규화)
    ↓
신경망 처리
```

#### 2. **처리 과정**
- **Vision Transformer 또는 Hybrid CNN-Transformer** 사용
- 이미지의 공간 정보와 의미 정보를 학습한 가중치로 분석
- 실제 3D 장면 이해 능력 보유 (KITTI, NYU-Depth 등의 대규모 데이터셋으로 학습)

#### 3. **Output: Depth Map**
```
깊이 지도 (1920x1080)
- 각 픽셀: 상대적 거리 값 (0.0 ~ 1.0 또는 0.0 ~ 255)
- 높은 값: 가까운 거리
- 낮은 값: 먼 거리
```

### ⚠️ 중요한 특징

```python
# 코드 주석
"- Depth 정보는 상대적 거리 (절대값 아님)"
```

**MiDaS의 depth는 절대값이 아니라 상대값입니다:**

| 정보 | MiDaS (현재 코드) | 절대값 거리 측정 |
|------|------------------|-----------------|
| 단위 | 정규화된 값 (상대값) | 미터(m) |
| 측정 방식 | 신경망 추론 | 초음파/스테레오 카메라 |
| 정확성 | 중간 수준 | 높음 |
| 비용 | 무료 | 추가 센서 필요 |

---

## 현재 코드의 소실점 구현

### VanishingPointDetector 클래스

코드에는 소실점 검출기가 정의되어 있습니다:

```python
class VanishingPointDetector:
    """
    이미지 내의 직선들을 분석하여 주 소실점(Dominant Vanishing Point)을 추정하는 클래스
    """
    def find_vanishing_point(self, image):
        """
        이미지에서 소실점 (vx, vy)를 찾아 반환합니다.
        """
```

### 소실점 검출 알고리즘

```python
def find_vanishing_point(self, image):
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 1단계: Canny 엣지 검출
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # 2단계: Hough Transform으로 직선 검출
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                            minLineLength=50, maxLineGap=10)
    
    # 3단계: 직선 필터링 (대각선 방향만)
    filtered_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x1 == x2:  # 수직선 제외
            continue
        
        slope = (y2 - y1) / (x2 - x1)
        angle = math.degrees(math.atan(slope))
        
        # 15도~75도 사이의 대각선만 수집
        if 15 < abs(angle) < 75:
            filtered_lines.append((x1, y1, x2, y2, slope))
    
    # 4단계: 최소 제곱법(Least Squares)으로 소실점 계산
    # 모든 직선의 교점을 가장 잘 설명하는 점 = 소실점
    A_matrix = []
    b_vector = []
    
    for x1, y1, x2, y2, m in filtered_lines:
        # 직선 방정식: mx - y = -(y1 - m*x1)
        c = y1 - m * x1
        A_matrix.append([m, -1])
        b_vector.append([-c])
    
    # Ax = b 형태의 선형 방정식 해결
    A = np.array(A_matrix)
    b = np.array(b_vector)
    vx, vy = np.linalg.lstsq(A, b, rcond=None)[0]
    
    return (int(vx), int(vy))
```

### 소실점 검출의 수학적 의미

#### 관점 1: 고전 기하학적 관점
```
원근법(Perspective) 투시 도형:
모든 평행선이 한 점(소실점)으로 수렴

       │
       │
       │
    \  │  /
     \ │ /
      \│/
       V  ← 소실점
```

#### 관점 2: 선형대수 관점
```
직선 1: y = m₁x + b₁
직선 2: y = m₂x + b₂
직선 3: y = m₃x + b₃
...

모든 직선들의 "평균적인 교점" = 소실점

최소 제곱법: 모든 직선을 동시에 만족하는 점을 찾음
(정확한 교점이 없을 때 최적의 근사점 계산)
```

### ⚠️ 현재 코드의 문제점

```python
# find_vanishing_point 정의만 있고 실제로 사용되지 않음!

detector = VanishingPointDetector()  # ← 클래스 정의됨
vx, vy = detector.find_vanishing_point(image)  # ← 메서드 존재

# 하지만 depth map 생성 어디서도 사용 안 됨!
```

**현재 상황:**
- ✅ 소실점 검출 알고리즘 **구현됨**
- ❌ Depth 계산에 **사용되지 않음**
- ❌ Visualization에 **사용되지 않음**

---

## Wall 깊이 패턴 분석

### 관찰된 현상

당신이 발견한 현상:
```
평면인 벽에서:
- 중앙: 깊음 (depth 값이 큼) → 멀다
- 주변: 얕음 (depth 값이 작음) → 가깝다

시각적으로:
  ┌─────────────────────┐
  │  가까움│ │  가까움   │
  │   (얕음)│깊음│(얕음)   │
  │  가까움│ │  가까움   │
  └─────────────────────┘
```

### 진짜 원인: MiDaS의 학습 특성

**이것은 소실점 때문이 아니라, MiDaS 신경망의 학습 특성입니다:**

#### 원인 1: 렌즈 왜곡
```
카메라 렌즈의 특성:
- 화면 중앙: 렌즈의 주 광축
  → 카메라에서 더 멀리 떨어진 부분을 촬영
  
- 화면 가장자리: 왜곡된 각도
  → 실제로는 더 가까운 부분을 촬영 (광각 효과)

  카메라 뷰:
  
         카메라 렌즈
            │
            │
    ╱───────┼───────╲
   │        │        │
   │      중앙       │   ← 중앙이 실제로 더 먼 거리
   │    (더 멈)      │
   │        │        │
   ╲───────┼───────╱
   가장자리(더 가까움)
```

#### 원인 2: 학습 데이터의 통계적 특성
```
MiDaS 학습 데이터 (KITTI, NYU-Depth 등):

실내 장면:
- 벽의 가장자리: 바닥/천장과의 경계
- 벽의 중앙: 평면 영역

MiDaS는 이들 특징으로부터:
"벽의 중앙이 경계보다 더 멀다"는 패턴을 학습

결과:
  Wall depth distribution
  
  낮음  ┌──────────────┐  높음
  (가까움)│              │(멈)
       │  ╱──────────╲ │
       │ ╱            ╲│
       │╱              ╲
  높음  └──────────────┘  낮음
  (멈)     가장자리(가까움)
```

#### 원인 3: CNN의 수용장(Receptive Field)
```
신경망의 각 픽셀 예측은 주변 context 정보 사용

벽의 경계 픽셀:
- 주변에 floor/ceiling 정보
- "이곳은 경계이므로 다른 깊이"

벽의 중앙 픽셀:
- 주변이 모두 비슷함
- "이곳은 평면 영역이므로 더 먼 부분"
```

### 코드에서의 depth 계산

```python
def calculate_segment_depth(seg_map, segment_id, depth_map):
    """세그먼트의 평균 depth"""
    mask = (seg_map == segment_id)
    segment_depths = depth_map[mask]  # Wall의 모든 픽셀 depth
    return float(np.mean(segment_depths))  # 평균값
```

**Wall의 depth 분포 (예시):**

```
픽셀 위치    Depth 값
가장자리 1:  0.45  (가까움)
가장자리 2:  0.48  (가까움)
중앙 위쪽:   0.62  (멈)
중앙 중간:   0.65  (더 멈)
중앙 아래:   0.63  (멈)
가장자리 3:  0.46  (가까움)
가장자리 4:  0.47  (가까움)

평균 depth = (0.45 + 0.48 + 0.62 + 0.65 + 0.63 + 0.46 + 0.47) / 7
           ≈ 0.53 (중간값)
```

### Visualization에서의 반영

```python
def draw_wall_structure(img, bbox_2d, depth_map, segment_mask, ...):
    """벽 구조를 그립니다 (수직 평면, floor/ceiling과 만나는 곳에서 수직선으로 분리)"""
    
    # 외곽선 그리기 (가장자리 = 가까운 거리)
    mask_uint8 = (segment_mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, [largest_contour], -1, color, thickness)
    
    # Depth edge 감지 (중앙의 깊이 변화)
    horizontal_edges, vertical_edges = detect_depth_edges_in_segment(
        segment_mask, depth_map, threshold_ratio=0.15
    )
    # 이 엣지들이 바로 중앙의 깊이 차이를 나타냄
```

---

## Perspective 변환의 실제 사용

### ✅ 실제로 소실점이 사용되는 부분: Cube/Table 시각화

**Table, Bed, Chair 같은 3D 객체를 그릴 때만 수동으로 원근법 구현:**

```python
def draw_table_bed_cube(img, bbox_2d, depth_map, segment_mask, color, thickness=2):
    """
    테이블/침대 구조를 그립니다 (직육면체, 퍼스펙티브 9개 선).
    
    이것이 실제로 소실점 원리를 구현한 유일한 부분입니다!
    """
    x_min, y_min, x_max, y_max = bbox_2d
    
    # ===== 앞면 4개 정점 (가까운 면) =====
    front_bottom_left = (x_min, y_max)
    front_bottom_right = (x_max, y_max)
    front_top_right = (x_max, y_min)
    front_top_left = (x_min, y_min)
    
    # ===== 뒷면 4개 정점 (원근 효과 = 소실점 효과) =====
    perspective_scale = 0.75  # 원근 축소 비율 (핵심!)
    center_x = (x_min + x_max) / 2  # 이미지 중심 = 소실점으로 가정
    center_y = (y_min + y_max) / 2
    
    # 뒷면의 모든 좌표가 중심(소실점)으로 0.75배 축소됨
    back_bottom_left = (
        int(center_x + (x_min - center_x) * perspective_scale),
        int(center_y + (y_max - center_y) * perspective_scale)
    )
    back_bottom_right = (
        int(center_x + (x_max - center_x) * perspective_scale),
        int(center_y + (y_max - center_y) * perspective_scale)
    )
    back_top_right = (
        int(center_x + (x_max - center_x) * perspective_scale),
        int(center_y + (y_min - center_y) * perspective_scale)
    )
    back_top_left = (
        int(center_x + (x_min - center_x) * perspective_scale),
        int(center_y + (y_min - center_y) * perspective_scale)
    )
```

### 수학적 해석: 선형 보간(Linear Interpolation)

```python
# 뒷면 좌표 계산의 원리

# 일반 공식:
# 뒷면_x = center_x + (원본_x - center_x) * scale
#        = center_x * (1 - scale) + 원본_x * scale

# scale = 0.75일 때:
# 뒷면_x = center_x * 0.25 + 원본_x * 0.75

# 의미:
# - scale = 1.0 → 뒷면과 앞면이 같음 (3D 깊이 없음)
# - scale = 0.75 → 뒷면이 75% 앞면 위치 + 25% 중심
# - scale = 0.5 → 뒷면이 중간점 (강한 원근)
# - scale = 0.0 → 뒷면이 모두 중심점 (최강 원근)

# 시각적 표현:
# 
# 앞면 정점      뒷면 정점 (scale=0.75)
#     A ────────→ A'
#     │          │
#     │          │ ← 모두 중심으로 축소됨
#     │          │
#     B ────────→ B'
#     
#      center (소실점)
```

### 9개 선의 구성

```python
front_edges = [  # 앞면 사각형
    (front_bottom_left, front_bottom_right),    # 아래
    (front_bottom_right, front_top_right),      # 오른쪽
    (front_top_right, front_top_left),          # 위
    (front_top_left, front_bottom_left),        # 왼쪽
]

back_edges = [  # 뒷면 사각형
    (back_bottom_left, back_bottom_right),
    (back_bottom_right, back_top_right),
    (back_top_right, back_top_left),
    (back_top_left, back_bottom_left),
]

connecting_edges = [  # 깊이 연결선
    (front_bottom_left, back_bottom_left),      # 왼쪽 아래 깊이선
    (front_bottom_right, back_bottom_right),    # 오른쪽 아래 깊이선
    (front_top_right, back_top_right),          # 오른쪽 위 깊이선
    (front_top_left, back_top_left),            # 왼쪽 위 깊이선
]

# 총 4 + 4 + 4 = 12개 선?
# 아니면 일부만 그림 (코드에 따라)
```

### 선 그리기 (마스크 내부에만)

```python
def draw_line_in_mask(img, pt1, pt2, mask, color, thickness=2):
    """
    마스크 내부에만 선을 그립니다.
    
    이는 depth와 무관한 순수 시각화입니다:
    - depth map을 직접 사용하지 않음
    - 단순히 기하학적 구조만 그림
    """
```

---

## 개선 제안

### 현재 상태의 한계

| 항목 | 현재 코드 | 한계 |
|------|---------|------|
| Depth 생성 | MiDaS 모델 직접 사용 | 상대값만 제공 |
| 소실점 | VanishingPointDetector 정의됨 | 실제 사용 안 함 |
| Wall 깊이 | MiDaS 자체 학습 | 정확성 낮음 |
| Cube 그리기 | 수동 perspective (중심 기준) | 실제 소실점 미적용 |

### 제안 1: 소실점을 Depth 수정에 사용

**아이디어:**
```python
def refine_depth_with_vanishing_point(depth_map, vx, vy):
    """
    소실점을 기반으로 depth map을 조정합니다.
    
    원리:
    - 소실점에 가까울수록 더 깊다 (먼 거리)
    - 소실점에서 멀수록 더 얕다 (가까운 거리)
    
    수식:
    distance_to_vp = sqrt((x - vx)^2 + (y - vy)^2)
    adjusted_depth = original_depth + distance_to_vp * weight
    """
```

### 제안 2: 소실점을 Cube 그리기에 사용

**현재:**
```python
perspective_scale = 0.75  # 고정값
center_x = (x_min + x_max) / 2  # 이미지 중심 기준
```

**개선:**
```python
detector = VanishingPointDetector()
vx, vy = detector.find_vanishing_point(image)  # 실제 소실점

# 뒷면 좌표를 실제 소실점 기준으로 계산
back_bottom_left = (
    int(vx + (x_min - vx) * 0.75),  # ← center_x 대신 vx 사용
    int(vy + (y_max - vy) * 0.75)
)
```

**효과:**
```
현재:                    개선:
  B'                       B'
   │ (중심 기준)             │ (실제 소실점 기준)
   └─→ center              └─→ actual VP
                           (더 현실적)
```

### 제안 3: Wall을 Depth 인식하여 그리기

**아이디어:**
```python
def draw_wall_with_depth_contours(img, segment_mask, depth_map, color):
    """
    Wall의 depth 차이를 등고선으로 표현합니다.
    """
    valid_depths = depth_map[segment_mask]
    depth_min = valid_depths.min()
    depth_max = valid_depths.max()
    
    # 10개 깊이 레벨 생성
    levels = np.linspace(depth_min, depth_max, 10)
    
    for level in levels:
        level_mask = (np.abs(depth_map - level) < 0.05) & segment_mask
        # 각 깊이 레벨의 경계를 선으로 그림
        contours = cv2.findContours(level_mask.astype(np.uint8) * 255)
        cv2.drawContours(img, contours, -1, color, 1)
```

---

## 결론

### 질문에 대한 최종 답변

**Q: Wall을 보면 평면인데 가운데는 멀고 주변으로 갈수록 가까워지는데, 소실점을 사용하는 것 같아?**

**A: 아니, 소실점이 아니라 MiDaS 모델의 학습 특성입니다.**

1. **소실점은 정의되어 있지만 사용되지 않음**
   - `VanishingPointDetector` 클래스만 있음
   - Depth 계산에 미적용

2. **Wall의 깊이 패턴은 세 가지 원인:**
   - 렌즈 왜곡 (중앙이 실제로 더 멈)
   - CNN의 수용장 (경계와 중앙을 다르게 해석)
   - 학습 데이터의 통계적 특성

3. **실제로 소실점이 사용되는 부분:**
   - Table/Bed/Chair 같은 3D 객체
   - 수동으로 중심(또는 소실점) 기준 perspective 변환

4. **개선 방향:**
   - 소실점 검출 활성화
   - Depth 보정에 소실점 적용
   - 더 정확한 원근법 구현

---

## 🔧 구현된 개선사항

### 1. 소실점 활성화 (Vanishing Point Activation)

**문제점:**

```python
# 현재 코드
class VanishingPointDetector:
    """정의만 되어있고 사용되지 않음"""
    def find_vanishing_point(self, image):
        # 구현되어있음

# 하지만 실제 사용은:
# detector = VanishingPointDetector()
# vx, vy = detector.find_vanishing_point(image)  # 호출하지 않음
```

**개선 방향:**

```python
# 1단계: draw_table_bed_cube()에 소실점 파라미터 추가
def draw_table_bed_cube(img, bbox_2d, depth_map, segment_mask, color, thickness=2, 
                       vanishing_point=None):  # ← 소실점 추가
    """
    테이블/침대 구조를 그립니다 (직육면체, 실제 소실점 기반 퍼스펙티브).
    """
    x_min, y_min, x_max, y_max = bbox_2d
    
    # 앞면 4개 정점 (가까운 면)
    front_bottom_left = (x_min, y_max)
    front_bottom_right = (x_max, y_max)
    front_top_right = (x_max, y_min)
    front_top_left = (x_min, y_min)
    
    # [개선] 실제 소실점 또는 이미지 중심 사용
    if vanishing_point is not None:  # 소실점이 제공되면
        vp_x, vp_y = vanishing_point  # 실제 소실점 사용
    else:  # 소실점이 없으면
        vp_x = (x_min + x_max) / 2  # 이미지 중심 (기본값)
        vp_y = (y_min + y_max) / 2
    
    perspective_scale = 0.75  # 원근 축소 비율
    
    # 뒷면 4개 정점 (소실점 기준 계산)
    back_bottom_left = (int(vp_x + (x_min - vp_x) * perspective_scale),  # ← vp_x 사용
                        int(vp_y + (y_max - vp_y) * perspective_scale))   # ← vp_y 사용
    back_bottom_right = (int(vp_x + (x_max - vp_x) * perspective_scale),
                         int(vp_y + (y_max - vp_y) * perspective_scale))
    back_top_right = (int(vp_x + (x_max - vp_x) * perspective_scale),
                      int(vp_y + (y_min - vp_y) * perspective_scale))
    back_top_left = (int(vp_x + (x_min - vp_x) * perspective_scale),
                     int(vp_y + (y_min - vp_y) * perspective_scale))
    # ... (이후 엣지 그리기는 동일)
```

#### 2단계: main() 루프에서 소실점 검출 추가

```python
def main():
    # ... 초기화 코드 ...
    
    # [개선] 소실점 검출기 초기화
    vp_detector = VanishingPointDetector()  # ← 이제 사용
    
    # ... 이미지 로드 루프 ...
    while True:
        # ... 기존 추론 코드 ...
        
        # [개선] 소실점 검출
        vanishing_point = vp_detector.find_vanishing_point(img_bgr)  # ← 활성화
        print(f"🎯 소실점 (VP): {vanishing_point}")  # 디버그 출력
        
        # [개선] 시각화 시 소실점 정보 전달
        visualize_cv2_all(..., vanishing_point=vanishing_point)
```

---

### 2. Depth 기반 Wall 개선 (Depth-Aware Wall Rendering)

**현재 상황:**

- Wall의 깊이 차이가 단순히 MiDaS의 학습 특성에서 비롯됨
- Visualization이 평면으로 표시됨

**개선 방안:**

```python
def draw_wall_structure_improved(img, bbox_2d, depth_map, segment_mask, 
                                all_segments_info, seg_map_resized, color, 
                                thickness=2, show_depth_contours=True):  # ← 옵션 추가
    """
    벽 구조를 그립니다 (개선: depth 등고선 표시).
    """
    # 기존 외곽선 그리기
    mask_uint8 = (segment_mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(img, [largest_contour], -1, color, thickness)
    
    # [개선] Depth 등고선으로 wall의 깊이 차이 시각화
    if show_depth_contours:
        valid_depths = depth_map[segment_mask]
        if len(valid_depths) > 0:
            depth_min = valid_depths.min()
            depth_max = valid_depths.max()
            depth_range = depth_max - depth_min
            
            if depth_range > 0:
                # 5개의 깊이 레벨 생성
                num_levels = 5
                levels = np.linspace(depth_min, depth_max, num_levels)
                
                for level in levels[1:-1]:  # 첫/마지막 제외
                    # 해당 깊이 근처 픽셀들 찾기
                    level_mask = (np.abs(depth_map - level) < depth_range * 0.1) & segment_mask
                    
                    if np.any(level_mask):
                        # 등고선으로 표시
                        y_coords, x_coords = np.where(level_mask)
                        for i in range(0, len(y_coords), max(1, len(y_coords) // 30)):
                            y, x = y_coords[i], x_coords[i]
                            # 밝기로 깊이 표현 (밝을수록 가까움)
                            brightness = int(255 * (level - depth_min) / (depth_max - depth_min))
                            circle_color = (brightness, brightness, brightness)
                            cv2.circle(img, (x, y), 1, circle_color, 1)
    
    # floor/ceiling 연결선은 기존과 동일...
```

---

### 3. Cube Perspective에 Depth 반영 (Depth-Aware Perspective)

**현재:**

```python
# 고정된 perspective_scale = 0.75
```

**개선:**

```python
def draw_table_bed_cube_with_depth(img, bbox_2d, depth_map, segment_mask, color, 
                                   thickness=2, vanishing_point=None):
    """
    테이블/침대를 그립니다 (깊이에 따라 perspective 조정).
    """
    x_min, y_min, x_max, y_max = bbox_2d
    
    # [개선] Depth 범위에서 perspective_scale 동적 계산
    valid_depths = depth_map[segment_mask]
    if len(valid_depths) > 0:
        min_depth = valid_depths.min()
        max_depth = valid_depths.max()
        depth_diff = max_depth - min_depth
        
        # 깊이가 크면 perspective가 강함 (뒷면이 더 작음)
        if depth_diff > 0:
            # 깊이 차이 비율을 perspective_scale에 반영 (0.5~0.9 범위)
            perspective_scale = 0.9 - (depth_diff / (max_depth + 1)) * 0.4  # 범위: 0.5~0.9
        else:
            perspective_scale = 0.75  # 깊이 차이 없으면 기본값
    else:
        perspective_scale = 0.75
    
    # 앞/뒷면 계산은 위의 개선된 버전 사용...
    # (소실점 기반 계산과 동일한 로직)
```

---

### 4. 시각화 개선 (Visualization Enhancement)

```python
def visualize_cv2_all_improved(..., show_vanishing_point=False, 
                               show_depth_contours=False):
    """
    시각화 개선 버전.
    """
    # ... 기존 시각화 코드 ...
    
    # [개선] 소실점 표시 (옵션)
    if show_vanishing_point and vanishing_point is not None:
        vp_x, vp_y = vanishing_point
        cv2.circle(blended, (vp_x, vp_y), 15, (0, 0, 255), 2)  # 빨간 원
        cv2.putText(blended, "VP", (vp_x + 20, vp_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # [개선] 3D 박스 그릴 때 소실점 정보 전달
    if show_3d_boxes:
        for cid in unique_ids:
            # ...
            draw_shape_based_3d(..., vanishing_point=vanishing_point)
```

---

## 📊 개선 효과 비교

| 항목 | 개선 전 | 개선 후 |
|------|--------|--------|
| **소실점 사용** | ❌ 미사용 (코드만 있음) | ✅ 실제 활성화 |
| **Cube 원근감** | 고정된 0.75 | 동적 계산 (깊이 기반) |
| **Wall 시각화** | 평면 외곽선만 | + 깊이 등고선 추가 |
| **Depth 정보** | 평균값만 표시 | + 공간 분포 시각화 |
| **사용자 제어** | 불가능 | 토글 옵션 가능 |

---

## 💡 사용 예시

```python
# main() 함수에서:
vp_detector = VanishingPointDetector()
show_vp_toggle = False  # 사용자가 키보드로 토글
show_depth_contours_toggle = False

while True:
    # ... 이미지 로드 ...
    
    vanishing_point = vp_detector.find_vanishing_point(img_bgr)
    
    # 시각화
    visualize_cv2_all_improved(
        ...,
        vanishing_point=vanishing_point,
        show_vanishing_point=show_vp_toggle,
        show_depth_contours=show_depth_contours_toggle
    )
    
    # 키보드 입력
    key = cv2.waitKey(1) & 0xFF
    if key == ord('v'):  # V: 소실점 토글
        show_vp_toggle = not show_vp_toggle
    elif key == ord('c'):  # C: 깊이 등고선 토글
        show_depth_contours_toggle = not show_depth_contours_toggle
    elif key == ord('q'):  # Q: 종료
        break
```

---

## 🎯 결론

현재 코드의 문제점:

1. **소실점 구현만 있고 사용 안 함** → 활성화 필요
2. **Wall의 깊이 차이가 설명되지 않음** → 등고선 추가로 시각화
3. **Cube의 perspective가 고정됨** → Depth 기반 동적 계산
4. **사용자 제어 불가** → 토글 옵션 추가 가능

이 개선사항들을 적용하면:

- ✅ 소실점이 실제로 기능함
- ✅ Wall의 깊이 차이가 시각적으로 명확함
- ✅ 3D 표현이 더욱 현실적임
- ✅ 사용자가 시각화 옵션을 제어할 수 있음

---

## ✨ 구현 상태 (2025년 11월 25일 업데이트)

| 개선사항 | 상태 | 파일 위치 |
|---------|------|---------|
| VanishingPointDetector 활성화 | ✅ 완료 | main() 1416-1418줄 |
| draw_table_bed_cube() 개선 | ✅ 완료 | 762-815줄 |
| draw_shape_based_3d() 개선 | ✅ 완료 | 981-985줄 |
| vanishing_point 파라미터 전달 | ✅ 완료 | 1235-1237줄 |
| visualize_cv2_all() 개선 | ✅ 완료 | 1065줄 |
| Main loop VP 검출 | ✅ 완료 | 1456-1575줄 |

**이미 적용된 핵심 개선사항:**

1. ✅ VanishingPointDetector 인스턴스 생성 및 초기화
2. ✅ draw_table_bed_cube()에서 **상단/하단 별도 perspective_scale** 적용 (상단을 더 축소)
3. ✅ **Depth 기반 동적 perspective_scale** (0.4~0.85 범위)
4. ✅ visualize_cv2_all 및 draw_shape_based_3d에서 소실점 수신/전달
5. ✅ draw_wall_structure()에 **소실점 기반 기울기** 추가 (수직선→대각선)
6. ✅ 모든 시각화 함수에서 **소실점 활용**으로 공간감 강화

**2차 개선사항 (방금 추가):**

- ✅ top_perspective_scale = perspective_scale * 0.85 (상단 추가 축소)
- ✅ 바운딩 박스 중심에서 소실점까지 거리 계산
- ✅ Wall의 수직선을 **기울어진 대각선**으로 변경 (소실점 방향)

**권장 추가 개선사항 (미구현):**

- ⏳ draw_wall_structure()에 depth 등고선 추가
- ⏳ VP 시각화 마크 표시 (화면에 빨간 원)
- ⏳ V 키: 소실점 표시/숨김 토글
- ⏳ C 키: depth 등고선 토글
