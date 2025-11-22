# OneFormer Panoptic Segmentation + Depth Estimation - Interactive 3D Visualization Tool

## 목차

1. [프로그램 개요](#1-프로그램-개요)
2. [핵심 기술 스택](#2-핵심-기술-스택)
3. [주요 기능 모듈](#3-주요-기능-모듈)
4. [기술 원리 상세](#4-기술-원리-상세)
5. [데이터 흐름](#5-데이터-흐름)
6. [핵심 알고리즘](#6-핵심-알고리즘)
7. [성능 최적화](#7-성능-최적화)
8. [사용 가이드](#8-사용-가이드)

---

## 1. 프로그램 개요

이 프로그램은 **OneFormer** 딥러닝 모델을 사용하여 이미지의 **Panoptic Segmentation**을 수행하고, **MiDaS** 모델을 사용하여 **Depth Estimation**을 수행한 후, **소실점(Vanishing Point)** 정보를 활용하여 depth 값을 보정하고, 클래스별로 적절한 **3D 구조**를 그려서 시각화하는 고급 인터랙티브 도구입니다.

### 주요 특징
- **실시간 인터랙티브 시각화**: 키보드/마우스 입력으로 다양한 모드 전환 및 세그먼트 선택
- **클래스별 3D 구조 추론**: 객체의 클래스 이름을 기반으로 적절한 3D 형태 자동 추정
- **소실점 기반 depth 보정**: 원근법을 고려한 정확한 depth 값 보정
- **Edge 결합 기술**: Depth edge와 이미지 edge를 결합하여 더 정확한 구조선 추출

---

## 2. 핵심 기술 스택

### 2.1 AI 모델

#### Segmentation: OneFormer
- **모델**: `shi-labs/oneformer_ade20k_swin_large`
- **기능**: Panoptic Segmentation (Thing + Stuff 통합 세그멘테이션)
- **데이터셋**: ADE20K (150개 클래스)
- **특징**: 
  - Thing(객체): 개별 인스턴스 세그멘테이션 (예: 사람, 자동차)
  - Stuff(배경): 의미론적 세그멘테이션 (예: 하늘, 벽, 바닥)
  - 공식 Thing/Stuff 분류 사용 (CSAILVision MIT)

#### Depth Estimation: MiDaS
- **모델**: `Intel/dpt-hybrid-midas`
- **방식**: Monocular Depth Estimation (단일 이미지로 depth 추정)
- **출력**: 상대적 depth 값 (절대 거리 아님)
- **특징**: 
  - 큰 값 = 가까운 객체
  - 작은 값 = 먼 객체

### 2.2 컴퓨터 비전 기술

#### 소실점 검출 (Vanishing Point Detection)
- **방법**: Hough Transform + 최소 제곱법
- **목적**: 원근법 기반 depth 보정 및 3D 구조 그리기

#### Depth Edge 검출
- **방법**: Sobel 필터 기반 gradient 분석
- **목적**: 객체 내부의 depth 변화가 큰 곳(구조선) 추출

#### 이미지 Edge 검출
- **방법**: Canny 엣지 검출
- **목적**: 실제 이미지의 경계선 추출

#### Depth 보정
- **방법**: 소실점 기반 원근법 보정
- **목적**: 원근감을 고려한 정확한 depth 값 계산

### 2.3 3D 시각화

- **클래스별 3D 구조 추론**: 클래스 이름 → 3D 형태 타입 매핑
- **Depth 기반 내부 구조 선 그리기**: Depth edge를 따라 객체 내부 구조 표현
- **이미지 edge와 depth edge 결합**: 더 정확한 구조선 추출
- **마스크 제약 선 그리기**: 세그먼트 내부에만 선 그리기 (제1원칙)

---

## 3. 주요 기능 모듈

### 3.1 소실점 검출 (VanishingPointDetector)

#### 기능
이미지 내의 직선들을 분석하여 주 소실점(Dominant Vanishing Point)을 추정합니다.

#### 알고리즘
1. **Canny 엣지 검출**: 그레이스케일 이미지에서 엣지 추출
2. **Hough Transform**: 엣지에서 직선 검출
3. **대각선 필터링**: 15도~75도 사이의 대각선만 선택 (소실점을 찾기 위한 주된 단서)
4. **최소 제곱법**: 직선들의 교점을 계산하여 소실점 추정

#### 수학적 원리
- 각 직선을 `y = mx + c` 형태로 표현
- 행렬 방정식 `Ax = b`로 변환:
  - `A = [[m1, -1], [m2, -1], ...]`
  - `b = [[-c1], [-c2], ...]`
- 최소 제곱법으로 해 `(vx, vy)` 계산: `(vx, vy) = argmin ||Ax - b||²`

#### 코드 위치
```python
class VanishingPointDetector:
    def find_vanishing_point(self, image):
        # Canny 엣지 검출
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        # Hough Transform 직선 검출
        lines = cv2.HoughLinesP(edges, ...)
        # 최소 제곱법으로 소실점 계산
        vx, vy = np.linalg.lstsq(A, b, rcond=None)[0]
```

---

### 3.2 Depth 보정 (correct_depth_with_vanishing_point)

#### 기능
소실점을 이용하여 depth 값을 보정합니다.

#### 원리
- **소실점의 위치**: 렌즈 쪽(화면 가운데 뒤쪽)에 위치
- **원근법 원리**: 소실점 방향으로 갈수록 더 멀어짐
- **보정 방향**:
  - 소실점에 가까운 픽셀 → depth 값 감소 (더 먼 것으로)
  - 소실점에서 먼 픽셀 → depth 값 증가 (더 가까운 것으로)

#### 수학적 공식
```
distance = sqrt((x - vx)² + (y - vy)²)  # 소실점과의 거리
normalized_distance = distance / max_distance  # 0~1 정규화
vp_correction = -normalized_distance * depth_range * weight  # 보정값
corrected_depth = original_depth + vp_correction  # 보정 적용
```

#### 파라미터
- `weight` (기본값 0.3): 보정 강도 조절 (0.0~1.0)

#### 코드 위치
```python
def correct_depth_with_vanishing_point(depth_map, vanishing_point, weight=0.3):
    # 소실점에서 각 픽셀까지의 거리 계산
    distances = np.sqrt((x_coords - vx)**2 + (y_coords - vy)**2)
    # 보정값 계산 및 적용
    vp_correction = -normalized_distances * depth_range * weight
    corrected_depth = depth_map + vp_correction
```

---

### 3.3 클래스별 3D 구조 그리기

#### 3.3.1 형태 타입 매핑 (get_shape_type_from_class)

클래스 이름을 기반으로 3D 형태를 추정합니다.

| 형태 타입 | 클래스 예시 | 설명 |
|---------|-----------|------|
| `outline_only` | window, curtain, mirror | 외곽선만 그리기 |
| `horizontal_plane` | floor, road, ceiling | 수평면 (접힌 종이 효과) |
| `wall` | wall | 수직면 (floor/ceiling과의 경계선 포함) |
| `topographic` | mountain, earth | 등고선 |
| `cube_perspective` | bed, table, building | 큐브 (depth edge + 이미지 edge 결합) |
| `person` | person | 사람 (실린더 + 골격 구조) |
| `cylinder_symmetric` | column, tree | 실린더 (기둥, 나무) |
| `sphere` | ball, globe | 구 |

#### 3.3.2 큐브 구조 그리기 (draw_cube_perspective)

**핵심 기술**: Depth edge와 이미지 edge를 결합하여 더 정확한 구조선 추출

**알고리즘**:
1. **외곽선 그리기**: 세그먼트 마스크의 외곽선 추출 및 그리기
2. **Depth edge 감지**: Sobel 필터로 depth gradient 계산 → 수평/수직 edge 추출
3. **이미지 edge 감지**: Canny 엣지 검출
4. **Edge 결합**: Depth edge와 이미지 edge가 모두 있는 곳만 선 그리기
5. **마스크 제약**: 모든 선은 세그먼트 마스크 내부에만 그려짐

**수평/수직 선 그리기**:
- 수평 선: 수직 edge를 따라 y 좌표별로 그리기
- 수직 선: 수평 edge를 따라 x 좌표별로 그리기
- 간격 처리: 연속된 점들을 선분으로 연결 (간격이 5픽셀 이하일 때)

#### 3.3.3 사람 구조 그리기 (draw_person_structure)

- **몸통**: 수직 실린더 (타원형 외곽선)
- **머리**: 상단에 원형
- **골격 구조**: 중심선과 주요 관절 위치 표시

#### 3.3.4 평면 구조 그리기 (draw_plane_structure)

- **접힌 종이 효과**: depth 변화가 큰 곳에 선을 그어 평면이 접힌 것처럼 표현
- **수평/수직 선**: depth edge를 따라 평면의 구조 표현

#### 3.3.5 벽 구조 그리기 (draw_wall_structure)

- **외곽선**: 벽의 외곽선 그리기
- **경계선**: floor/ceiling과의 교차점에 수직선 추가

#### 3.3.6 등고선 그리기 (draw_contour_lines)

- **원리**: depth 값을 기반으로 등고선 추출
- **방법**: depth 범위를 여러 구간으로 나누어 각 구간의 경계선 그리기

#### 3.3.7 마스크 제약 선 그리기 (draw_line_in_mask)

**핵심 원칙**: 모든 선은 세그먼트 마스크 내부에만 그려짐

**알고리즘**: Bresenham 선 그리기 알고리즘 변형
1. 시작점에서 끝점까지 선을 따라 이동
2. 각 점이 마스크 내부에 있는지 확인
3. 마스크 내부에 있는 연속된 점들만 선으로 연결
4. 마스크 외부로 나가면 선을 끊고, 다시 마스크 내부로 들어오면 새로운 선분 시작

---

### 3.4 Depth Edge 검출 (detect_depth_edges_in_segment)

#### 기능
세그먼트 내부의 depth 변화가 큰 곳(edge)을 감지합니다.

#### 알고리즘
1. **Sobel 필터 적용**: 
   - 수평 gradient: `Sobel(depth, dx=1, dy=0)`
   - 수직 gradient: `Sobel(depth, dx=0, dy=1)`
2. **임계값 계산**: `threshold = depth_range * threshold_ratio` (기본값 0.15)
3. **Edge 마스크 생성**: gradient가 임계값을 넘는 곳만 edge로 표시

#### 수학적 원리
```
Gx = Sobel_x(depth)  # 수평 gradient
Gy = Sobel_y(depth)  # 수직 gradient
horizontal_edges = |Gx| > threshold  # 수평 edge
vertical_edges = |Gy| > threshold    # 수직 edge
```

---

### 3.5 인터랙티브 기능

#### 3.5.1 키보드 입력

| 키 | 기능 |
|---|------|
| `A` / `←` | 이전 이미지로 이동 |
| `D` / `→` | 다음 이미지로 이동 |
| `S` | Depth Map 모드 토글 (밝기 = 가까움) |
| `E` | 3D 박스 표시 토글 |
| `Q` | 프로그램 종료 |

#### 3.5.2 마우스 입력

**마우스 이동 (EVENT_MOUSEMOVE)**:
- **기능**: 실시간 세그먼트 정보 표시
- **표시 정보**:
  - 클래스 이름 (초록색, 굵게)
  - ID: [세그먼트 ID]
  - Depth: [depth 값]
- **선택 로직**: 여러 세그먼트가 겹치면 depth가 가장 큰 것(가장 가까운 것) 선택
- **검색 반경**: 마우스 위치 주변 3픽셀 범위에서 세그먼트 검색

**왼쪽 클릭 (EVENT_LBUTTONDOWN)**:
- **기능**: 세그먼트 선택 (해당 세그먼트만 표시)
- **선택 로직**:
  1. 여러 세그먼트가 겹치면:
     - **다른 클래스**: 가장 작은 영역 우선
     - **같은 클래스**: 소실점에서 멀리 떨어진 것 우선 (더 앞에 있는 것으로 간주)
  2. 선택된 세그먼트만 시각화

**오른쪽 클릭 (EVENT_RBUTTONDOWN)**:
- **기능**: 모든 인스턴스 표시 (선택 해제)

---

### 3.6 시각화 기능

#### Thing vs Stuff 구분
- **Thing (객체)**: 
  - 외곽선으로 표시 (노란색 텍스트)
  - 개별 인스턴스 세그멘테이션
- **Stuff (배경)**: 
  - 반투명 색상 오버레이 (흰색 텍스트)
  - 의미론적 세그멘테이션

#### Depth Map 모드
- **배경**: Depth 맵을 배경으로 사용
- **정규화**: 큰 값(가까운 것)을 밝게 표시
- **공식**: `normalized = (depth - min) / (max - min) * 255`

#### 3D 구조 표시
- **조건**: `show_3d_boxes=True`일 때만 표시
- **방법**: 클래스별 적절한 3D 형태 그리기
- **색상**: 각 세그먼트마다 고유한 색상 (HSV 색상 공간, 137.5도 간격)

#### 마우스 정보 오버레이
- **위치**: 마우스 위치 근처 (화면 경계 자동 조정)
- **표시**: 반투명 검은색 배경 + 초록색 십자가 + 텍스트 정보

---

## 4. 기술 원리 상세

### 4.1 Panoptic Segmentation 원리

**Panoptic Segmentation**은 **Semantic Segmentation**과 **Instance Segmentation**을 통합한 기술입니다.

- **Semantic Segmentation**: 각 픽셀에 클래스 라벨 할당 (예: "사람", "자동차")
- **Instance Segmentation**: 각 객체 인스턴스를 개별적으로 구분 (예: "사람1", "사람2")
- **Panoptic Segmentation**: Thing(객체)은 인스턴스로, Stuff(배경)는 의미론적으로 구분

**OneFormer 모델**:
- **입력**: RGB 이미지
- **출력**: 
  - `seg_map`: 각 픽셀의 세그먼트 ID
  - `segments_info`: 각 세그먼트의 정보 (id, label_id, score 등)

---

### 4.2 Monocular Depth Estimation 원리

**Monocular Depth Estimation**은 단일 이미지로부터 depth 정보를 추정하는 기술입니다.

**MiDaS 모델**:
- **입력**: RGB 이미지
- **출력**: 상대적 depth 값 (절대 거리 아님)
- **특징**: 
  - 큰 값 = 가까운 객체
  - 작은 값 = 먼 객체
  - 픽셀 단위 depth 정보 제공

**제한사항**:
- 절대 거리는 알 수 없음 (상대적 거리만)
- 스케일 불변성 (같은 객체라도 이미지 크기에 따라 값이 다를 수 있음)

---

### 4.3 소실점 기반 원근법 원리

**소실점(Vanishing Point)**은 평행한 직선들이 원근 투영에서 만나는 점입니다.

**원근법 원리**:
- 소실점 방향으로 갈수록 객체가 작아지고 멀어 보임
- 소실점에 가까운 픽셀은 더 먼 것으로 간주
- 소실점에서 먼 픽셀은 더 가까운 것으로 간주

**Depth 보정 공식**:
```
distance_to_vp = sqrt((x - vx)² + (y - vy)²)
normalized_distance = distance_to_vp / max_distance
correction = -normalized_distance * depth_range * weight
corrected_depth = original_depth + correction
```

**보정 효과**:
- 소실점 방향으로 갈수록 depth 값 감소 → 더 먼 것으로 보정
- 소실점에서 멀수록 depth 값 증가 → 더 가까운 것으로 보정

---

### 4.4 Edge 결합 기술

**문제**: Depth edge만으로는 실제 객체의 경계를 정확히 잡기 어려움

**해결**: Depth edge와 이미지 edge를 결합

**알고리즘**:
1. **Depth edge 검출**: Sobel 필터로 depth gradient 계산
2. **이미지 edge 검출**: Canny 엣지 검출
3. **결합**: 두 edge가 모두 있는 곳만 최종 edge로 사용
   ```python
   final_edges = depth_edges & image_edges
   ```

**장점**:
- Depth 정보로 구조를 파악
- 이미지 정보로 정확한 경계 추출
- 더 정확하고 자연스러운 3D 구조선

---

### 4.5 마스크 제약 선 그리기 원리

**핵심 원칙**: 모든 선은 세그먼트 마스크 내부에만 그려짐

**Bresenham 알고리즘 변형**:
1. 시작점에서 끝점까지 선을 따라 이동
2. 각 점이 마스크 내부에 있는지 확인
3. 마스크 내부에 있는 연속된 점들만 선으로 연결
4. 마스크 외부로 나가면 선을 끊고, 다시 마스크 내부로 들어오면 새로운 선분 시작

**구현**:
```python
def draw_line_in_mask(img, pt1, pt2, mask, color, thickness=2):
    # Bresenham 알고리즘으로 선을 따라 이동
    # 마스크 내부에 있는 점들만 수집
    # 연속된 점들을 선분으로 그리기
```

**효과**:
- 세그먼트 경계를 넘지 않음
- 객체의 실제 형태를 정확히 표현
- 시각적으로 깔끔한 3D 구조

---

## 5. 데이터 흐름

### 전체 파이프라인

```
1. 이미지 로드
   ↓
2. Segmentation 추론 (OneFormer)
   - seg_map: 세그멘테이션 맵
   - segments_info: 세그먼트 정보
   ↓
3. Depth 추론 (MiDaS)
   - depth_map: Depth 맵
   ↓
4. 소실점 검출 (VanishingPointDetector)
   - vanishing_point: 소실점 좌표 (vx, vy)
   ↓
5. Depth 보정 (correct_depth_with_vanishing_point)
   - corrected_depth_map: 보정된 Depth 맵
   ↓
6. 시각화 (visualize_cv2_all)
   - Stuff 오버레이 그리기
   - Thing 외곽선 그리기
   - 3D 구조 그리기 (show_3d_boxes=True일 때)
   - 마우스 정보 오버레이
   ↓
7. 사용자 인터랙션 처리
   - 키보드 입력 (A/D/S/E/Q)
   - 마우스 입력 (이동/클릭)
   ↓
8. 결과 업데이트
   - 모드 전환 시 재시각화
   - 이미지 변경 시 재추론
```

### 상세 데이터 흐름

#### Segmentation 추론
```python
img_rgb → processor → inputs → segmentation_model → seg_outputs
→ processor.post_process_panoptic_segmentation → seg_map, segments_info
```

#### Depth 추론
```python
img_rgb → Image.fromarray → depth_estimator → depth_result
→ depth_result["depth"] → depth_map (numpy array)
```

#### 소실점 검출
```python
img_bgr → gray → Canny → edges → HoughLinesP → lines
→ 필터링 (대각선만) → 최소 제곱법 → vanishing_point
```

#### Depth 보정
```python
depth_map + vanishing_point → 거리 계산 → 보정값 계산
→ corrected_depth_map
```

#### 3D 구조 그리기
```python
seg_map + depth_map + class_name → get_shape_type_from_class
→ draw_cube_perspective / draw_person_structure / ...
→ detect_depth_edges_in_segment → draw_line_in_mask
→ 최종 3D 구조
```

---

## 6. 핵심 알고리즘

### 6.1 소실점 검출 알고리즘

**입력**: 이미지 (BGR)
**출력**: 소실점 좌표 (vx, vy)

**단계**:
1. 그레이스케일 변환
2. Canny 엣지 검출
3. Hough Transform으로 직선 검출
4. 대각선 필터링 (15도~75도)
5. 최소 제곱법으로 교점 계산
6. 범위 검증 (화면 밖이면 중심점 반환)

**시간 복잡도**: O(n²) (Hough Transform)

---

### 6.2 Depth 보정 알고리즘

**입력**: depth_map, vanishing_point, weight
**출력**: corrected_depth_map

**단계**:
1. 소실점에서 각 픽셀까지의 거리 계산: `distance = sqrt((x-vx)² + (y-vy)²)`
2. 거리 정규화: `normalized = distance / max_distance`
3. 보정값 계산: `correction = -normalized * depth_range * weight`
4. 보정 적용: `corrected = original + correction`
5. 범위 제한: `clipped = clip(corrected, min, max)`

**시간 복잡도**: O(n) (n = 픽셀 수)

---

### 6.3 Edge 결합 알고리즘

**입력**: depth_map, segment_mask, img_bgr
**출력**: horizontal_edges, vertical_edges

**단계**:
1. Sobel 필터로 depth gradient 계산
2. 임계값 계산: `threshold = depth_range * 0.15`
3. Depth edge 마스크 생성: `|gradient| > threshold`
4. Canny 엣지 검출 (이미지)
5. Edge 결합: `final_edges = depth_edges & image_edges`

**시간 복잡도**: O(n) (Sobel 필터)

---

### 6.4 마스크 제약 선 그리기 알고리즘

**입력**: pt1, pt2, mask
**출력**: 마스크 내부의 선분들

**단계** (Bresenham 알고리즘 변형):
1. 시작점에서 끝점까지 선을 따라 이동
2. 각 점이 마스크 내부에 있는지 확인
3. 마스크 내부에 있는 연속된 점들만 수집
4. 마스크 외부로 나가면 현재 선분 종료
5. 다시 마스크 내부로 들어오면 새로운 선분 시작
6. 수집된 선분들을 그리기

**시간 복잡도**: O(max(|dx|, |dy|))

---

## 7. 성능 최적화

### 7.1 추론 결과 캐싱

**전략**: 이미지 변경 시에만 재추론, 모드 전환 시에는 재시각화만

**구현**:
```python
# 현재 추론 결과 저장
current_seg_map = None
current_depth_map = None
current_vanishing_point = None

# 이미지 변경 시에만 재추론
if cur_idx != prev_idx:
    current_seg_map, current_depth_map, ... = run_inference(...)

# 모드 전환 시에는 재시각화만
if show_depth_mode changed or show_3d_boxes changed:
    visualize_cv2_all(...)  # 재추론 없이 시각화만
```

**효과**: 모드 전환 시 즉시 반응 (재추론 시간 없음)

---

### 7.2 마우스 이벤트 실시간 처리

**전략**: `cv2.waitKey(1)`로 마우스 이벤트를 실시간으로 처리

**구현**:
```python
cv2.setMouseCallback(window_name, mouse_callback)
while True:
    key = cv2.waitKey(1) & 0xFF
    # 마우스 이동 시 mouse_callback이 자동 호출
    # mouse_segment_info가 업데이트됨
    visualize_cv2_all(..., mouse_segment_info=mouse_segment_info)
```

**효과**: 마우스 이동 시 즉시 정보 표시

---

### 7.3 리사이즈 최적화

**전략**: 시각화를 위해 이미지를 리사이즈 (원본은 유지)

**구현**:
```python
TARGET_HEIGHT = 800  # 목표 높이
target_w = int(w * TARGET_HEIGHT / h)  # 비율 유지
depth_resized = cv2.resize(depth_map, (target_w, TARGET_HEIGHT))
```

**효과**: 
- 시각화 속도 향상 (작은 이미지로 처리)
- 원본 데이터 보존 (정확한 추론)

---

## 8. 사용 가이드

### 8.1 설치 및 실행

#### 필수 패키지
```bash
pip install torch torchvision
pip install transformers
pip install opencv-python
pip install numpy pillow
```

#### 실행
```bash
python OneFormerTyny_f_depth_VA.py
```

#### 이미지 디렉토리 설정
```python
IMAGE_DIR = r"D:/git/detectron2/ade20k_consistency/original_ade20k"
```

---

### 8.2 기본 사용법

1. **프로그램 시작**: 첫 번째 이미지가 자동으로 로드되고 추론됩니다.
2. **이미지 이동**: 
   - `A` 또는 `←`: 이전 이미지
   - `D` 또는 `→`: 다음 이미지
3. **Depth Map 모드**: `S` 키로 토글 (밝기 = 가까움)
4. **3D 구조 표시**: `E` 키로 토글
5. **세그먼트 선택**: 마우스 왼쪽 클릭
6. **모든 인스턴스 표시**: 마우스 오른쪽 클릭
7. **종료**: `Q` 키

---

### 8.3 고급 사용법

#### 마우스 호버 정보
- 마우스를 이동하면 해당 위치의 세그먼트 정보가 실시간으로 표시됩니다.
- 여러 세그먼트가 겹치면 가장 가까운 것(큰 depth 값)이 표시됩니다.

#### 세그먼트 선택 우선순위
- **다른 클래스**: 가장 작은 영역 우선
- **같은 클래스**: 소실점에서 멀리 떨어진 것 우선 (더 앞에 있는 것으로 간주)

#### Depth 보정 강도 조절
```python
# visualize_cv2_all 함수 내에서
depth_resized = correct_depth_with_vanishing_point(
    depth_resized, vanishing_point, weight=0.3  # weight 조절 (0.0~1.0)
)
```

---

### 8.4 문제 해결

#### Depth 값이 이상한 경우
- **원인**: 소실점 검출 실패 또는 잘못된 보정
- **해결**: `weight` 파라미터를 낮추거나 (0.1~0.2) 소실점 검출 로직 개선

#### 3D 구조가 세그먼트 밖으로 나가는 경우
- **원인**: `draw_line_in_mask` 함수의 버그
- **해결**: 마스크 검증 로직 확인

#### 성능이 느린 경우
- **원인**: GPU 미사용 또는 큰 이미지
- **해결**: 
  - GPU 사용 확인: `torch.cuda.is_available()`
  - `TARGET_HEIGHT` 값 낮추기 (예: 600)

---

## 부록

### A. 클래스별 3D 형태 매핑 상세

| 클래스 이름 패턴 | 형태 타입 | 설명 |
|----------------|---------|------|
| window, curtain, mirror, cushion, palm, sky, poster, picture | outline_only | 외곽선만 |
| floor, road, sidewalk, runway, ceiling, carpet, rug | horizontal_plane | 수평면 |
| wall | wall | 수직면 (경계선 포함) |
| mountain, earth | topographic | 등고선 |
| bed, table, desk, cabinet, counter, building, house | cube_perspective | 큐브 |
| person | person | 사람 (실린더+골격) |
| column, tree, pole, post | cylinder_symmetric | 실린더 |
| ball, globe | sphere | 구 |

### B. 주요 상수

```python
IMAGE_DIR = r"D:/git/detectron2/ade20k_consistency/original_ade20k"
SEGMENTATION_MODEL = "shi-labs/oneformer_ade20k_swin_large"
DEPTH_MODEL = "Intel/dpt-hybrid-midas"
TARGET_HEIGHT = 800
```

### C. 참고 자료

- **OneFormer**: [Paper](https://arxiv.org/abs/2211.06257), [Hugging Face](https://huggingface.co/shi-labs/oneformer_ade20k_swin_large)
- **MiDaS**: [Paper](https://arxiv.org/abs/1907.01341), [Hugging Face](https://huggingface.co/Intel/dpt-hybrid-midas)
- **ADE20K**: [Dataset](https://groups.csail.mit.edu/vision/datasets/ADE20K/)

---

**작성일**: 2024
**버전**: 1.0
**작성자**: AI Assistant

