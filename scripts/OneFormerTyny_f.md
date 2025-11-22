# OneFormer Panoptic Segmentation 시각화 도구 - 기술 개요

## 📋 **프로젝트 목적**

이 코드는 **OneFormer 모델**을 사용하여 **Panoptic Segmentation**을 수행하고, 결과를 대화형으로 시각화하는 도구입니다. 자율주행 시스템이나 시각장애인 안내 시스템 개발을 위한 **데이터 라벨링 및 품질 검증** 도구로 활용됩니다.

---

## 🏗️ **시스템 아키텍처**

```
┌─────────────────────────────────────────────────────────────┐
│                     데이터 입력                               │
│  ADE20K 이미지 (.jpg) → OpenCV로 로드 (BGR)                 │
└───────────────────┬─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│                  전처리 (Processor)                          │
│  BGR → RGB 변환 → 정규화 → Tensor 변환                      │
└───────────────────┬─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│              모델 추론 (OneFormer)                           │
│  Transformer 기반 Universal Segmentation                    │
│  출력: Logits (클래스별 예측값)                              │
└───────────────────┬─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│                후처리 (Processor)                            │
│  Logits → Segmentation Map (픽셀 마스크)                    │
│  Thing/Stuff 메타데이터 생성                                │
└───────────────────┬─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│                   시각화 (OpenCV)                            │
│  1. 이미지 리사이즈 (800px)                                  │
│  2. Stuff → 반투명 채우기                                    │
│  3. Thing → 폴리곤 외곽선                                    │
│  4. 라벨 및 정보 표시                                        │
└───────────────────┬─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│              대화형 탐색 (Keyboard Input)                    │
│  A/D 키 또는 화살표로 이미지 이동                            │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔧 **핵심 기술 스택**

### **1. 딥러닝 프레임워크**

#### **PyTorch (2.x)**
- **역할**: 모델 추론 엔진
- **사용 이유**: 
  - OneFormer가 PyTorch로 구현됨
  - 동적 그래프로 디버깅 용이
  - CUDA 지원으로 GPU 가속
- **주요 사용:**
  ```python
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  with torch.no_grad():  # 추론 모드 (그래디언트 계산 비활성화)
      outputs = model(**inputs)
  ```

#### **Hugging Face Transformers**
- **역할**: 사전 학습된 모델 및 전/후처리 라이브러리
- **사용 컴포넌트**:
  - `OneFormerProcessor`: 이미지 전처리 및 후처리
  - `OneFormerForUniversalSegmentation`: Panoptic Segmentation 모델
- **장점**:
  - 모델을 단 2줄로 로드 가능
  - 자동 다운로드 및 캐싱
  - 일관된 API

---

### **2. Computer Vision 라이브러리**

#### **OpenCV (cv2)**
- **역할**: 이미지 로드, 처리, 시각화
- **주요 기능 활용**:

| 기능 | 함수 | 목적 |
|------|------|------|
| 이미지 로드 | `cv2.imread()` | BGR 형식으로 이미지 읽기 |
| 색상 변환 | `cv2.cvtColor()` | BGR ↔ RGB, HSV ↔ BGR |
| 리사이즈 | `cv2.resize()` | 이미지 크기 조정 |
| 외곽선 검출 | `cv2.findContours()` | 마스크에서 폴리곤 추출 |
| 폴리곤 그리기 | `cv2.drawContours()` | Thing 외곽선 그리기 |
| 알파 블렌딩 | `cv2.addWeighted()` | 반투명 오버레이 |
| 텍스트 표시 | `cv2.putText()` | 라벨 및 정보 표시 |
| 윈도우 제어 | `cv2.namedWindow()` 등 | 대화형 디스플레이 |

**OpenCV 선택 이유**:
- ✅ PIL/matplotlib 대비 **빠른 성능**
- ✅ 풍부한 **이미지 처리 함수**
- ✅ **윈도우 제어** 및 키보드 입력 지원
- ✅ BGR 형식으로 비디오/카메라와 호환성 좋음

---

### **3. OneFormer 모델**

#### **모델 아키텍처**
```
입력 이미지
    ↓
Swin Transformer (Backbone)
    ↓
Feature Pyramid Network
    ↓
Query-based Decoder (Transformer)
    ├─ Task Token (semantic/instance/panoptic)
    ├─ Object Queries (DETR 스타일)
    └─ Pixel Decoder
    ↓
출력: Segmentation Map + Metadata
```

#### **모델 특징**
- **Universal**: 하나의 모델로 3가지 task 수행
  - Semantic Segmentation
  - Instance Segmentation
  - Panoptic Segmentation ← **현재 사용**
- **Query-based**: DETR 계열 아키텍처
- **Task-conditioned**: Task token으로 모드 전환

#### **사용 모델 상세**
- **이름**: `shi-labs/oneformer_ade20k_swin_tiny`
- **백본**: Swin Transformer (Tiny)
- **학습 데이터**: ADE20K (150개 클래스)
- **파라미터 수**: ~약 50M (Tiny 버전)
- **추론 속도**: ~0.1-0.5초/이미지 (GPU)

---

## 🎨 **시각화 알고리즘**

### **1. 이미지 리사이즈 전략**
```python
# 원리: 고해상도 이미지에서 텍스트와 선이 너무 작게 보이는 문제 해결
원본 (예: 2048×1536) → 800px 높이로 리사이즈 → 모든 그리기 수행
```

**장점**:
- 일관된 텍스트 크기 (12px)
- 적절한 선 두께 (2px)
- 빠른 렌더링

### **2. 색상 생성 (HSV Golden Angle)**
```python
hue = (idx * 137.5) % 180  # Golden angle approximation
```

**원리**:
- **Golden angle (137.5°)**: 피보나치 수열 기반 최적 분포
- **HSV 색상 공간**: 인접한 Hue 값이 시각적으로 구별됨
- **채도/명도 최대**: 선명한 색상

**예시**:
```
idx=0 → Hue=0   → 빨강
idx=1 → Hue=137 → 청록
idx=2 → Hue=95  → 연두
...
```

### **3. Thing vs Stuff 시각화**

| 요소 | Thing (객체) | Stuff (배경) |
|------|--------------|--------------|
| **그리기 방식** | 폴리곤 외곽선 | 반투명 채우기 |
| **두께/투명도** | 2px | 알파=120/255 |
| **색상** | 구별되는 색 | 구별되는 색 |
| **라벨 색상** | 노란색 | 흰색 |
| **목적** | 개별 객체 강조 | 배경 영역 표시 |

**알파 블렌딩 수식**:
```
blended = α × overlay + (1-α) × original
        = 0.47 × overlay + 0.53 × original
```

---

## 💾 **데이터 구조**

### **Segmentation Map**
```python
seg_map = np.ndarray (H, W)  # 각 픽셀의 인스턴스 ID
# 예시:
# [[0, 0, 1, 1, 1],
#  [0, 0, 1, 1, 2],
#  [3, 3, 3, 2, 2]]
```

### **Segments Info**
```python
segments_info = [
    {
        'id': 1,              # seg_map의 ID와 매칭
        'label_id': 15,       # 클래스 ID (0-149)
        'was_fused': False,   # 병합 여부
        'score': 0.95         # 신뢰도
        # 주의: 'isthing' 필드는 없음!
    },
    ...
]
```

### **Thing/Stuff 판단**
```python
# label_id를 기반으로 판단 (모델 설정에서)
is_thing_map = {
    0: False,   # wall (Stuff)
    1: False,   # building (Stuff)
    12: True,   # person (Thing)
    20: True,   # car (Thing)
    ...
}
```

---

## ⚡ **성능 최적화**

### **1. GPU 가속**
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)  # 모델을 GPU로 이동
```
- **속도 향상**: CPU 대비 10-50배
- **메모리**: VRAM 사용 (~2GB)

### **2. 추론 모드**
```python
with torch.no_grad():  # 그래디언트 계산 비활성화
    outputs = model(**inputs)
```
- **메모리 절약**: 50% 감소
- **속도 향상**: 20-30%

### **3. 마스크 리사이즈**
```python
interpolation=cv2.INTER_NEAREST  # 최근접 보간
```
- **빠름**: Bilinear/Bicubic 대비 3배 빠름
- **정확**: 마스크는 이산 값이므로 보간 불필요

---

## 🔄 **실행 흐름**

```
프로그램 시작
    │
    ├─ 이미지 리스트 로드 (glob)
    ├─ OneFormer 모델 로드 (Transformers)
    ├─ Thing/Stuff 매핑 로드 (모델 설정)
    │
    ▼
첫 번째 이미지 추론
    │
    ├─ cv2.imread() → BGR 이미지
    ├─ BGR → RGB 변환
    ├─ Processor 전처리
    ├─ 모델 forward pass (GPU)
    ├─ Processor 후처리
    │   ├─ seg_map 생성
    │   └─ segments_info 생성
    │
    ▼
시각화
    │
    ├─ 800px 리사이즈
    ├─ Stuff 그리기 (채우기)
    ├─ Thing 그리기 (외곽선)
    ├─ 라벨 표시
    └─ OpenCV 윈도우 표시
    │
    ▼
대화형 루프
    │
    ├─ 키 입력 대기 (cv2.waitKey)
    ├─ A/D/화살표 → 이미지 변경 → 재추론
    └─ Q → 종료
```

---

## 📊 **시스템 요구사항**

| 구성 요소 | 최소 | 권장 |
|-----------|------|------|
| **GPU** | - | NVIDIA GPU (4GB+ VRAM) |
| **RAM** | 8GB | 16GB |
| **Python** | 3.8+ | 3.10+ |
| **CUDA** | - | 11.8+ |
| **디스크** | 5GB | 10GB (모델 캐시) |

---

## 🎯 **활용 분야**

### **1. 자율주행**
- 차선 감지 (lane segmentation)
- 장애물 인식 (car, person, bicycle)
- 주행 가능 영역 (road, sidewalk)

### **2. 시각장애인 안내**
- 공간 인식 (wall, door, stairs)
- 장애물 경고 (pole, fire hydrant)
- 안전 경로 안내 (sidewalk vs road)

### **3. 데이터 라벨링**
- OneFormer로 초벌 라벨 생성
- 수동 검수 및 수정
- Detectron2 학습 데이터로 활용

---

## 🔍 **핵심 기술 상세**

### **Panoptic Segmentation이란?**

**Panoptic = Semantic + Instance**

| 유형 | 설명 | 예시 클래스 |
|------|------|------------|
| **Thing** | 개별적으로 세어지는 객체 | person, car, chair |
| **Stuff** | 배경 영역 (셀 수 없음) | sky, wall, road |

**출력 형식**:
- **Segmentation Map**: 각 픽셀이 어느 인스턴스에 속하는지
- **Segments Info**: 각 인스턴스의 메타데이터 (클래스, 신뢰도)

### **Transformer 기반 Segmentation**

**전통적 방식 (Mask R-CNN 등)**:
```
Image → CNN → RPN → RoI → Mask Head → Masks
```

**OneFormer (Transformer)**:
```
Image → Swin Transformer → Query Decoder → Masks
         ↓
    Task Token (panoptic/instance/semantic)
```

**장점**:
- ✅ End-to-end 학습
- ✅ Global context 이해
- ✅ 하나의 모델로 다중 task

---

## 📝 **코드 구조**

```python
OneFormerTyny_f.py (407줄)
├─ 문서 헤더 (1-31)           # 전체 개요 및 목적 설명
├─ Import (33-42)              # 필요 라이브러리
├─ visualize_cv2_all() (45-155)# 시각화 함수 (핵심)
│  ├─ 이미지 리사이즈
│  ├─ Stuff 그리기
│  ├─ Thing 그리기 (폴리곤)
│  ├─ 라벨 표시
│  └─ 정보 오버레이
├─ 전역 초기화 (160-202)      # 모델 로드, 설정
├─ run_inference() (207-249)   # 추론 파이프라인
└─ main() (255-280)            # 대화형 루프
```

---

## 🚀 **향후 확장 가능성**

### **1. Depth 통합 (공간 인식)**
```python
# Depth Estimation 추가
depth_map = depth_model(image)

# 거리 계산
for segment in segments_info:
    mask = (seg_map == segment['id'])
    distance = depth_map[mask].mean()
    if distance < 2.0:  # 2m 이내
        alert(f"{segment['class']} at {distance}m")
```

### **2. 라벨 저장 (Detectron2 학습용)**
```python
# COCO Panoptic 형식으로 저장
save_panoptic_annotation(seg_map, segments_info, filename)
```

### **3. 실시간 비디오 처리**
```python
# 비디오 스트리밍
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    result = run_inference_on_frame(frame)
    cv2.imshow('Live', result)
```

---

이 코드는 **최신 Transformer 기반 Segmentation 모델**과 **효율적인 OpenCV 시각화**를 결합하여, 실무에서 바로 활용 가능한 도구입니다! 🚀
