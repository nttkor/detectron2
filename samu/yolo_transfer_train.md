# YOLO/Detectron2 전이학습 (Transfer Learning) 정리

## 📌 개요

기존 모델에 레이어를 추가하거나 수정해서 새로운 객체를 학습시키는 방법

---

## 🔄 Pose 모델 커스터마이징 옵션

### 방법 1: Pose 모델에 커스텀 객체 학습

#### 개념
```
현재: yolo11n-pose → 사람 17개 keypoints
목표: yolo11n-pose-custom → 자동차 4개 keypoints (바퀴 위치)
                         → 개 15개 keypoints (관절)
```

#### 필요한 것
1. **커스텀 데이터셋**: 객체 이미지 + keypoint 라벨링
2. **YAML 설정 파일**: keypoint 개수/이름 정의
3. **학습**: `yolo train` 명령어

#### 예시: 자동차 keypoints
```yaml
# car_pose.yaml
names:
  0: car

kpt_shape: [4, 3]  # 4개 keypoints, (x, y, visible)

# keypoints: 네 바퀴 위치
# 0: front_left_wheel
# 1: front_right_wheel
# 2: rear_left_wheel
# 3: rear_right_wheel
```

#### 학습 명령어
```bash
yolo pose train data=car_pose.yaml model=yolo11n-pose.pt epochs=100
```

#### 장점/단점
| 장점 | 단점 |
|------|------|
| 객체별 "자세" 추적 가능 | 데이터셋 라벨링 필요 (시간 많이 듦) |
| 우리 Re-ID 코드 그대로 사용 가능 | keypoint 정의가 객체마다 다름 |

---

### 방법 2: 일반 Detection + Pose 결합

#### 개념
```
yolo11n.pt (80 클래스 감지)
    ↓
사람이면 → yolo11n-pose.pt로 포즈 추출
자동차면 → bbox만 사용 (또는 별도 pose 모델)
```

#### 코드 예시
```python
detect_model = YOLO("yolo11n.pt")      # 일반 감지
pose_model = YOLO("yolo11n-pose.pt")   # 사람 포즈

results = detect_model(frame)

for box in results[0].boxes:
    cls = int(box.cls)
    
    if cls == 0:  # person
        # 포즈 모델로 keypoints 추출
        pose_results = pose_model(crop_image)
        # → 우리 Re-ID 적용
    else:
        # 다른 객체는 bbox 기반 추적
        # → DeepSORT 또는 Appearance Embedding
```

#### 장점/단점
| 장점 | 단점 |
|------|------|
| 데이터셋 라벨링 불필요 | 사람 외 객체는 포즈 Re-ID 불가 |
| 빠르게 구현 가능 | 두 모델 사용 → 속도 저하 |

---

### 방법 3: 동물 Pose 모델 (이미 존재!)

#### Animal Pose 데이터셋
```
AP-10K: 10,000+ 동물 이미지, 17개 keypoints
- 개, 고양이, 말, 소, 양, 사슴 등
```

#### 사전 학습된 모델
```python
# MMPose (오픈소스)
from mmpose.apis import inference_top_down_pose_model

# 동물용 pose 모델 로드
animal_pose_model = init_pose_model('animal_pose_config.py', 'animal_pose.pth')
```

---

## 📊 추천 전략

| 목표 | 추천 방법 |
|------|----------|
| **사람 + 일반 객체** | 방법 2 (Detection + Pose 결합) |
| **동물 Re-ID** | 방법 3 (Animal Pose 모델) |
| **특수 객체 (자동차 등)** | 방법 1 (커스텀 학습) |
| **빠른 MVP** | 사람만 Pose Re-ID, 나머지는 YOLO ID |

---

## 📈 실현 가능성

```
쉬움 ←――――――――――――――――――――――――――――→ 어려움

방법 2          방법 3          방법 1
(코드 수정)     (모델 교체)     (데이터셋 + 학습)
```

---

## 🏗️ Detectron2의 커스텀 학습 방식

### 기본 구조
```
사전 학습된 Backbone (ResNet, etc.)
        ↓
    Feature Pyramid Network (FPN)
        ↓
    Detection Head (ROI Head)  ← 여기를 수정/추가
        ↓
    출력: bbox, class, (mask), (keypoints)
```

### 파인튜닝 vs 처음부터 학습

| 방식 | 설명 | 장점 | 단점 |
|------|------|------|------|
| **Fine-tuning** | 사전 학습 가중치 + 일부 레이어만 재학습 | 빠름, 적은 데이터 | 기존 구조에 제한 |
| **Transfer Learning** | Backbone 고정 + Head만 학습 | 매우 빠름 | 성능 제한 |
| **From Scratch** | 전체 처음부터 학습 | 자유도 높음 | 많은 데이터/시간 필요 |

---

## 🔧 Detectron2에서 커스텀 객체 학습

### 1. 새로운 클래스 추가 (Detection)
```python
from detectron2.config import get_cfg
from detectron2 import model_zoo

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))

# 사전 학습 가중치 로드
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")

# 새로운 클래스 수 설정 (예: 내 객체 3개)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3

# 학습
trainer = DefaultTrainer(cfg)
trainer.train()
```

### 2. 새로운 Keypoints 추가 (Pose)
```python
cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))

# 커스텀 keypoint 설정
cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 10  # 내 객체의 keypoint 수
```

---

## ⚖️ Detectron2 vs YOLO 파인튜닝 비교

| 항목 | Detectron2 | YOLO |
|------|------------|------|
| **프레임워크** | PyTorch (Meta) | Ultralytics |
| **유연성** | 매우 높음 (모듈화) | 중간 |
| **사용 난이도** | 어려움 | 쉬움 |
| **속도** | 느림 | 빠름 |
| **커스텀 학습** | 세밀한 제어 가능 | 간단한 명령어 |

---

## 📝 예시: Detectron2로 "내 객체" 학습

### 데이터셋 준비 (COCO 형식)
```json
{
  "images": [...],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [x, y, w, h],
      "keypoints": [x1, y1, v1, x2, y2, v2, ...]  // 커스텀 keypoints
    }
  ],
  "categories": [
    {"id": 1, "name": "my_object", "keypoints": ["point1", "point2", ...]}
  ]
}
```

### 학습 코드
```python
from detectron2.data import DatasetCatalog, MetadataCatalog

# 데이터셋 등록
DatasetCatalog.register("my_dataset", lambda: load_my_data())
MetadataCatalog.get("my_dataset").set(
    thing_classes=["my_object"],
    keypoint_names=["point1", "point2", ...],
    keypoint_flip_map=[...]
)

# 학습
cfg.DATASETS.TRAIN = ("my_dataset",)
trainer = DefaultTrainer(cfg)
trainer.train()
```

---

## 🚗 우리 프로젝트에 적용한다면?

### 시나리오: 자동차 바퀴 위치로 Re-ID

```
1. Detectron2로 자동차 + 바퀴 keypoints 학습
   - 4개 keypoints: 앞왼쪽, 앞오른쪽, 뒤왼쪽, 뒤오른쪽 바퀴

2. 우리 PoseTracker에 통합
   - normalize_keypoints() 수정 (4개 keypoints용)
   - 바퀴 간 거리 비율로 Re-ID

3. 결과
   - 자동차도 "포즈 기반" Re-ID 가능!
```

---

## 📚 용어 정리

| 용어 | 의미 |
|------|------|
| **Fine-tuning** | 사전 학습 모델의 일부/전체를 새 데이터로 재학습 |
| **Transfer Learning** | 사전 학습 지식을 새 작업에 활용 |
| **Head 추가** | 기존 Backbone 위에 새로운 출력 레이어 추가 |
| **Backbone** | 이미지에서 특징을 추출하는 기본 네트워크 (ResNet 등) |
| **FPN** | Feature Pyramid Network, 다양한 크기의 객체 감지용 |
| **ROI Head** | Region of Interest Head, 객체별 분류/위치 예측 |

---

## 🎯 결론

**Detectron2는 커스터마이징에 특화된 프레임워크**
- YOLO보다 복잡하지만, **세밀한 제어가 필요할 때** 강력
- 새로운 객체/keypoints 학습에 적합
- 우리 PoseTracker의 Re-ID 로직과 결합 가능

**YOLO는 빠른 프로토타이핑에 적합**
- 간단한 명령어로 학습 가능
- 실시간 추론에 강점
- 커스터마이징은 제한적

