## YOLO 포즈 트래킹 노트북 개요 (`yolo11_pose.ipynb`)

### 목적
- **YOLO11 Pose 모델**을 사용해 사람의 포즈(관절 키포인트)를 추론하고,
- 동영상 또는 웹캠 스트림에서 **프레임 간 사람 ID를 유지하는 추적(Tracking)** 을 수행하며,
- 향후 **포즈 기반 ID 재할당/분석 로직**을 실험하기 위한 베이스 노트북입니다.

### 사용 환경 및 의존성
- Python + Jupyter Notebook
- 주요 패키지:
  - `ultralytics` (YOLO11)
  - `opencv-python` / `opencv-python-headless`
  - `numpy`
  - `IPython.display` (노트북 내 이미지 표시용)

### 노트북 구성
- **셀 0 – 패키지 설치**
  - `pip install ultralytics opencv-python-headless` 를 통해 필요 패키지를 설치 (최초 1회).

- **셀 1 – 단일 이미지 테스트**
  - `YOLO("yolo11n-pose.pt")` 모델을 로드하고,
  - `ADE_val_00001054.jpg` 한 장에 대해 포즈 추론을 실행해 동작 여부를 빠르게 확인.

- **셀 2 – 공통 import**
  - `YOLO`, `cv2`, `numpy`, `IPython.display` 등 노트북 전체에서 사용하는 모듈을 불러옴.

- **셀 3 – (옵션) gdown 예시**
  - 구글 드라이브 공유 링크를 통해 동영상을 `D:/git/detectron2/video/movepeople753.mp4` 로 다운로드하는 예시 코드.
  - 현재는 로컬에 이미 다운로드해 둔 경우 생략 가능.

- **셀 4 – YOLO Pose 기반 동영상 트래킹**
  - `model = YOLO("yolo11n-pose.pt")` 로 포즈 모델을 로드.
  - `video_path = r"D:/git/detectron2/video/movepeople753.mp4"` 로 로컬 동영상을 지정
    (또는 `video_path = 0` 으로 웹캠 사용 가능).
  - `cv2.VideoCapture` 로 프레임을 읽고,
    각 프레임에 대해 `model.track(img, persist=True)` 로 사람별 포즈 + ID를 추론/추적.
  - `results[0].plot()` 으로 시각화된 프레임을 생성한 뒤,
    `IPython.display.Image` 를 사용해 노트북에서 실시간에 가깝게 표시.

- **셀 6, 7 – 분석/요약 문서**
  - 커스텀 포즈 기반 ID 할당 로직의 개념, 정규화 방법, 포즈 유사도 계산, 장점/한계를
    한국어로 정리한 문서 셀.

### 실행 흐름 요약
1. (필요 시) 셀 0에서 패키지 설치.
2. 셀 1을 실행해 YOLO Pose 모델이 정상 동작하는지 이미지 한 장으로 테스트.
3. 셀 2를 실행해 공통 모듈 import.
4. `video_path` 를 자신의 동영상/웹캠에 맞게 설정.
5. 셀 4를 실행해 동영상 전체에 대해 포즈 추론 + 추적 결과를 노트북에서 확인.

### 향후 확장 아이디어
- `results[0].keypoints` 를 사용해 각 사람의 포즈 벡터를 수집하고,
  - 포즈 패턴에 따라 ID를 재할당하거나,
  - 행동 분류, 비정상 행동 탐지 등으로 확장 가능.
- YOLO 내부 트래커 ID와 **포즈 기반 보조 ID** 를 조합한 하이브리드 추적 전략 실험.


