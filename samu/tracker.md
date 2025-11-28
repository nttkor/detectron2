## PoseTracker 모듈 개요 (`tracker.py`)

### 목적
- **YOLO11 포즈 모델(`yolo11n-pose.pt`)** 을 사용해 영상에서 사람의 관절 키포인트를 추론하고,
- 프레임 간 **포즈(자세) 유사도**를 이용해 사람별 **커스텀 ID**를 부여·유지하는 트래커를 구현한 코드입니다.
- YOLO 내부 트래커 ID와는 독립적인, **포즈 기반 보조 ID 시스템** 예제로 설계되었습니다.

---

### 전체 처리 흐름
1. **YOLO 포즈 모델 로드**
   - `PoseTracker.__init__` 에서 `YOLO(model_path)` 호출로 `yolo11n-pose.pt` 모델을 메모리에 로드.
   - `similarity_threshold` 파라미터로 포즈 유사도 임계값을 설정 (기본 0.5).

2. **동영상 프레임 읽기 (`__main__` 블록)**
   - 기본 경로: `D:/git/detectron2/video/movepeople753.mp4`
   - `cv2.VideoCapture(video_path)` 로 프레임 스트림 생성.
   - 루프에서 매 프레임을 읽어 `PoseTracker.process_frame(frame)` 에 전달.

3. **포즈 추론 + 기본 시각화**
   - `process_frame` 내부에서:
     - `results = self.model.track(frame, persist=True, verbose=False)` 호출.
     - `results[0].keypoints` 에서 각 사람의 포즈 키포인트(17개)를 얻고,
     - `results[0].boxes.xyxy` 에서 각 사람의 바운딩 박스를 얻음.
     - `results[0].plot()` 으로 YOLO 기본 시각화 이미지를 생성.

4. **포즈 정규화 (`normalize_keypoints`)**
   - 키포인트 배열 `(N, 3)` (x, y, conf)을 받아:
     - conf > threshold 인 키포인트만 사용.
     - 왼쪽/오른쪽 엉덩이(index 11, 12)의 중간점을 `mid_hip` 으로 계산해 모든 포인트를 `-mid_hip` 만큼 이동.
     - 왼쪽/오른쪽 어깨(index 5, 6)의 중간점 `mid_shoulder` 와 `mid_hip` 사이 거리를 `torso_length` 로 정의.
     - `normalized = (coords - mid_hip) / torso_length` 로 크기까지 정규화.
   - 유효 키포인트가 부족하거나 `torso_length` 가 매우 작은 경우는 빈 배열을 반환해 이후 유사도 계산에서 제외.

5. **포즈 유사도 계산 (`get_pose_similarity`)**
   - 두 사람의 정규화된 포즈 벡터(`kpts1`, `kpts2`)를 받아:
     - 형태가 다르거나 비어 있으면 `np.inf` 반환.
     - 그렇지 않으면 `np.linalg.norm(kpts1 - kpts2)` 로 유클리드 거리 계산.
   - 값이 작을수록 두 포즈가 비슷하다는 의미.

6. **ID 할당 로직 (`process_frame`)**
   - `self.person_id_to_pose_history` 에 이전 프레임까지의 ID별 최신 정규화 포즈를 보관.
   - 현재 프레임에서 각 사람에 대해:
     1. 정규화된 포즈를 구하고,
     2. 이력에 있는 모든 ID와 유사도(`get_pose_similarity`)를 비교해 최소 거리와 ID를 찾음.
     3. 최소 거리가 `similarity_threshold` 보다 작고, ID가 존재하면 **해당 ID 재사용**.
     4. 아니면 새 ID(`self.next_person_id`) 발급 후 1 증가.
   - 현재 프레임에서 사용된 ID와 포즈만 모아 `newly_assigned_pose_history` 를 만들고,
     프레임 끝에서 `self.person_id_to_pose_history = newly_assigned_pose_history` 로 교체.

7. **커스텀 ID 시각화**
   - YOLO가 그려준 `annotated_frame = results[0].plot()` 위에:

     ```python
     for idx, assigned_id in current_frame_assigned_ids.items():
         bbox = results[0].boxes.xyxy[idx].cpu().numpy()
         x1, y1, x2, y2 = map(int, bbox)
         cv2.putText(
             annotated_frame,
             f"ID: {assigned_id}",
             (x1, max(20, y1 - 10)),
             cv2.FONT_HERSHEY_SIMPLEX,
             0.8,
             (0, 255, 0),
             2,
         )
     ```

   - 이렇게 해서 각 바운딩 박스 위에 `ID: N` 형식의 **포즈 기반 ID** 를 표기.
   - 메인 루프에서 `cv2.imshow` 로 실시간에 가깝게 확인 가능.

---

### 기술적 특징 / 한계
- **특징**
  - 카메라 움직임, 거리 변화에 비교적 강한 **포즈 형태 기반 ID 유지**.
  - YOLO 내장 트래커 ID와 별개로 동작하므로, 끊긴 트랙을 포즈로 다시 연결하는 보조 수단으로 활용 가능.

- **한계**
  - 포즈가 크게 변하거나, 사람이 가려지는 구간(occlusion)이 길면 유사도가 급변할 수 있음.
  - 2D 키포인트만 사용하기 때문에, 비슷한 체형/자세의 다른 사람을 완벽히 구분하기는 어렵다.
  - 실제 서비스에서는 외형 임베딩(appearance embedding) + 포즈 유사도를 함께 사용하는
    하이브리드 접근이 더 안정적일 수 있음.

네, **지금 코드도 포즈로 ID를 만들고 있지만 “실행할 때마다 같은 번호가 보장되지는 않아요.”**  
어디까지 보장되고, 왜 매번 숫자가 달라지는지 정리해볼게요.

### 1. 현재 코드가 하는 일 (한 번 실행 안에서)

- `tracker.py` 실행하면:

  ```python
  tracker = PoseTracker(...)
  self.person_id_to_pose_history = {}
  self.next_person_id = 1
  ```

  로 **ID 이력과 다음 ID가 1부터 초기화**됩니다.
- 첫 프레임에서:
  - YOLO가 검출한 사람 순서대로 `normalize_keypoints` → 포즈 벡터 만든 뒤
  - 이력이 비어 있으니 모두 `ID: 1, 2, 3, ...` 순서로 새로 부여.
- 그 다음 프레임부터는:
  - 이전 프레임의 포즈와 유사도(`get_pose_similarity`)를 비교해서
  - 거리 < `similarity_threshold` 이면 **그 ID를 이어 쓰고**, 아니면 새 ID 발급.

즉, **한 번 실행하는 동안에는** 같은 사람이 프레임을 따라 이동해도  
포즈가 크게 변하지 않는 한 같은 ID를 유지하려고 합니다.

### 2. 실행할 때마다 ID가 달라지는 이유

- 프로그램을 다시 실행하면:
  - `PoseTracker`가 새로 생성되면서 `next_person_id`가 다시 1부터 시작합니다.
  - 첫 프레임에서 YOLO가 검출한 **사람의 순서(인덱스)** 가 매 실행마다 100% 동일하다고 보장되지 않습니다.
    - GPU 연산, NMS 정렬, decode 순서 등에서 미세한 비결정성이 있어서  
      어떤 사람에게 `ID:1`이 붙을지, `ID:2`가 붙을지 **초기 번호는 매번 달라질 수 있습니다.**
- 포즈 기반 로직은 “**이전 프레임의 누구와 비슷하냐**”만 보는 거라,
  - 프로그램을 새로 켰을 때 “이 사람은 항상 1번, 저 사람은 항상 2번” 같은 **전역적인 고정 번호**를 보장해 주지는 않습니다.
  - 그냥 “이번 실행에서, 첫 등장 순서 + 이후 포즈 연속성” 기준으로만 ID가 만들어지는 구조입니다.

### 3. 요약

- **포즈 기반 ID는 “실행 중 프레임 사이의 일관성”만 책임지고**,  
  “프로그램을 다시 켤 때도 항상 같은 사람에게 같은 번호를 준다”까지는 설계되어 있지 않습니다.
- 그걸 원한다면:
  - 첫 프레임에서 사람별로 어떤 기준(위치, 이름, 초기 프레임 인덱스 등)을 정해서 **ID 매핑을 파일에 저장**하고,
  - 다음 실행 때 그 매핑을 다시 불러와 이어 쓰는 별도 로직이 필요합니다.

지금 코드에서 **동일 실행 안에서 움직이면서도 ID가 자주 바뀐다**면,  
그건 `similarity_threshold` 값이나 포즈 정규화 쪽을 손봐야 하는 문제고,  
그런 사례가 있으면 구체적인 프레임 상황을 알려주시면 그 부분은 따로 잡아볼 수 있습니다.

[3 tools called]

이제 요청하신 대로, **매 실행 때 JSON을 읽어서 포즈 기반 ID를 매치하고, 매치된 사람은 라벨 색을 다르게** 처리하도록 바꿔 두었습니다.

### 1. 실행 시 JSON에서 이전 ID/포즈 로드

- `PoseTracker`에 `load_global_ids(json_path)` 메서드를 추가했습니다.
  - `samu/output/movepeople753_pose_ids.json` 이 있으면 열어서,
  - 각 항목의 `keypoints`로 다시 `normalize_keypoints()`를 돌려
    **`track_id → 마지막 포즈` 맵(`person_id_to_pose_history`)** 을 만듭니다.
  - 이때 얻은 ID들을 `preexisting_ids`에 저장하고,
    `next_person_id`를 `max(ID)+1`로 맞춰서 **이전 실행의 ID를 그대로 이어 쓰도록** 했습니다.

- `__main__`에서:

```python
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "output")
output_json = os.path.join(output_dir, "movepeople753_pose_ids.json")

tracker = PoseTracker(...)
tracker.load_global_ids(output_json)
```

이렇게 해서 프로그램을 다시 켤 때도, **기존 JSON에 있던 사람들은 같은 ID로 매칭될 수 있게** 했습니다.

### 2. 라벨 색상: 기존에 있던 ID vs 새로 생긴 ID

`process_frame()`에서 ID를 그릴 때:

```python
color = (255, 0, 0) if assigned_id in self.preexisting_ids else (0, 255, 0)
cv2.putText(..., color, 2)
```

- **이전 JSON에 이미 있던 ID** → `preexisting_ids`에 포함 → **빨간색** 라벨  
- **이번 실행에서 처음 만들어진 ID** → `preexisting_ids`에 없음 → **초록색** 라벨

그래서:

- 첫 실행 때는 모두 새 ID라 초록색만 나오고,
- 두 번째 실행부터는, JSON에 기록된 사람과 포즈가 비슷하게 매칭되면  
  **같은 ID 번호 + 빨간색 라벨**로 표시됩니다.

### 3. JSON 로그 누적 저장

- `export_tracking_log()`를 수정해서:
  - 기존 JSON이 있으면 먼저 읽어 `existing`에 넣고,
  - 이번 실행의 `tracking_log`를 뒤에 붙여서 다시 저장합니다.

```python
combined = existing + self.tracking_log
json.dump(combined, ...)
```

즉, 여러 번 실행해도 `movepeople753_pose_ids.json` 안에  
**모든 실행의 프레임/ID/포즈 기록이 계속 누적**되고,  
다음 실행에서 이 기록을 기반으로 포즈 매칭을 시도하게 됩니다.