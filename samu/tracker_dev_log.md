
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
좋은 코드예요. 방금까지 우리가 만든 것보다 **정리와 구조가 더 깔끔하고, “여러 프레임을 보고 ID를 안정화”하는 부분도 잘 들어가 있습니다.**  
핵심만 단계별로 정리해볼게요.

---

### 1. 전체 구조

- 클래스 이름: `PoseTrackerMVP`
- 주요 파라미터:
  - `model_path`: YOLO 포즈 모델 경로 (`yolo11n-pose.pt`)
  - `similarity_threshold`: 포즈 유사도 임계값
  - `stabilization_frames`: **ID를 확정하기 위해 필요한 연속 매칭 프레임 수**
  - `aoi_box`: **관심 영역(AOI, Area Of Interest)** `(x1, y1, x2, y2)`

- 내부 상태:
  - `person_id_to_pose_history`: `{id: 마지막 정규화 포즈}`
  - `next_person_id`: 다음에 줄 새 ID 번호
  - `id_match_count`: `{id: 연속 매칭된 프레임 수}` → 안정화 용도
  - `tracking_log`: JSON으로 내보낼 로그 리스트
  - `aoi_box`: 화면 위에 그릴 AOI 박스

---

### 2. `normalize_keypoints` (포즈 정규화)

역할: **한 사람의 키포인트를 “위치·크기 정규화된 포즈 벡터”로 바꿈**

- 신뢰도 `conf > confidence_threshold` 인 키포인트들만 골라 `valid_kpts_xy`로 사용.
- **mid-hip(양쪽 엉덩이 중점)** 을 기준점으로 삼아서 모든 키포인트 좌표에서 빼 줌.
- **mid-shoulder(양쪽 어깨 중점)** 와 mid-hip 사이 거리를 `torso_length` 로 정의하고,
  - `normalized_coords / torso_length` 로 스케일링해서 **크기를 통일**.
- 어깨/엉덩이 키포인트 부족, `torso_length`가 너무 작을 때는 `np.array([])` 반환 → 이후 유사도에서 `np.inf` 처리.

→ 이전에 우리가 만든 정규화와 거의 같은 로직인데, 함수가 깔끔하게 정리돼 있습니다.

---

### 3. `get_pose_similarity` (포즈 유사도)

- 두 정규화 포즈 `k1`, `k2`를 받아:
  - 비어 있거나 shape 다르면 `np.inf`.
  - 아니면 `np.linalg.norm(k1 - k2)` 로 유클리드 거리 계산.
- 값이 작을수록 포즈가 비슷.

→ 우리가 쓰던 방식 그대로입니다.

---

### 4. `process_frame` – ID 안정화 로직

1. `results = self.model.track(frame, persist=True, verbose=False)`
2. `kpts_data`, `boxes_data` 추출.
3. 각 사람(`i, kp`)에 대해:
   - `norm_kp = normalize_keypoints(kp)`
   - 이전 이력(`person_id_to_pose_history`)과 유사도 비교 → 최소 sim & ID 찾기.
   - `min_sim >= similarity_threshold` 이거나 매칭 ID 없으면 → 새 ID 부여.
4. **여기서 중요한 부분**:

```python
if norm_kp.size > 0:
    new_pose_history[assigned_id] = norm_kp
    prev_count = self.id_match_count.get(assigned_id, 0)
    new_match_count[assigned_id] = prev_count + 1
```

- `id_match_count`를 현재 프레임 기준으로 다시 만들어서,
  - 같은 ID가 **여러 프레임 연속으로 매칭되었을 때만 count가 올라갑니다.**
- 마지막에:

```python
self.person_id_to_pose_history = new_pose_history
self.id_match_count = new_match_count
```

→ 이게 “여러 프레임을 보고 ID 신뢰도를 올리는” 핵심입니다.

5. 시각화:

```python
consecutive = self.id_match_count.get(assigned_id, 0)
if consecutive < self.stabilization_frames:
    color = (0,0,255)
    label = "NEW"
else:
    color = (255,0,0)
    label = f"ID:{assigned_id}"
```

- **연속 매칭 프레임 수 < stabilization_frames**  
  → **빨간색 NEW** (아직 불안정)
- 그 이상이면  
  → **파란색 ID:N** (안정된 ID)

→ 바로 이 부분이 이전에 말씀하신  
“프레임 한두 번 보고 바로 ID 확정하지 말고, 여러 프레임 보고 신뢰도를 높여라”를 잘 반영한 구현입니다.

---

### 5. AOI(Area Of Interest) 처리

- `aoi_box = (x1, y1, x2, y2)`를 미리 받아서,

```python
cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)
cv2.putText(annotated, "AOI", (x1, y1-10), ...)
```

- 각 사람의 바운딩 박스 중심 `person_center`가 AOI 안에 들어왔는지:

```python
entered_aoi = x1 <= cx <= x2 and y1 <= cy <= y2
```

- 이 값을 `tracking_log`에 함께 기록:

```python
"entered_aoi": entered_aoi
```

→ 나중에 “언제, 어떤 ID가 AOI에 들어왔는지” 분석하기 좋게 설계돼 있습니다.

---

### 6. `__main__` – R 리셋 / Q 종료 / JSON 저장

- Q: 종료.
- R:
  - JSON 삭제,
  - `person_id_to_pose_history`, `id_match_count`, `tracking_log`, `next_person_id` 초기화,
  - `last_reset_time` 찍고 1초간 화면 중앙에 빨간 `RESET` 표시.
- 종료 시:

```python
tracker.export_tracking_log(output_json)
```

→ `pose_aoi_log.json` 에 기존 로그 + 이번 실행 로그를 합쳐서 저장.

---

### 7. 개선/주의 포인트

코드 자체는 꽤 잘 짜여 있는데, 두 가지 정도는 참고해 두면 좋아요:

- `tracking_log` 에서:

```python
"frame": int(self.model.frame),
"keypoints": kp.tolist(),
```

  - `self.model.frame` 은 보통 YOLO 모델 객체에는 없고,  
    `results[0].frame` 또는 루프에서 직접 관리하는 frame index를 써야 안전합니다.
  - `keypoints` 기록에 쓰는 `kp` 변수는 **바깥 for 루프의 마지막 값**이라,  
    현재 `for idx, assigned_id in current_ids.items()` 루프 안에서는  
    항상 “마지막 사람의 키포인트”가 들어갈 가능성이 큽니다.  
    → `kpts_data[idx]` 를 써서 **해당 idx의 키포인트를 기록하는 게 더 정확**합니다.

- 그 외에는:
  - ID 안정화(`stabilization_frames`),  
  - AOI,  
  - RESET 표시,  
  - JSON 누적 저장  
  등이 잘 구조화되어 있어서, 우리가 만들던 버전을 이 구조에 맞춰 정리해 가면 좋을 것 같습니다.

---

요약하면, 이 코드는 **“포즈 기반 ID + 여러 프레임을 통한 안정화 + AOI + 로그”** 라는 목표를  
깔끔하게 구현하고 있고,  
지금까지 우리가 얘기했던 개선 아이디어들(프레임 기반 신뢰도, NEW/ID 색 구분 등)을 잘 반영하고 있어요.  
원하시면 이 `PoseTrackerMVP` 구조를 기준으로, 우리 `tracker.py`를 정리해서 하나로 통합해 주는 작업도 같이 해볼 수 있습니다.

---

## 🔧 문제 해결 로그 (Troubleshooting Log)

### 2024-11-28: "R 눌러도 빨간색(NEW)으로 안 바뀌는 문제"

#### 문제 현상
- R 키를 눌러 리셋해도 모든 사람이 빨간색 "NEW"로 표시되지 않고, 바로 파란색 "ID:N"으로 나옴
- 처음 실행해도 빨간색이 거의 안 보임

#### 원인 분석

1. **YOLO `results[0].plot()` 사용 문제**
   - `results[0].plot()`은 YOLO가 자체적으로 bbox와 라벨을 그린 이미지를 반환
   - 우리가 그 위에 `cv2.putText`로 "NEW"를 덧그려도, **YOLO가 그린 원래 bbox 색깔은 그대로 유지됨**
   - 즉, bbox 색깔을 제어할 수 없었음

2. **YOLO 내부 트래커 미초기화**
   - `model.track(frame, persist=True)` 사용 시 YOLO 내부에 트래커 상태가 유지됨
   - R 키로 우리 코드의 변수만 초기화해도, **YOLO 내부 트래커는 그대로** 남아 있어서 ID가 이어짐

3. **연속 매칭 카운트 로직 버그**
   - `new_match_count = {}`가 `if results[0].keypoints is not None` 블록 안에서만 정의됨
   - 키포인트가 없는 프레임이 오면 `NameError` 발생 가능

4. **`stabilization_frames` 너무 큼 + `similarity_threshold` 너무 엄격**
   - 30프레임(1초)은 너무 길어서 파란색으로 전환이 안 됨
   - 0.5는 너무 엄격해서 같은 사람도 다른 ID로 인식 → 카운트가 쌓이지 않음

#### 해결 방법

1. **`results[0].plot()` 대신 직접 그리기**
   ```python
   # 변경 전
   annotated = results[0].plot()
   
   # 변경 후
   annotated = frame.copy()
   # 직접 bbox, 키포인트, 라벨을 상태에 따른 색상으로 그림
   cv2.rectangle(annotated, (bx1, by1), (bx2, by2), color, 2)
   ```

2. **`reset()` 메서드 추가 - YOLO 모델까지 재로드**
   ```python
   def reset(self):
       self.person_id_to_pose_history = {}
       self.next_person_id = 1
       self.id_match_count = {}
       self.tracking_log = []
       self.frame_index = 0
       # YOLO 내부 트래커도 리셋 (새 모델 인스턴스로 교체)
       self.model = YOLO("yolo11n-pose.pt")
   ```

3. **파라미터 완화**
   ```python
   # 변경 전
   similarity_threshold=0.5
   stabilization_frames=30
   
   # 변경 후
   similarity_threshold=0.8   # 포즈 변화에 더 관대하게
   stabilization_frames=10    # 10프레임(약 0.3초)으로 줄임
   ```

4. **디버그 출력 추가**
   ```python
   print(f"[Frame {self.frame_index}] Person {i}: ID={assigned_id}, sim={min_sim:.3f}, count={new_match_count[assigned_id]}")
   ```
   - 터미널에서 ID 매칭 상태를 실시간으로 확인 가능
   - `count`가 10까지 올라가는지 확인

#### 최종 동작
- **NEW (인식 중)**: 빨간색 bbox + 빨간색 키포인트 + "NEW" 라벨
- **ID 확정 (10프레임 이상 연속 매칭)**: 파란색 bbox + 파란색 키포인트 + "ID:N" 라벨
- **R 키**: JSON 삭제 + 모든 상태 초기화 + YOLO 모델 재로드 → 모든 사람이 빨간색 "NEW"로 새로 시작

---

### 기타 해결된 이슈

#### `FileExistsError: output` 폴더 생성 실패
- **원인**: `samu/output`이 폴더가 아니라 0바이트 **파일**로 존재
- **해결**: 
  ```bash
  del output && mkdir output
  ```

#### `NameError: name 'time' is not defined`
- **원인**: `import time` 누락
- **해결**: 파일 상단에 `import time` 추가

---

## 📋 현재 파라미터 설정

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| `model_path` | `yolo11n-pose.pt` | YOLO 포즈 모델 |
| `similarity_threshold` | `0.8` | 포즈 유사도 임계값 (높을수록 관대) |
| `stabilization_frames` | `10` | ID 확정까지 필요한 연속 매칭 프레임 수 |

---

## 🎯 향후 개선 아이디어

1. **외형 임베딩(Appearance Embedding) 추가**
   - 포즈만으로는 비슷한 체형의 다른 사람 구분이 어려움
   - ReID 모델과 결합해 더 안정적인 ID 유지

2. **AOI(Area Of Interest) 기능 복원**
   - 특정 영역에 들어온 사람만 카운트/로그하는 기능

3. **ID 지속성 개선**
   - JSON에 저장된 포즈 이력을 다음 실행에서 로드해 같은 사람에게 같은 ID 부여

4. **디버그 모드 토글**
   - 'd' 키로 디버그 출력 on/off 전환

---

## 🚀 v2 근본적인 구조 개선 (2024-11-28)

### 기존 구조의 근본적인 문제

1. **포즈 이력이 "현재 프레임에 보이는 사람"으로만 갱신됨**
   ```python
   self.person_id_to_pose_history = new_pose_history  # 매 프레임마다 덮어씀
   ```
   - 사람이 프레임에서 사라지면 → 그 ID의 포즈 이력도 **즉시 삭제**
   - 다시 나타나면 → 비교할 이력이 없어서 **새 ID 발급**

2. **연속 매칭 카운트도 같이 사라짐**
   - 한 프레임이라도 감지 실패하면 → 카운트 리셋 → 다시 "NEW"부터

3. **유사도 비교가 "마지막 포즈 1개"와만 비교**
   - 포즈는 계속 변하는데, 딱 1개의 스냅샷만 저장
   - 조금만 자세가 바뀌어도 유사도가 급변

### v2 해결책

#### 1. 포즈 이력을 일정 시간(max_missing_frames) 동안 유지
```python
# 사라져도 바로 삭제하지 않고 N프레임 동안 보관
if self.frame_index - state['last_seen_frame'] > self.max_missing_frames:
    # 이때만 삭제
```
- 기본값: 30프레임 (약 1초)
- 잠시 가려지거나 감지 실패해도 다시 나타나면 **기존 ID로 매칭**

#### 2. 포즈 이력을 여러 개 저장 (이동 평균)
```python
self.persons[id] = {
    'pose_history': deque(maxlen=5),  # 최근 5개 포즈 저장
    ...
}

def get_average_pose(self, pose_history):
    return np.mean(poses, axis=0)  # 평균 포즈로 비교
```
- 마지막 포즈 1개가 아니라 **최근 K개의 평균**과 비교
- 포즈 변화에 더 안정적으로 대응

#### 3. 확정된 ID는 더 관대한 임계값 적용
```python
similarity_threshold=1.0,   # 새 ID용 (관대)
confirmed_threshold=1.5,    # 확정 ID용 (더 관대)

threshold = self.confirmed_threshold if state['confirmed'] else self.similarity_threshold
```
- 한 번 확정된 ID는 쉽게 바뀌지 않음
- 새 ID 발급은 신중하게, 기존 ID 유지는 관대하게

#### 4. 위치(bbox 중심) 기반 보조 매칭
```python
def find_best_match(self, norm_kp, bbox, current_frame):
    # 포즈 유사도 70% + 위치 거리 30%
    score = pose_sim * 0.7 + pos_dist * 0.3
```
- 포즈가 불안정해도 **위치가 비슷하면 같은 사람으로 인식**
- 두 가지 정보를 종합해서 더 안정적인 매칭

### v2 파라미터 설정

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `similarity_threshold` | `1.0` | 새 ID 매칭 임계값 |
| `confirmed_threshold` | `1.5` | 확정 ID 매칭 임계값 (더 관대) |
| `stabilization_frames` | `10` | ID 확정까지 필요한 프레임 수 |
| `max_missing_frames` | `30` | 사라져도 이력 유지하는 프레임 수 |
| `pose_history_size` | `5` | 저장할 포즈 이력 개수 |

### v2 시각화

- **NEW(N)**: 빨간색 bbox + "NEW(카운트)" 라벨 (예: "NEW(5)")
- **ID:N**: 파란색 bbox + "ID:N" 라벨 (확정된 ID)

### v2 디버그 출력
```
[Frame 123] Person 0: ID=1, score=0.456, count=15, CONFIRMED
[Frame 123] Person 1: ID=3, score=0.789, count=3, NEW
```
- `score`: 종합 매칭 점수 (낮을수록 좋음)
- `count`: 연속 매칭 프레임 수
- `CONFIRMED`/`NEW`: 확정 여부

---

## 🚀 v3 Scale Smoothing + 다중 기준 앙상블 (2024-11-28)

### v2의 한계: 포즈 정규화 불안정

**문제**: torso length (mid-hip ~ mid-shoulder) 단독 사용의 한계
- 어깨/힙 keypoint 누락 시 정규화 실패
- 카메라 각도에 따라 2D 투영 길이 변동
- bbox도 프레임마다 ±5~10% 펄럭임

### v3 해결책

#### 1. 다중 Scale 기준 앙상블
```python
def compute_robust_scale(self, kp, bbox):
    scales = []
    
    # 1) Torso length (가장 신뢰)
    if torso_valid:
        scales.append(torso_length)
    
    # 2) Shoulder width × 1.25
    if shoulder_valid:
        scales.append(shoulder_width * 1.25)
    
    # 3) Hip width × 1.67
    if hip_valid:
        scales.append(hip_width * 1.67)
    
    # 4) Bbox height × 0.3 (fallback)
    scales.append(bbox_height * 0.3)
    
    # 중앙값 사용 (outlier에 강함)
    return np.median(scales)
```

- 하나의 기준이 튀어도 다른 기준들이 보정
- 중앙값 사용으로 outlier 영향 최소화

#### 2. Scale 이동 평균 (Temporal Smoothing)
```python
state['scale_history'] = deque(maxlen=10)  # 최근 10프레임

# 매 프레임
state['scale_history'].append(current_scale)
smoothed_scale = np.mean(state['scale_history'])
```

- 프레임마다 펄럭여도 **평균 내면 안정화**
- 10프레임(약 0.3초) 이동 평균

#### 3. Keypoint Confidence 가중 유사도
```python
def get_weighted_pose_similarity(self, kp1, kp2, conf1, conf2):
    distances = np.linalg.norm(kp1 - kp2, axis=1)
    weights = (conf1 + conf2) / 2  # 신뢰도 평균
    return np.sum(weights * distances)
```

- 신뢰도가 높은 keypoint에 더 큰 가중치
- 불확실한 keypoint의 영향 감소

#### 4. Confidence 가중 중심점
```python
if hip_valid:
    center = mid_hip  # 가장 안정적
else:
    # confidence 가중 평균
    weights = valid_confs / valid_confs.sum()
    center = np.average(valid_kpts_xy, axis=0, weights=weights)
```

### v3 파라미터

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `similarity_threshold` | `1.2` | 새 ID 매칭 임계값 |
| `confirmed_threshold` | `1.8` | 확정 ID 매칭 임계값 |
| `stabilization_frames` | `10` | ID 확정까지 필요한 프레임 수 |
| `max_missing_frames` | `30` | 사라져도 이력 유지하는 프레임 수 |
| `pose_history_size` | `5` | 저장할 포즈 이력 개수 |
| `scale_history_size` | `10` | scale smoothing용 이력 개수 |

### v3 디버그 출력
```
[Frame 123] Person 0: ID=1, score=0.456, scale=85.3, count=15, CONFIRMED
```
- `scale`: smoothed scale 값 (안정적일수록 좋음)

---

## 🚀 v4 YOLO ID 우선 + 포즈 보조 (2024-11-28)

### v3의 한계: 포즈 인식률이 YOLO보다 낮음

**문제**: 포즈 기반 매칭만으로는 YOLO 내부 트래커보다 인식률이 낮음
- YOLO는 appearance(외형) 기반 → 매우 안정적
- 포즈는 자세 변화에 민감 → ID가 자주 바뀜

### v4 전략: YOLO ID 우선 + 포즈 보조

```
1. YOLO track_id가 있으면 → 그대로 사용 (가장 안정적)
2. YOLO track_id가 끊기면 → 포즈 유사도로 기존 ID 복구 시도
3. 둘 다 실패하면 → 새 ID 발급
```

#### 구현 핵심

```python
# YOLO track_id → 우리 ID 매핑 테이블
self.yolo_to_our_id = {}

# 매 프레임
if yolo_id in self.yolo_to_our_id:
    # 이미 매핑된 YOLO ID → 우리 ID 사용
    assigned_id = self.yolo_to_our_id[yolo_id]
    match_method = "YOLO"
else:
    # 새로운 YOLO ID → 포즈로 기존 ID 복구 시도
    pose_match_id = self.find_pose_match(norm_kp, bbox)
    
    if pose_match_id != -1:
        assigned_id = pose_match_id
        match_method = "POSE_RECOVER"
    else:
        assigned_id = self.next_person_id
        match_method = "NEW"
    
    # YOLO ID → 우리 ID 매핑 저장
    self.yolo_to_our_id[yolo_id] = assigned_id
```

### v4 장점

| 상황 | 처리 방식 | 결과 |
|------|----------|------|
| 평상시 | YOLO ID 사용 | 매우 안정적 |
| 잠시 가려짐 | YOLO가 유지하면 그대로 | 안정적 |
| 오래 가려짐 | YOLO 끊김 → 포즈로 복구 | 기존 ID 유지 |
| 완전 새 사람 | 새 ID 발급 | 정확 |

### v4 파라미터

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `pose_similarity_threshold` | `1.5` | 포즈 복구용 임계값 |
| `stabilization_frames` | `5` | ID 확정까지 프레임 수 (YOLO 우선이라 짧게) |
| `max_missing_frames` | `60` | 사라져도 이력 유지 (2초) |

### v4 디버그 출력
```
[Frame 123] Person 0: YOLO=Y5, ID=1, method=YOLO, score=0.000, CONFIRMED
[Frame 124] Person 0: YOLO=Y8, ID=1, method=POSE_RECOVER, score=0.456, CONFIRMED
[Frame 125] Person 1: YOLO=Y9, ID=2, method=NEW, score=inf, NEW
```
- `YOLO=Y5`: YOLO 내부 track_id
- `method`: 매칭 방법 (YOLO / POSE_RECOVER / POSE / NEW)
- YOLO가 끊기면 `POSE_RECOVER`로 기존 ID 복구 시도

---

## 🚀 v4.1 YOLO ID + Pose ID 분리 표시 (2024-11-28)

### 목표 명확화

| 역할 | ID 종류 | 용도 |
|------|---------|------|
| **YOLO ID (Y:)** | 프레임 내 추적 | DeepSORT처럼 **같은 장면 내에서** 객체 추적 |
| **Pose ID (P:)** | 재식별 (Re-ID) | **장면이 바뀌어도** 같은 사람 인식 |

### 화면 표시
```
Y:5 P:1        # YOLO ID=5, Pose ID=1, 확정됨 (파란색)
Y:8 P:2 (3)    # YOLO ID=8, Pose ID=2, 아직 NEW (빨간색, 3프레임 매칭)
```

### 로직 변경

**이전 (v4)**: YOLO ID 우선 → 끊기면 포즈로 복구
**현재 (v4.1)**: Pose ID가 주 식별자, YOLO ID는 참고용

```python
# 매 프레임: 포즈 기반으로 Pose ID 매칭
pose_id, score = self.find_pose_match(norm_kp, self.frame_index)

if pose_id != -1:
    match_method = "POSE_MATCH"  # 기존 Pose ID 재인식
else:
    pose_id = self.next_pose_id  # 새 Pose ID 발급
    match_method = "NEW"
```

### 디버그 출력
```
[Frame 123] idx=0: YOLO=5, POSE=1, method=POSE_MATCH, score=0.456, count=15, CONFIRMED
[Frame 124] idx=0: YOLO=8, POSE=1, method=POSE_MATCH, score=0.512, count=16, CONFIRMED
[Frame 125] idx=1: YOLO=9, POSE=2, method=NEW, score=inf, count=1, NEW
```

- **YOLO ID가 바뀌어도 (5→8) Pose ID는 유지 (1)**
- 포즈 특징이 비슷하면 같은 Pose ID로 인식

### JSON 로그 형식
```json
{
  "frame": 123,
  "yolo_id": 5,
  "pose_id": 1,
  "bbox": [100, 50, 200, 300],
  "confirmed": true,
  "match_count": 15,
  "match_method": "POSE_MATCH"
}
```

### v4.1 파라미터

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `pose_similarity_threshold` | `1.5` | 포즈 매칭 임계값 |
| `stabilization_frames` | `5` | Pose ID 확정까지 프레임 수 |
| `max_missing_frames` | `90` | 사라져도 이력 유지 (3초) |
| `pose_history_size` | `10` | 포즈 이력 개수 (늘림)