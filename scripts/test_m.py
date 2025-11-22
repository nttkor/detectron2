import os
import cv2
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

# Detectron2 관련 임포트
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog
from detectron2.modeling import ROI_HEADS_REGISTRY, StandardROIHeads

# =========================================================
# [설정] 경로 및 파라미터 (사용자 환경에 맞게 수정하세요)
# =========================================================

# 1. 학습된 모델 가중치 경로 (.pth 파일)
MODEL_WEIGHTS_PATH = "/home/elicer/dev/gt/data/model/20251121_074740/model_final.pth" 

# 2. 테스트할 이미지 경로
TEST_IMAGE_PATH = "/home/elicer/dev/gt/data/test_data/371_ND_000_FC_176.jpg"

# 3. 결과 저장 경로
OUTPUT_IMAGE_PATH = "result_final.jpg"

# 4. 시각화 임계값 (이 점수 이상인 객체만 표시)
VIS_SCORE_THRESH = 0.6 

# 5. 클래스 목록 (학습 코드와 순서/내용이 정확히 일치해야 합니다)
THING_CLASSES = [
  "vehicle", "bus", "truck", "othercar",
  "motorcycle", "bicycle", "pedestrian", "rider",
  "trafficsign", "trafficlight", "constructionguide", "trafficdrum", 
]
STUFF_CLASSES = [
  "freespace", "curb", "sidewalk", "crosswalk", 
  "roadmark", "whitelane", "yellowlane", 
]
ALL_CLASSES = THING_CLASSES + STUFF_CLASSES

# 1. 거리 정보를 표시할 '중요 객체' 명단 정의
# (여기에 포함된 애들만 '23.5m' 처럼 거리를 띄워줍니다)
DISTANCE_TARGET_CLASSES = [
    "vehicle", "bus", "truck", "othercar", 
    "motorcycle", "bicycle", "pedestrian", "rider",
    # 필요하면 표지판도 포함/제외 가능
    # "trafficsign", "trafficlight", "constructionguide", "trafficdrum"
]

# 2. ★ 아예 화면에 그리지 않을 '숨김 객체' (박스조차 안 그림) ★
# 랜드마크, 차선, 연석 등 불필요한 정보는 여기에 추가하세요.
HIDDEN_CLASSES = [
    "roadmark",   # 랜드마크 (화살표 등)
    "whitelane",  # 흰색 차선
    "yellowlane", # 노란 차선
    "stoplane",   # 정지선
    "curb",       # 연석
    "sidewalk"    # 인도 (필요하면 주석 해제해서 숨기기)
]

# =========================================================
# [핵심] Custom ROI Head 정의 (학습 코드와 동일)
# =========================================================
@ROI_HEADS_REGISTRY.register()
class DistanceROIHeads(StandardROIHeads):
    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        # Box Head의 출력 차원 확인 (기본 1024)
        input_dim = self.box_head.output_shape.channels if hasattr(self.box_head, 'output_shape') else 1024
        
        # [수정 1] 학습 코드와 동일하게 ReLU 및 구조 변경
        self.distance_fc = nn.Sequential(
            nn.Linear(input_dim, 1),
            nn.ReLU() # 음수 방지
        )
        self.max_distance = 100.0 # 정규화 복원용 상수

    def _forward_box(self, features, proposals):
        # Feature Map 준비
        features_list = [features[f] for f in self.box_in_features]
        
        # Box Head 실행
        box_features = self.box_pooler(features_list, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        pred_class_logits, pred_proposal_deltas = self.box_predictor(box_features)

        if self.training:
            # [학습 모드] (test_m.py에서는 실행되지 않으나 구조 유지를 위해 작성)
            pred_normalized = self.distance_fc(box_features)
            losses = self.box_predictor.losses((pred_class_logits, pred_proposal_deltas), proposals)
            losses["loss_distance"] = self._get_distance_loss(pred_normalized, proposals)
            return losses 
        
        else:
            # [추론 모드]
            pred_instances, _ = self.box_predictor.inference((pred_class_logits, pred_proposal_deltas), proposals)
            
            if len(pred_instances) == 0:
                return pred_instances
            
            # Re-Pooling (살아남은 박스들에 대해 다시 특징 추출)
            pred_boxes = [x.pred_boxes for x in pred_instances]
            final_box_features = self.box_pooler(features_list, pred_boxes)
            final_box_features = self.box_head(final_box_features)
            
            # [수정 2] 거리 예측 후 스케일 복원 (0~1 -> 0~100m)
            pred_normalized = self.distance_fc(final_box_features)
            final_distances = pred_normalized * self.max_distance
            
            # 결과 인스턴스에 거리 정보 삽입
            start_idx = 0
            for instances in pred_instances:
                num_boxes = len(instances)
                instances.pred_distances = final_distances[start_idx : start_idx + num_boxes]
                start_idx += num_boxes
                
            return pred_instances

    def _get_distance_loss(self, pred_distances, proposals):
        return 0.0 # 테스트 시엔 미사용


# =========================================================
# [메인] 실행 로직
# =========================================================
def main():
    # 1. Config 설정
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    
    # 모델 구조 설정
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(ALL_CLASSES)
    cfg.MODEL.ROI_HEADS.NAME = "DistanceROIHeads"  # ★ Custom Head 이름 지정
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5    # 모델 내부 1차 필터링 (너무 낮으면 느려짐)
    
    # 가중치 로드
    cfg.MODEL.WEIGHTS = MODEL_WEIGHTS_PATH
    if not os.path.exists(MODEL_WEIGHTS_PATH):
        print(f"Error: 모델 가중치 파일이 없습니다 -> {MODEL_WEIGHTS_PATH}")
        return

    # 디바이스 설정 (GPU 없으면 cpu)
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # 2. 메타데이터 등록 (클래스 이름을 알기 위해 임시 등록)
    # JSON 파일을 로드하지 않고 직접 리스트를 등록합니다.
    test_metadata_name = "my_test_dataset"
    try:
        MetadataCatalog.get(test_metadata_name).set(thing_classes=ALL_CLASSES)
    except:
        pass # 이미 등록되어 있으면 패스
    
    # 3. 예측기 생성
    predictor = DefaultPredictor(cfg)
    print("모델 로드 완료. 예측 시작...")

    # 4. 이미지 로드
    im = cv2.imread(TEST_IMAGE_PATH)
    if im is None:
        print(f"Error: 이미지를 읽을 수 없습니다 -> {TEST_IMAGE_PATH}")
        return

    # 5. 추론 실행
    outputs = predictor(im)

    # 6. 결과 시각화
    instances = outputs["instances"].to("cpu")
    result_img = im.copy()

    print(f"감지된 객체 수: {len(instances)}")

    if instances.has("pred_distances"):
        boxes = instances.pred_boxes.tensor.numpy()
        distances = instances.pred_distances.numpy()
        classes = instances.pred_classes.numpy()
        scores = instances.scores.numpy()
        
        class_names = MetadataCatalog.get(test_metadata_name).thing_classes
        
        has_drawn_freespace = False # freespace 중복 방지용

        for i in range(len(boxes)):
            score = scores[i]
            class_id = classes[i]
            class_name = class_names[class_id] if class_id < len(class_names) else "unknown"

            # (1) 점수 필터링
            if score < 0.5: continue

            # (2) ★ 숨김 처리 로직 ★
            # 이 리스트에 있는 애들은 아예 그리지 않고 넘어갑니다.
            if class_name in HIDDEN_CLASSES:
                continue

            # (3) Freespace 중복 방지 (가장 점수 높은 1개만)
            if class_name == "freespace":
                if has_drawn_freespace: continue
                has_drawn_freespace = True

            # (4) 데이터 추출
            x1, y1, x2, y2 = boxes[i].astype(int)
            dist_val = distances[i][0]

            # (5) 텍스트 및 색상 결정
            if class_name in DISTANCE_TARGET_CLASSES:
                # 중요 객체: 거리 표시 O, 초록색
                label_text = f"{class_name}: {dist_val:.1f}m"
                color = (0, 255, 0) 
                text_color = (0, 255, 255) # 노란 글씨
            else:
                # 기타 객체 (freespace 등): 거리 표시 X, 파란색
                label_text = f"{class_name}"
                color = (255, 0, 0)
                text_color = (255, 255, 255) # 흰 글씨

            # (6) 그리기
            cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
            
            (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            # 글씨 배경 (검은색)
            cv2.rectangle(result_img, (x1, y1 - text_h - 10), (x1 + text_w + 10, y1), (0, 0, 0), -1)
            # 글씨 쓰기
            cv2.putText(result_img, label_text, (x1 + 5, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
            
            print(f" - 표시됨: {label_text} (정확도: {score*100:.1f}%)")

    # 7. 결과 저장
    cv2.imwrite(OUTPUT_IMAGE_PATH, result_img)
    print(f"결과 이미지가 저장되었습니다: {OUTPUT_IMAGE_PATH}")

if __name__ == "__main__":
    main()