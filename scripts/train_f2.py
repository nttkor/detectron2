import os
import cv2
import random
import numpy as np
import torch
import datetime
import json
import logging

# Detectron2 관련 임포트
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultTrainer, HookBase
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_train_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import Visualizer, ColorMode
import detectron2.utils.comm as comm
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator

# --- 1. 설정 ---
# [기존 데이터 경로]
TRAIN_JSON_PATH = '/home/elicer/dev/gt/data/label_data/train_coco_full.json' 
TRAIN_IMAGE_ROOT = '/home/elicer/dev/detectron2/final_data/train/images'

# [추가할 데이터 경로] (새로 추가됨)
NEW_DATA_JSON = '/home/elicer/dev/gt/data/label_data/more_data_coco.json'
NEW_DATA_IMAGE_ROOT = '/home/elicer/dev/gt/data/new_data/image'

thistime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = '/home/elicer/dev/gt/custom_model/' + thistime

# 클래스 정의 (기존과 동일해야 함)
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

# --- 2. Custom Hook ---
# (기존 코드와 동일)
class ValidationLossHook(HookBase):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.clone()
        self.cfg.DATASETS.TRAIN = cfg.DATASETS.TEST
        self._loader = iter(build_detection_train_loader(self.cfg, num_workers=0))
        
    def after_step(self):
        if (self.trainer.iter + 1) % self.cfg.TEST.EVAL_PERIOD != 0:
            return
        try:
            data = next(self._loader)
        except StopIteration:
            self._loader = iter(build_detection_train_loader(self.cfg, num_workers=0))
            data = next(self._loader)
        with torch.no_grad():
            self.trainer.model.train()
            loss_dict = self.trainer.model(data)
            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict
            loss_dict_reduced = {"val_" + k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            total_loss = sum(loss for loss in loss_dict_reduced.values())
            loss_dict_reduced["val_total_loss"] = total_loss
            if comm.is_main_process():
                self.trainer.storage.put_scalars(**loss_dict_reduced)

# --- 3. Custom Trainer ---
# (기존 코드와 동일)
class MyTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, output_dir=output_folder)
    
    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1, ValidationLossHook(self.cfg))
        return hooks

# --- 4. 데이터 등록 및 분할 ---
def register_split_datasets():
    """
    기존 데이터를 로드하여 Train/Val로 분할
    """
    full_name = "my_dataset_full_source"
    # 기존 데이터 등록
    register_coco_instances(full_name, {}, TRAIN_JSON_PATH, TRAIN_IMAGE_ROOT)
    
    full_dicts = DatasetCatalog.get(full_name)
    
    # 랜덤 셔플
    random.seed(42)
    random.shuffle(full_dicts)
    
    VAL_SET_SIZE = 50 
    if len(full_dicts) < VAL_SET_SIZE:
        VAL_SET_SIZE = int(len(full_dicts) * 0.1)

    val_dicts = full_dicts[:VAL_SET_SIZE]
    train_dicts = full_dicts[VAL_SET_SIZE:]
    
    # 기존 분할 데이터셋 등록
    DatasetCatalog.register("my_dataset_train", lambda: train_dicts)
    MetadataCatalog.get("my_dataset_train").set(thing_classes=ALL_CLASSES, image_root=TRAIN_IMAGE_ROOT, evaluator_type="coco")
    
    DatasetCatalog.register("my_dataset_val", lambda: val_dicts)
    MetadataCatalog.get("my_dataset_val").set(thing_classes=ALL_CLASSES, image_root=TRAIN_IMAGE_ROOT, evaluator_type="coco")
    
    return "my_dataset_train", "my_dataset_val"

# --- 5. Config 설정 (수정됨) ---
def setup_config(train_dataset_names, val_dataset_name): # 인자 변경됨
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    
    # [중요 변경] Train 데이터셋을 튜플로 설정 (기존 + 추가 데이터)
    cfg.DATASETS.TRAIN = train_dataset_names 
    cfg.DATASETS.TEST = (val_dataset_name,)
    
    cfg.DATALOADER.NUM_WORKERS = 8
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    
    cfg.SOLVER.IMS_PER_BATCH = 16
    cfg.SOLVER.BASE_LR = 0.002
    cfg.SOLVER.MAX_ITER = 2000
    cfg.SOLVER.STEPS = (1500, )
    cfg.SOLVER.CHECKPOINT_PERIOD = 1000 
    cfg.TEST.EVAL_PERIOD = 1000
    
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(ALL_CLASSES)
    
    cfg.DATALOADER.SAMPLER_TRAIN = "RepeatFactorTrainingSampler"
    cfg.DATALOADER.REPEAT_THRESHOLD = 0.001 
    
    cfg.OUTPUT_DIR = OUTPUT_DIR
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    return cfg

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    setup_logger(output=OUTPUT_DIR) 
    
    # 1. 기존 데이터셋 등록 및 분할 (Train/Val)
    base_train_name, val_name = register_split_datasets()
    
    # 2. [새로운 데이터셋 등록]
    # 추가 데이터를 "my_new_dataset"이라는 이름으로 등록합니다.
    new_dataset_name = "my_new_dataset"
    register_coco_instances(new_dataset_name, {}, NEW_DATA_JSON, NEW_DATA_IMAGE_ROOT)
    
    # 메타데이터 설정 (클래스 정보 등 일치시켜야 함)
    MetadataCatalog.get(new_dataset_name).set(
        thing_classes=ALL_CLASSES, 
        image_root=NEW_DATA_IMAGE_ROOT, 
        evaluator_type="coco"
    )
    
    print(f"기존 데이터셋: {base_train_name}")
    print(f"추가 데이터셋: {new_dataset_name}")

    # 3. 설정 로드 (두 데이터셋 이름을 튜플로 묶어서 전달)
    # 이렇게 하면 Detectron2가 알아서 두 데이터셋을 합쳐서 학습합니다.
    combined_train_names = (base_train_name, new_dataset_name)
    cfg = setup_config(combined_train_names, val_name)
    
    with open(os.path.join(cfg.OUTPUT_DIR, "config.yaml"), "w") as f:
        f.write(cfg.dump())
        
    print(f"학습 시작! 로그는 {cfg.OUTPUT_DIR}/log.txt 에 저장됩니다.")
    
    trainer = MyTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()
    
    # --- 이하 기존 시각화 코드와 동일 ---
    print("\n[보고서용] 학습 완료 후 시각화 결과 저장 중...")
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05  
    predictor = DefaultPredictor(cfg)
    val_loader = DatasetCatalog.get("my_dataset_val")
    sample_count = min(5, len(val_loader))
    
    for d in random.sample(val_loader, sample_count):    
        im = cv2.imread(d["file_name"])
        if im is None: continue 
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1],
                       metadata=MetadataCatalog.get("my_dataset_val"), 
                       scale=0.8, 
                       instance_mode=ColorMode.IMAGE_BW
        )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        save_path = os.path.join(cfg.OUTPUT_DIR, f"result_{d['image_id']}.jpg")
        cv2.imwrite(save_path, out.get_image()[:, :, ::-1])
        print(f" - 저장됨: {save_path}")

if __name__ == "__main__":
    main()