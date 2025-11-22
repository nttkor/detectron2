# -------------------------------------------------
# OneFormer Panoptic Segmentation Visualization
# Thing(객체): Polygon, Stuff(배경): Fill
# -------------------------------------------------

import os
import glob
import torch
import cv2
import time
import numpy as np
from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation

def visualize_cv2_all(img_bgr, seg_map, segments_info, id2label, is_thing_map, filename, inference_time):
    """Panoptic Segmentation 결과 시각화 - 리사이즈 후 모든 그리기 수행"""
    # ① 먼저 리사이즈 (높이 800px 고정)
    h, w = img_bgr.shape[:2]
    target_h = 800
    target_w = int(w * target_h / h)
    resized_orig = cv2.resize(img_bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    
    # ② Stuff용 오버레이 초기화 (리사이즈된 크기)
    overlay_stuff = np.zeros_like(resized_orig, dtype=np.uint8)
    
    # 색상 생성 함수 (HSV 기반으로 구별되는 색상 생성)
    def get_color(idx):
        hue = int((idx * 137.5) % 180)
        hsv = np.uint8([[[hue, 255, 255]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        return tuple(map(int, bgr))

    inst_info = {s['id']: s for s in segments_info}
    unique_ids = np.unique(seg_map)
    centroids = {}
    
    # ③ [Stuff 그리기]
    for i, cid in enumerate(unique_ids):
        if cid not in inst_info:
            continue
        info = inst_info[cid]
        label_id = info['label_id']
        is_thing = is_thing_map.get(label_id, False) if isinstance(is_thing_map, dict) else is_thing_map[label_id]
        
        if is_thing:
            continue

        mask = seg_map == cid
        if not np.any(mask):
            continue
        
        mask_resized = cv2.resize(mask.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        
        b, g, r = get_color(i)
        overlay_stuff[mask_resized > 0] = (b, g, r)
        
        y, x = np.where(mask_resized > 0)
        if len(y) > 0 and len(x) > 0:
            centroids[int(cid)] = (int(x.mean()), int(y.mean()))

    # ④ 알파 블렌딩 (Stuff만 적용)
    alpha = 120 / 255.0
    blended = cv2.addWeighted(overlay_stuff, alpha, resized_orig, 1 - alpha, 0)

    # ⑤ [Thing 그리기]
    for i, cid in enumerate(unique_ids):
        if cid not in inst_info:
            continue
            
        info = inst_info[cid]
        label_id = info['label_id']
        is_thing = is_thing_map.get(label_id, False) if isinstance(is_thing_map, dict) else is_thing_map[label_id]
        
        if not is_thing:
            continue

        mask = seg_map == cid
        if not np.any(mask):
            continue
        
        mask_resized = cv2.resize(mask.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        mask_resized = (mask_resized * 255).astype(np.uint8)
        
        b, g, r = get_color(i)
        contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(blended, contours, -1, (b, g, r), 2)
        
        y, x = np.where(mask_resized > 0)
        if len(y) > 0 and len(x) > 0:
            centroids[int(cid)] = (int(x.mean()), int(y.mean()))

    # ⑥ 라벨 텍스트 표시
    font_scale = cv2.getFontScaleFromHeight(cv2.FONT_HERSHEY_SIMPLEX, 12, 1)
    
    for cid, (cx, cy) in centroids.items():
        label_id = inst_info[cid]['label_id']
        class_name = id2label.get(label_id, str(label_id))
        label_text = f"{label_id}: {class_name}"
        
        is_thing = is_thing_map.get(label_id, False) if isinstance(is_thing_map, dict) else is_thing_map[label_id]
        text_color = (0, 255, 255) if is_thing else (255, 255, 255)
        
        cv2.putText(blended, label_text, (cx - 10, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1, cv2.LINE_AA)

    # ⑦ 상단 정보
    thing_count = sum(1 for s in segments_info if is_thing_map.get(s['label_id'], False))
    stuff_count = len(segments_info) - thing_count
    
    info_text = f"{filename} | T:{thing_count} S:{stuff_count}"
    cv2.putText(blended, info_text, (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), 1, cv2.LINE_AA)
    
    time_text = f"Inference: {inference_time:.4f}s"
    (tw, th), _ = cv2.getTextSize(time_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    cv2.putText(blended, time_text, (target_w - tw - 10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), 1, cv2.LINE_AA)

    # ⑧ 화면 표시
    window_name = "OneFormer – Panoptic Segmentation"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, target_w, target_h)
    cv2.moveWindow(window_name, 0, 0)
    cv2.imshow(window_name, blended)
    return blended

# 이미지 리스트
IMAGE_DIR = r"D:/git/detectron2/ade20k_consistency/original_ade20k"
image_files = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.jpg")))
if not image_files:
    raise FileNotFoundError(f"'{IMAGE_DIR}'에 이미지가 없습니다.")
print(f"📂 {len(image_files)}개 이미지")

# 모델
model_name = "shi-labs/oneformer_ade20k_swin_tiny"
print(f"🔧 모델 로드: {model_name}")
processor = OneFormerProcessor.from_pretrained(model_name)
model = OneFormerForUniversalSegmentation.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

id2label = model.config.id2label

# Thing/Stuff 매핑
if hasattr(model.config, 'class_is_thing'):
    is_thing_map = model.config.class_is_thing
    print(f"✓ class_is_thing 사용")
else:
    print("⚠ 기본 규칙 사용 (0-91: Thing, 92+: Stuff)")
    is_thing_map = {i: (i < 92) for i in range(150)}

def run_inference(idx):
    """추론 수행"""
    img_path = image_files[idx]
    filename = os.path.basename(img_path)
    
    # OpenCV로 이미지 로드 (BGR)
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"❌ 이미지 로드 실패: {img_path}")
        return
   
    # BGR -> RGB로 변환 (processor가 RGB를 기대)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_bgr.shape[:2]
    
    print(f"\n📂 [{idx+1}/{len(image_files)}] {filename}")
    
    # 추론
    inputs = processor(images=img_rgb, task_inputs=["panoptic"], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    start_time = time.time()
    with torch.no_grad():
        outputs = model(**inputs)
    inference_time = time.time() - start_time
        
    # 후처리 (원본 이미지 크기 전달)
    panoptic_result = processor.post_process_panoptic_segmentation(
        outputs, target_sizes=[(h, w)])[0]
    
    seg_map = panoptic_result["segmentation"].cpu().numpy()
    segments_info = panoptic_result["segments_info"]
    
    # 디버그
    thing_count = sum(1 for s in segments_info if is_thing_map.get(s['label_id'], False))
    print(f"DEBUG - Thing: {thing_count}, Stuff: {len(segments_info) - thing_count}")

    print(f"✓ 추론 완료 ({inference_time:.4f}초)")
    visualize_cv2_all(img_bgr, seg_map, segments_info, id2label, is_thing_map, filename, inference_time)

def main():
    """메인 루프"""
    cur_idx = 0
    run_inference(cur_idx)

    print("\n키: A/← (이전), D/→ (다음), Q (종료)")

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('a') or key == 0x250000:
            cur_idx = (cur_idx - 1) % len(image_files)
            run_inference(cur_idx)
        elif key == ord('d') or key == 0x270000:
            cur_idx = (cur_idx + 1) % len(image_files)
            run_inference(cur_idx)
        elif key == ord('q'):
            print("\n👋 종료")
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()