"""
OneFormer Panoptic Segmentation - Interactive Visualization Tool
ADE20K 공식 Thing/Stuff 분류 (CSAILVision MIT) + 신뢰도 표시

프로그램 개요:
    이 프로그램은 OneFormer 딥러닝 모델을 사용하여 이미지의 Panoptic Segmentation을 수행하고
    인터랙티브하게 결과를 시각화하는 도구입니다.
    
    주요 기능:
    1. 이미지 디렉토리에서 JPG 파일을 자동으로 로드
    2. OneFormer 모델을 사용한 Panoptic Segmentation 추론 수행
    3. Thing(객체)과 Stuff(배경)를 구분하여 시각화
    4. 각 세그먼트의 클래스 이름과 신뢰도 점수 표시
    5. 키보드 입력(A/D 또는 화살표 키)으로 이미지 간 이동
    6. 추론 시간 및 Thing/Stuff 개수 정보 표시
    
    사용 모델:
    - 모델: shi-labs/oneformer_ade20k_swin_tiny
    - 데이터셋: ADE20K (150개 클래스)
    - 분류: Thing(객체) / Stuff(배경) 공식 분류 사용
    
    실행 방법:
    - 스크립트 실행 시 첫 번째 이미지가 자동으로 로드됨
    - 'A' 또는 왼쪽 화살표: 이전 이미지
    - 'D' 또는 오른쪽 화살표: 다음 이미지
    - 'Q': 프로그램 종료
    
    시각화 특징:
    - Stuff 영역: 반투명 색상 오버레이
    - Thing 영역: 외곽선으로 표시
    - 텍스트 색상: Thing(노란색), Stuff(흰색)
    - 각 세그먼트 중심에 클래스 이름과 신뢰도 표시
"""

import os  # 파일 시스템 경로 조작
import glob  # 파일 패턴 매칭
import torch  # PyTorch 딥러닝 프레임워크
import cv2  # OpenCV 이미지 처리
import time  # 시간 측정
import numpy as np  # 수치 연산
from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation  # OneFormer 모델 및 프로세서

def visualize_cv2_all(img_bgr, seg_map, segments_info, id2label, is_thing_map, filename, inference_time):
    """
    Panoptic Segmentation 결과를 OpenCV를 사용하여 시각화합니다.
    
    이 함수는 세그멘테이션 결과를 받아서 Thing과 Stuff를 구분하여 시각화하고,
    각 세그먼트의 클래스 이름, 신뢰도 점수, 통계 정보를 이미지에 오버레이합니다.
    
    Args:
        img_bgr (numpy.ndarray): 원본 이미지 (BGR 형식)
        seg_map (numpy.ndarray): 세그멘테이션 맵 (각 픽셀의 세그먼트 ID)
        segments_info (list): 각 세그먼트의 정보 딕셔너리 리스트 (id, label_id, score 등)
        id2label (dict): 라벨 ID를 클래스 이름으로 매핑하는 딕셔너리
        is_thing_map (dict): 라벨 ID를 Thing 여부(bool)로 매핑하는 딕셔너리
        filename (str): 현재 처리 중인 이미지 파일명
        inference_time (float): 추론에 소요된 시간(초)
    
    Returns:
        numpy.ndarray: 시각화된 이미지 (BGR 형식)
    
    처리 과정:
        1. 이미지를 800px 높이로 리사이즈 (비율 유지)
        2. Stuff 영역을 반투명 색상 오버레이로 표시
        3. Thing 영역을 외곽선으로 표시
        4. 각 세그먼트 중심에 클래스 이름과 신뢰도 점수 텍스트 추가
        5. 상단에 파일명, Thing/Stuff 개수, 추론 시간 정보 표시
        6. OpenCV 윈도우에 결과 이미지 표시
    """
    h, w = img_bgr.shape[:2]  # 이미지 높이와 너비 추출
    target_h = 800  # 목표 높이 설정
    target_w = int(w * target_h / h)  # 비율 유지하며 목표 너비 계산
    resized_orig = cv2.resize(img_bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)  # 원본 이미지 리사이즈
    
    overlay_stuff = np.zeros_like(resized_orig, dtype=np.uint8)  # Stuff 오버레이 초기화
    
    def get_color(idx):
        """
        인덱스 기반으로 고유한 색상을 생성합니다.
        
        HSV 색상 공간을 사용하여 각 인덱스마다 서로 다른 색상을 생성하며,
        137.5도 간격으로 색상을 배치하여 시각적으로 구분하기 쉽게 만듭니다.
        
        Args:
            idx (int): 색상을 생성할 인덱스
        
        Returns:
            tuple: (B, G, R) 형식의 정수 튜플 (OpenCV BGR 형식)
        """
        hue = int((idx * 137.5) % 180)  # HSV 색상 공간에서 색상값 계산
        hsv = np.uint8([[[hue, 255, 255]]])  # HSV 배열 생성
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]  # HSV를 BGR로 변환
        return tuple(map(int, bgr))  # 정수 튜플로 반환

    inst_info = {s['id']: s for s in segments_info}  # 세그먼트 정보를 딕셔너리로 변환
    unique_ids = np.unique(seg_map)  # 고유 세그먼트 ID 추출
    centroids = {}  # 중심점 저장 딕셔너리 초기화
    
    # Stuff 그리기
    for i, cid in enumerate(unique_ids):  # 각 고유 ID 순회
        if cid not in inst_info:  # 정보가 없으면 건너뛰기
            continue
        info = inst_info[cid]  # 세그먼트 정보 가져오기
        label_id = info['label_id']  # 라벨 ID 추출
        is_thing = is_thing_map.get(label_id, False)  # Thing 여부 확인
        
        if is_thing:  # Thing이면 건너뛰기
            continue

        mask = seg_map == cid  # 현재 ID에 해당하는 마스크 생성
        if not np.any(mask):  # 마스크가 비어있으면 건너뛰기
            continue
        
        mask_resized = cv2.resize(mask.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST)  # 마스크 리사이즈
        
        b, g, r = get_color(i)  # 색상 가져오기
        overlay_stuff[mask_resized > 0] = (b, g, r)  # Stuff 영역에 색상 적용
        
        y, x = np.where(mask_resized > 0)  # 마스크 영역의 좌표 추출
        if len(y) > 0 and len(x) > 0:  # 좌표가 있으면
            centroids[int(cid)] = (int(x.mean()), int(y.mean()))  # 중심점 계산 및 저장

    alpha = 120 / 255.0  # 투명도 설정
    blended = cv2.addWeighted(overlay_stuff, alpha, resized_orig, 1 - alpha, 0)  # Stuff 오버레이와 원본 이미지 블렌딩

    # Thing 그리기
    for i, cid in enumerate(unique_ids):  # 각 고유 ID 순회
        if cid not in inst_info:  # 정보가 없으면 건너뛰기
            continue
            
        info = inst_info[cid]  # 세그먼트 정보 가져오기
        label_id = info['label_id']  # 라벨 ID 추출
        is_thing = is_thing_map.get(label_id, False)  # Thing 여부 확인
        
        if not is_thing:  # Thing이 아니면 건너뛰기
            continue

        mask = seg_map == cid  # 현재 ID에 해당하는 마스크 생성
        if not np.any(mask):  # 마스크가 비어있으면 건너뛰기
            continue
        
        mask_resized = cv2.resize(mask.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST)  # 마스크 리사이즈
        mask_resized = (mask_resized * 255).astype(np.uint8)  # 마스크를 0-255 범위로 변환
        
        b, g, r = get_color(i)  # 색상 가져오기
        contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 외곽선 찾기
        cv2.drawContours(blended, contours, -1, (b, g, r), 2)  # Thing 외곽선 그리기
        
        y, x = np.where(mask_resized > 0)  # 마스크 영역의 좌표 추출
        if len(y) > 0 and len(x) > 0:  # 좌표가 있으면
            centroids[int(cid)] = (int(x.mean()), int(y.mean()))  # 중심점 계산 및 저장

    # 라벨 + 신뢰도 텍스트
    font_scale = cv2.getFontScaleFromHeight(cv2.FONT_HERSHEY_SIMPLEX, 12, 1)  # 폰트 크기 계산
    
    for cid, (cx, cy) in centroids.items():  # 각 중심점에 대해
        label_id = inst_info[cid]['label_id']  # 라벨 ID 가져오기
        class_name = id2label.get(label_id, str(label_id)).split(';')[0]  # 클래스 이름 가져오기 (첫 번째만)
        score = inst_info[cid].get('score', 0.0)  # 신뢰도 점수 가져오기
        
        # 첫 번째 줄: 클래스 이름
        label_text = f"{class_name}"  # 라벨 텍스트 생성
        
        is_thing = is_thing_map.get(label_id, False)  # Thing 여부 확인
        text_color = (0, 255, 255) if is_thing else (255, 255, 255)  # Thing은 노란색, Stuff는 흰색
        
        # 클래스 이름 그리기
        cv2.putText(blended, label_text, (cx - 10, cy),  # 텍스트 그리기
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1, cv2.LINE_AA)
        
        # 두 번째 줄: 신뢰도
        score_text = f"[{label_id}]{score:.2f}"  # 신뢰도 텍스트 생성
        cv2.putText(blended, score_text, (cx - 10, cy + 15),  # 신뢰도 텍스트 그리기
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, text_color, 1, cv2.LINE_AA)

    # 상단 정보
    thing_count = sum(1 for s in segments_info if is_thing_map.get(s['label_id'], False))  # Thing 개수 계산
    stuff_count = len(segments_info) - thing_count  # Stuff 개수 계산
    
    info_text = f"{filename} | T:{thing_count} S:{stuff_count}"  # 정보 텍스트 생성
    cv2.putText(blended, info_text, (10, 20),  # 상단 왼쪽에 정보 표시
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), 1, cv2.LINE_AA)
    
    time_text = f"Inference: {inference_time:.4f}s"  # 추론 시간 텍스트 생성
    (tw, th), _ = cv2.getTextSize(time_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)  # 텍스트 크기 계산
    cv2.putText(blended, time_text, (target_w - tw - 10, 20),  # 상단 오른쪽에 시간 표시
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), 1, cv2.LINE_AA)

    window_name = "OneFormer - Panoptic Segmentation"  # 윈도우 이름 설정
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # 윈도우 생성
    cv2.resizeWindow(window_name, target_w, target_h)  # 윈도우 크기 조정
    cv2.moveWindow(window_name, 0, 0)  # 윈도우 위치 이동
    cv2.imshow(window_name, blended)  # 이미지 표시
    return blended  # 블렌딩된 이미지 반환

# 이미지 리스트
IMAGE_DIR = r"D:/git/detectron2/ade20k_consistency/original_ade20k"  # 이미지 디렉토리 경로
image_files = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.jpg")))  # JPG 파일 목록 정렬
if not image_files:  # 이미지가 없으면
    raise FileNotFoundError(f"'{IMAGE_DIR}'에 이미지가 없습니다.")  # 에러 발생
print(f"📂 {len(image_files)}개 이미지")  # 이미지 개수 출력

# 모델 로드
# model_name = "shi-labs/oneformer_ade20k_swin_tiny"  # 모델 이름 설정
model_name = "shi-labs/oneformer_ade20k_swin_large"  # 모델 이름 설정
print(f"🔧 모델 로드: {model_name}")  # 모델 로드 메시지 출력
processor = OneFormerProcessor.from_pretrained(model_name)  # 프로세서 로드
model = OneFormerForUniversalSegmentation.from_pretrained(model_name)  # 모델 로드

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # GPU 사용 가능 여부에 따라 디바이스 선택
model.to(device)  # 모델을 디바이스로 이동

# ADE20K 공식 Thing/Stuff 분류 및 클래스 이름 사용
from ade20k_thing_stuff_map import ADE20K_THING_STUFF_CLASSES, ADE20K_CLASS_NAMES  # ADE20K 매핑 데이터 import

is_thing_map = ADE20K_THING_STUFF_CLASSES  # Thing/Stuff 분류 맵 설정
id2label = ADE20K_CLASS_NAMES  # 공식 클래스 이름 사용

thing_count = sum(1 for v in is_thing_map.values() if v)  # Thing 클래스 개수 계산
stuff_count = sum(1 for v in is_thing_map.values() if not v)  # Stuff 클래스 개수 계산
print(f"✓ ADE20K 공식 Thing/Stuff 분류 사용 (CSAILVision MIT)")  # 분류 사용 메시지 출력
print(f"  - Thing: {thing_count}개 클래스")  # Thing 개수 출력
print(f"  - Stuff: {stuff_count}개 클래스")  # Stuff 개수 출력

def run_inference(idx):
    """
    지정된 인덱스의 이미지에 대해 Panoptic Segmentation 추론을 수행합니다.
    
    이 함수는 이미지 파일을 로드하고, OneFormer 모델을 사용하여 세그멘테이션을 수행한 후,
    결과를 시각화 함수로 전달합니다.
    
    Args:
        idx (int): image_files 리스트에서 처리할 이미지의 인덱스
    
    처리 과정:
        1. 이미지 파일 경로 가져오기 및 로드
        2. BGR 형식을 RGB로 변환 (모델 입력 형식)
        3. OneFormer 프로세서로 이미지 전처리
        4. 모델 추론 수행 (GPU/CPU 자동 선택)
        5. 추론 시간 측정
        6. 후처리로 Panoptic Segmentation 결과 생성
        7. Thing/Stuff 개수 계산 및 디버그 정보 출력
        8. 시각화 함수 호출
    
    출력:
        - 현재 이미지 정보 (인덱스/총 개수, 파일명)
        - Thing/Stuff 개수 디버그 정보
        - 추론 완료 메시지 및 소요 시간
        - 시각화된 결과 이미지 (OpenCV 윈도우)
    
    예외:
        이미지 로드 실패 시 에러 메시지 출력 후 함수 종료
    """
    img_path = image_files[idx]  # 이미지 경로 가져오기
    filename = os.path.basename(img_path)  # 파일명 추출
    
    img_bgr = cv2.imread(img_path)  # 이미지 읽기 (BGR 형식)
    if img_bgr is None:  # 이미지 로드 실패 시
        print(f"❌ 이미지 로드 실패: {img_path}")  # 에러 메시지 출력
        return
   
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # BGR을 RGB로 변환
    h, w = img_bgr.shape[:2]  # 이미지 크기 추출
    
    print(f"\n📂 [{idx+1}/{len(image_files)}] {filename}")  # 현재 이미지 정보 출력
    
    inputs = processor(images=img_rgb, task_inputs=["panoptic"], return_tensors="pt")  # 이미지 전처리
    inputs = {k: v.to(device) for k, v in inputs.items()}  # 입력을 디바이스로 이동
    
    start_time = time.time()  # 시작 시간 기록
    with torch.no_grad():  # 그래디언트 계산 비활성화
        outputs = model(**inputs)  # 모델 추론 수행
    inference_time = time.time() - start_time  # 추론 시간 계산
        
    panoptic_result = processor.post_process_panoptic_segmentation(  # Panoptic 세그멘테이션 후처리
        outputs, target_sizes=[(h, w)])[0]
    
    seg_map = panoptic_result["segmentation"].cpu().numpy()  # 세그멘테이션 맵 추출
    segments_info = panoptic_result["segments_info"]  # 세그먼트 정보 추출
    
    thing_count = sum(1 for s in segments_info if is_thing_map.get(s['label_id'], False))  # Thing 개수 계산
    print(f"DEBUG - Thing: {thing_count}, Stuff: {len(segments_info) - thing_count}")  # 디버그 정보 출력

    print(f"✓ 추론 완료 ({inference_time:.4f}초)")  # 완료 메시지 출력
    visualize_cv2_all(img_bgr, seg_map, segments_info, id2label, is_thing_map, filename, inference_time)  # 결과 시각화

def main():
    """
    프로그램의 메인 실행 루프입니다.
    
    이 함수는 프로그램의 진입점으로, 첫 번째 이미지를 자동으로 로드하고
    키보드 입력을 받아 이미지 간 이동을 처리하는 인터랙티브 루프를 실행합니다.
    
    실행 흐름:
        1. 첫 번째 이미지(인덱스 0) 자동 추론 및 시각화
        2. 사용법 안내 메시지 출력
        3. 무한 루프로 키보드 입력 대기
        4. 입력에 따라 이전/다음 이미지로 이동하거나 프로그램 종료
    
    키보드 입력:
        - 'A' 또는 왼쪽 화살표 (0x250000): 이전 이미지로 이동
        - 'D' 또는 오른쪽 화살표 (0x270000): 다음 이미지로 이동
        - 'Q': 프로그램 종료
    
    특징:
        - 이미지 인덱스는 순환 구조 (마지막 이미지에서 다음 = 첫 이미지)
        - 각 이미지 이동 시 자동으로 추론 및 시각화 수행
        - 종료 시 모든 OpenCV 윈도우 자동 닫기
    """
    cur_idx = 0  # 현재 이미지 인덱스 초기화
    run_inference(cur_idx)  # 첫 번째 이미지 추론

    print("\n키: A/← (이전), D/→ (다음), Q (종료)")  # 사용법 안내 출력

    while True:  # 무한 루프
        key = cv2.waitKey(0) & 0xFF  # 키 입력 대기
        if key == ord('a') or key == 0x250000:  # 'a' 또는 왼쪽 화살표 키
            cur_idx = (cur_idx - 1) % len(image_files)  # 이전 이미지로 이동
            run_inference(cur_idx)  # 추론 수행
        elif key == ord('d') or key == 0x270000:  # 'd' 또는 오른쪽 화살표 키
            cur_idx = (cur_idx + 1) % len(image_files)  # 다음 이미지로 이동
            run_inference(cur_idx)  # 추론 수행
        elif key == ord('q'):  # 'q' 키
            print("\n👋 종료")  # 종료 메시지 출력
            break  # 루프 종료
    cv2.destroyAllWindows()  # 모든 윈도우 닫기

if __name__ == "__main__":  # 스크립트가 직접 실행될 때
    main()  # 메인 함수 실행
