import cv2
import time
import numpy as np
from config import SOURCES, MODEL_PATH, SIMILARITY_THRESHOLD, DISPLAY_WIDTH, DISPLAY_HEIGHT
from stream_loader import MultiStreamLoader
from tracker import PoseTracker

def main():
    # 1. 멀티 스트림 로더 초기화
    print(f"[System] Loading {len(SOURCES)} streams...")
    loader = MultiStreamLoader(SOURCES)
    loader.start()
    
    # 2. 카메라별 트래커 초기화 (각 카메라마다 독립적인 ID 관리)
    # 모델은 하나만 로드해서 공유할 수도 있지만, 여기서는 간단하게 각각 생성합니다.
    # (메모리가 부족하면 모델 인스턴스 하나를 공유하도록 수정 필요)
    print(f"[System] Initializing trackers for {len(SOURCES)} cameras...")
    trackers = [PoseTracker(model_path=MODEL_PATH, similarity_threshold=SIMILARITY_THRESHOLD) for _ in SOURCES]
    
    time.sleep(2.0) # 카메라 워밍업 대기
    print("[System] Started. Press 'q' to exit.")

    try:
        while True:
            start_time = time.time()
            
            # 모든 카메라에서 프레임 읽기
            frames_info = loader.read_all() # list of (name, frame)
            
            display_frames = []
            
            for i, (name, frame) in enumerate(frames_info):
                if frame is None:
                    # 프레임이 없는 경우(연결 끊김 등) 빈 화면 표시
                    blank_image = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), np.uint8)
                    cv2.putText(blank_image, f"{name}: No Signal", (10, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    display_frames.append(blank_image)
                    continue
                
                # 트래킹 처리
                # frame은 레퍼런스로 전달되므로, 원본을 수정하지 않으려면 .copy() 사용 고려
                processed_frame = trackers[i].process_frame(frame)
                
                # 화면에 카메라 이름 표시
                cv2.putText(processed_frame, name, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                
                # 화면 크기 통일 (타일링을 위해)
                resized_frame = cv2.resize(processed_frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
                display_frames.append(resized_frame)
            
            if not display_frames:
                continue

            # 화면 타일링 (단순 가로 연결)
            # 카메라가 많아지면 2x2 그리드 등으로 로직 변경 필요
            if len(display_frames) > 0:
                # numpy hconcat을 사용하여 가로로 붙임
                combined_display = cv2.hconcat(display_frames)
                
                # 결과 출력
                cv2.imshow("Multi-CCTV Tracking System", combined_display)
            
            # FPS 계산 및 지연 시간 조절 (필요 시)
            # process_time = time.time() - start_time
            # print(f"FPS: {1/process_time:.2f}")
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[System] Quitting...")
                break
                
    except KeyboardInterrupt:
        print("\n[System] Stopped by user")
    except Exception as e:
        print(f"\n[System] Error occurred: {e}")
    finally:
        # 자원 해제
        loader.stop()
        cv2.destroyAllWindows()
        print("[System] Resources released.")

if __name__ == "__main__":
    main()

