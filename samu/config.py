# CCTV 소스 설정
# 형식: ("비디오_경로_또는_URL", "카메라_이름")
# 예시:
# SOURCES = [
#     (0, "Webcam"),                            # 웹캠
#     ("videos/gate_1.mp4", "Main Gate"),       # 로컬 파일
#     ("rtsp://admin:1234@192.168.0.10/1", "CCTV 01") # RTSP 스트림
# ]

# 테스트를 위해 0번(웹캠) 하나만 활성화해둡니다. 필요에 따라 수정하세요.
SOURCES = [
    (0, "Camera 1"),
    # ("/home/elicer/dev/dj/video/test_video.mp4", "Camera 2"), # 파일 경로 예시
]

# 모델 설정
MODEL_PATH = "yolo11n-pose.pt"

# 포즈 유사도 임계값 (낮을수록 엄격하게 비교)
SIMILARITY_THRESHOLD = 0.5

# 화면 표시 설정
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480

