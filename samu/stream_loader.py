import cv2
from threading import Thread
import time

class InputStream:
    def __init__(self, src=0, name="Camera"):
        self.src = src
        self.name = name
        self.capture = cv2.VideoCapture(src)
        
        # 원활한 스트리밍을 위한 버퍼 설정 (선택 사항)
        # self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        
        self.grabbed, self.frame = self.capture.read()
        self.stopped = False
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def start(self):
        """별도 스레드에서 프레임 읽기 시작"""
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        """계속해서 프레임을 읽어 최신 프레임 유지"""
        while True:
            if self.stopped:
                return
            
            grabbed, frame = self.capture.read()
            if not grabbed:
                # 스트림이 끊기거나 파일이 끝났을 때 재시도 로직 등을 추가할 수 있음
                self.stop()
                return
            
            self.grabbed = grabbed
            self.frame = frame
            time.sleep(0.001) # CPU 과부하 방지

    def read(self):
        """최신 프레임 반환"""
        return self.frame

    def stop(self):
        self.stopped = True
        self.capture.release()

class MultiStreamLoader:
    def __init__(self, sources):
        """
        sources: [ (source_path, "Camera Name"), ... ]
        """
        self.streams = []
        for src, name in sources:
            stream = InputStream(src, name)
            self.streams.append(stream)

    def start(self):
        for stream in self.streams:
            stream.start()
        return self

    def read_all(self):
        """모든 카메라의 현재 프레임을 리스트로 반환"""
        frames = []
        for stream in self.streams:
            frames.append((stream.name, stream.read()))
        return frames

    def stop(self):
        for stream in self.streams:
            stream.stop()