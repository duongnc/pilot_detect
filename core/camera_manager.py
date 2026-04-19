import cv2
import threading

class CameraManager:
    def __init__(self, camera_source=0, width=640, height=480):
        self.cap = cv2.VideoCapture(camera_source)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.ret, self.frame = self.cap.read()
        self.running = False
        self.thread = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True # Thread sẽ tự động tắt khi chương trình chính tắt
        self.thread.start()

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame

    def get_latest_frame(self):
        return self.frame if self.ret else None

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join()
        self.cap.release()