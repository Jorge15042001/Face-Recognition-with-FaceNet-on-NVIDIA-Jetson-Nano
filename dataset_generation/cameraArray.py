
# import the opencv library
import cv2
from time import sleep

from datetime import datetime
import threading


def get_now_str():
    return datetime.utcnow().strftime('%Y-%m-%d__%H:%M:%S.%f')


class cameraThread:

    def __init__(self, camidx, fps=30, resolution=(640, 480)):
        self.cam_idx = camidx
        #  self.frames = Queue(fps)
        self.frame = None
        self.closed = False
        self.cap = cv2.VideoCapture(camidx)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        self.cap.set(cv2.CAP_PROP_FOURCC,
                     cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        def capture_function():
            for _ in range(30):
                self.cap.grab()
            while self.cap.isOpened() and not self.closed:
                sleep(1/fps)
                frame = self.cap.read()
                #  self.frames.put(frame)
                self.frame = frame
            self.cap.release()
        self.cap_thread = threading.Thread(
            target=capture_function, daemon=True)

    def close(self):
        self.closed = True
        self.cap_thread.join()

    def get_frame(self):
        return self.frame


class CamArray:
    def __init__(self, camidxs, fps=60, save_frames_to=None, rectifier=lambda x, y: (x, y), resolution=(640, 480)):
        assert (len(camidxs) == 2)

        self.cams = list(
            map(lambda x: cameraThread(x, fps, resolution), camidxs))
        self.fps = fps
        self.save_frames_to = save_frames_to
        self.video_writers = None
        self.video_writers_rectified = None
        if save_frames_to is not None:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            now_str = get_now_str()
            self.video_writers = list(map(lambda cam_id: cv2.VideoWriter(
                f"{save_frames_to}/{now_str}_{cam_id}.avi", fourcc, 30.0, resolution), camidxs))
            self.video_writers_rectified = list(map(lambda cam_id: cv2.VideoWriter(
                f"{save_frames_to}/{now_str}_{cam_id}_rectified.avi", fourcc, 30.0, resolution), camidxs))
        self.rectifier = rectifier

    def start(self):
        for cam in self.cams:
            cam.cap_thread.start()
        while any(map(lambda cam: type(cam.get_frame()) == type(None), self.cams)):
            sleep(1/self.fps)

    def close(self):
        for cam in self.cams:
            cam.close()

        if self.video_writers is not None and\
            self.video_writers_rectified is not None:
            for writer in self.video_writers + self.video_writers_rectified:
                writer.release()

    def get_frames(self):
        while any(map(lambda cam: not cam.get_frame()[0], self.cams)): continue
        frames = list(map(lambda cam: cam.get_frame(), self.cams))
        # record the frames into a video
        if self.video_writers is not None:
            print("writting frames")
            for (_, frame), writer in zip(frames, self.video_writers):
                writer.write(frame)

        frame1, frame2 = self.rectifier(frames[0][1], frames[1][1])
        frames = ((True, frame1), (True, frame2))

        if self.video_writers_rectified is not None:
            print("writting rectified frames")
            for (_, frame), writer in zip(frames, self.video_writers_rectified):
                writer.write(frame)
        return ((True, frame1), (True, frame2))

    def isOpened(self):
        return all(map(lambda cam: cam.cap.isOpened(), self.cams))
