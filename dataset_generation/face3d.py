import cv2
import numpy as np
from .calibration import getStereoRectifier
import threading
from .utils import startCameraArray, loadStereoCameraConfig, StereoConfig
from .featuresExtractor import FaceFeatures, FeaturesExtractor
import pyaudio
import wave
import time
from .triangulation import find_depth_from_disparities
import matplotlib.pyplot as plt
import sys


def face3d(face_left, face_right, baseline, f_px):
    assert len(face_left) == len(face_right)

    x = face_left[:, 0]
    y = face_left[:, 1]
    z = [find_depth_from_disparities(
        [x1[0]], [x2[0]], baseline, f_px) for x1, x2 in zip(face_left, face_right)]

    #  for p_left, p_right in zip(face_left, face_right):
    #      depth = find_depth_from_disparities(
    #          [p_left[0]], [p_right[0]], baseline, f_px)
    #      z.append(depth)

    return x, y, z


class LivePlot3d:
    def __init__(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.keep_loop = True

    def plot(self, x, y, z, marker="o"):
        def run():
            self.ax.scatter(x, y, z, marker=marker)
            plt.draw()
            #  time.sleep(1)
        thread = threading.Thread(target=run)
        thread.start()

        plt.pause(0.1)
        thread.join()

    def clean(self):
        self.ax.cla()


class Face3dAnalizer:
    def __init__(self, config_file):
        self.stereo_config = loadStereoCameraConfig(config_file)

        self.f_length = min(self.stereo_config.left_camera.fpx,
                            self.stereo_config.right_camera.fpx)
        self.cams = startCameraArray(self.stereo_config.left_camera,
                                     self.stereo_config.right_camera,
                                     self.stereo_config)
        rectifier = getStereoRectifier(self.stereo_config.stereo_map_file)

        self.cams.rectifier = rectifier

        self.keep_loop = True

        self.features_left = FeaturesExtractor()
        self.features_right = FeaturesExtractor()

    def run(self):
        self.cams.start()
        plot3d = LivePlot3d()
        while self.cams.isOpened() and self.keep_loop:
            frame_left, frame_right = self.cams.get_frames()
            succes_left, frame_left = frame_left
            succes_right, frame_right = frame_right

            features_left = self.features_left.extract_keypts(frame_left)
            features_right = self.features_right.extract_keypts(frame_right)

            if not features_left[0] or not features_right[0]:
                continue
            x, y, z = face3d(
                features_left[2], features_right[2], self.stereo_config.cam_separation, self.f_length)

            #  magnification = 1
            #  frame_left = cv2.resize(frame_left, (np.array(
            #      frame_left.shape[:2][::-1])*magnification).astype(int))
            #  frame_right = cv2.resize(frame_right, (np.array(
            #      frame_right.shape[:2][::-1])*magnification).astype(int))

            #  frame_left_gray = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
            #  frame_right_gray = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)
            #
            plot3d.clean()
            plot3d.plot(x, y, z)

            if not succes_right or not succes_left:
                print("Ignoring empty camera frame.")
                continue

            terminate = self.showImages(frame_left, frame_right, frame_left)
            if terminate:
                break
        self.cams.close()
        self.keep_loop = False

    def start(self):
        threading.Thread(target=self.run).start()

    def close(self):
        self.keep_loop = False

    def showImages(self, frame_left, frame_right, disparity, magnification=1):
        if self.stereo_config.show_images:
            cv2.imshow("frame right", cv2.resize(
                frame_right, (np.array(frame_right.shape[:2][::-1])*magnification).astype(int)))
            cv2.imshow("frame left", cv2.resize(
                frame_left, (np.array(frame_right.shape[:2][::-1])*magnification).astype(int)))
            cv2.imshow("depth_map", cv2.resize(
                disparity, (np.array(disparity.shape[:2][::-1])*magnification).astype(int)))
            if cv2.waitKey(5) & 0xFF == 27:
                return True
            return False


if __name__ == "__main__":
    stereo_config_file = sys.argv[1]
    face_analier = Face3dAnalizer(stereo_config_file)
    face_analier.run()
