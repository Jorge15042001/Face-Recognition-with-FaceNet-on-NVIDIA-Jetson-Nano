import cv2
import numpy as np
from .calibration import getStereoRectifier
import threading
from .utils import startCameraArray, loadStereoCameraConfig, StereoConfig
import pyaudio
import wave
import time
import matplotlib.pyplot as plt
import sys


def depth_map(imgL, imgR):

    # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
    window_size = 5

    left_matcher = cv2.StereoSGBM_create(
        minDisparity=-1,
        numDisparities=16*11,  # max_disp has to be dividable by 16 f. E. HH 192, 256
        blockSize=window_size,
        P1=9 * 3 * window_size,
        # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
        P2=128 * 3 * window_size,
        disp12MaxDiff=12,
        uniquenessRatio=40,
        speckleWindowSize=50,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    # FILTER Parameters
    lmbda = 70000
    sigma = 1.7
    visual_multiplier = 6

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(
        matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)

    displ = left_matcher.compute(imgL, imgR)  # .astype(np.float32)/16
    dispr = right_matcher.compute(imgR, imgL)  # .astype(np.float32)/16
    displ = np.int16(displ)
    dispr = np.int16(dispr)
    # important to put "imgL" here!!!
    filteredImg = wls_filter.filter(displ, imgL, None, dispr)
    #  print(filteredImg.dtype)
    filteredImg = cv2.normalize(
        src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
    filteredImg = np.uint8(filteredImg)

    return filteredImg


class Face3dAnalizer:
    def __init__(self, config_file):
        self.stereo_config = loadStereoCameraConfig(config_file)
        #  cam_center_left, cam_center_right = cam_centers
        self.f_length = min(self.stereo_config.left_camera.fpx,
                            self.stereo_config.right_camera.fpx)
        self.cams = startCameraArray(self.stereo_config.left_camera,
                                     self.stereo_config.right_camera,
                                     self.stereo_config)
        rectifier = getStereoRectifier(self.stereo_config.stereo_map_file)
        self.cams.rectifier = rectifier

        self.keep_loop = True

    def run(self):
        self.cams.start()
        while self.cams.isOpened() and self.keep_loop:
            frame_left, frame_right = self.cams.get_frames()
            succes_left, frame_left = frame_left
            succes_right, frame_right = frame_right

            magnification = 1
            frame_left = cv2.resize(frame_left, (np.array(
                frame_left.shape[:2][::-1])*magnification).astype(int))
            frame_right = cv2.resize(frame_right, (np.array(
                frame_right.shape[:2][::-1])*magnification).astype(int))

            frame_left_gray = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
            frame_right_gray = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)

            if not succes_right or not succes_left:
                print("Ignoring empty camera frame.")
                continue
            disparity = depth_map(frame_right_gray, frame_left_gray)
            #  print(disparity.max(), disparity.min())
            #  disparity = disparity.astype(float)
            #  disparity = (disparity - disparity.min())/(disparity.max() - disparity.min())
            #  disparity = (disparity*255).astype(np.uint8)

            terminate = self.showImages(frame_left_gray, frame_right_gray, disparity)
            if terminate:
                break
        self.cams.close()
        self.keep_loop = False

    def start(self):
        threading.Thread(target=self.run).start()

    def close(self):
        self.keep_loop = False

    def showImages(self, frame_left, frame_right, disparity, magnification = 1):
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
    video_recorder = Face3dAnalizer(stereo_config_file)
    video_recorder.start()

    while video_recorder.keep_loop:
        time.sleep(1)
