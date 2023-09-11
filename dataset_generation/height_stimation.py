import cv2
import numpy as np
from .calibration import getStereoRectifier
from .triangulation import find_depth_from_disparities
import mediapipe as mp
import threading
from .featuresExtractor import FaceFeatures, FeaturesExtractor
from .utils import startCameraArray, loadStereoCameraConfig, StereoConfig, get_now_str
from time import sleep
import time
import sys

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


def depth_to_pixels_size(x): return 0.001315666666666666*x+0.00979


def computeDepth(keypoinsL, keypoinsR, cams_sep, f_length):
    return find_depth_from_disparities(keypoinsL[:, 0], keypoinsR[:, 0],
                                       cams_sep, f_length)


def computeHeigth(features: FaceFeatures, pixel_size: float):
    mid_eye = np.mean((features.eye1, features.eye2), axis=0)
    mouth_eye = ((mid_eye[0]-features.mouth[0])**2 +
                 (mid_eye[1]-features.mouth[1])**2) ** 0.5
    head_size = mouth_eye * pixel_size * 3
    height = head_size * 8
    return height


def computeHeigth2(features: FaceFeatures, pixel_size: float, cam_center):
    mid_eye_y = np.mean((features.eye1, features.eye2), axis=0)[1]
    cam_center_y = cam_center[1]

    height = (mid_eye_y-cam_center_y)*pixel_size * -1

    #  height = head_size * 8
    return height


def showHeighResult(frame_left, frame_right, height, depth):
    success_height = height is not None

    if success_height:
        cv2.putText(frame_right, "Depth: " + str(round(depth, 1)) +
                    " Height:" + str(round(height, 1)), (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
        cv2.putText(frame_left,  "Depth: " + str(round(depth, 1)) +
                    " Height:" + str(round(height, 1)), (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

    else:
        cv2.putText(frame_right, "right TRACKING LOST", (75, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame_left, "left TRACKING LOST", (75, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("frame right", cv2.resize(
        frame_right, (np.array(frame_right.shape[:2][::-1])*1.5).astype(int)))
    cv2.imshow("frame left", cv2.resize(
        frame_left, (np.array(frame_right.shape[:2][::-1])*1.5).astype(int)))
    if cv2.waitKey(5) & 0xFF == 27:
        return True
    return False


class MoventAnalizer:
    def __init__(self, buff_time):
        self.data = []
        self.threashold: float = 5
        self.bufftime: float = buff_time
        self.on_person_detected = lambda x: None
        self.on_person_leaves = lambda: None
        self.person_detected: bool = False
        self.height: float = 0.
        self.event: threading.Event = threading.Event()
        self.keep_loop: bool = True

        def movement_analizer():
            while self.keep_loop:
                data = list(map(lambda x: x[0], self.data))
                total_elements = len(data)
                none_count = data.count(None)
                not_none_count = total_elements - none_count

                if total_elements == 0:
                    pass

                elif total_elements < 15:
                    pass

                elif total_elements == none_count and self.person_detected:
                    self.person_detected = False
                    print("No movement detected")
                    self.on_person_leaves()

                elif not_none_count/total_elements < 0.8:
                    pass

                else:
                    data_not_none = list(filter(lambda x: x is not None, data))
                    data_min = min(data_not_none)
                    data_max = max(data_not_none)

                    if data_max-data_min < self.threashold and not self.person_detected:
                        self.person_detected = True
                        self.on_person_detected(self.height)

                self.event.wait()
                self.event.clear()

        self.thread = threading.Thread(target=movement_analizer)

    def append_data(self, data_entry):

        now: float = time.time()
        self.data.append((data_entry, now))

        # filter old data
        self.data = list(
            filter(lambda x: now - x[1] < self.bufftime, self.data))
        self.event.set()

    def close(self):
        self.keep_loop = False
        self.event.set()

    def start(self):
        self.thread.start()


class HeightDaemon:
    def __init__(self, config_file):
        self.stereo_config = loadStereoCameraConfig(config_file)
        #  cam_center_left, cam_center_right = cam_centers
        self.f_length = min(self.stereo_config.left_camera.fpx,
                            self.stereo_config.right_camera.fpx)
        self.cams = startCameraArray(self.stereo_config.left_camera,
                                     self.stereo_config.right_camera,
                                     self.stereo_config)

        self.rectify = getStereoRectifier(self.stereo_config.stereo_map_file)
        self.features_left = FeaturesExtractor()
        self.features_right = FeaturesExtractor()
        self.movement_analizer = MoventAnalizer(2)
        self.keep_loop = True

        if self.stereo_config.save_predictions:
            self.out_file = open(
                self.stereo_config.save_predictions_file, "wa")

    def run(self):
        self.cams.start()
        self.movement_analizer.start()
        while self.cams.isOpened() and self.keep_loop:
            now_str = get_now_str()
            frame_left, frame_right = self.cams.get_frames()
            succes_left, frame_left = frame_left
            succes_right, frame_right = frame_right

            if not succes_right or not succes_left:
                print("Ignoring empty camera frame.")
                continue

            frame_left, frame_right = self.rectify(frame_left, frame_right)

            if self.stereo_config.save_rectified_images:
                cv2.imwrite(
                    f"{self.stereo_config.save_input_images_path}/{now_str}.left.rectified.png", frame_left)
                cv2.imwrite(
                    f"{self.stereo_config.save_input_images_path}/{now_str}.right.rectified.png", frame_right)

            features_left = self.features_left.extract_keypts(frame_left)
            features_right = self.features_right.extract_keypts(frame_right)

            if self.stereo_config.save_predictions:
                pass

            if not features_left[0] or not features_right[0]:
                self.movement_analizer.append_data(None)

                terminate = self.showHeighResult(frame_left, frame_right,
                                                 None, None)
                if terminate:
                    break
                continue

            depth = computeDepth(features_left[2], features_right[2],
                                 self.stereo_config.cam_separation,
                                 self.f_length)

            self.movement_analizer.append_data(depth)
            px_size = self.stereo_config.depth_to_pixel_size * depth
            #  height = computeHeigth(features_left[1], px_size)
            height = computeHeigth2(features_left[1], px_size,
                                    self.stereo_config.left_camera.center)

            self.movement_analizer.height = height

            if self.stereo_config.save_predictions:
                self.out_file.write(f"{now_str},{features_left[4]},{features_right[4]},{depth},{height}\n")

            terminate = self.showHeighResult(
                frame_left, frame_right, height, depth)
            if terminate:
                break
        self.movement_analizer.close()
        self.cams.close()
        self.keep_loop = False

    def start(self):
        threading.Thread(target=self.run).start()

    def close(self):
        self.keep_loop = False

    def set_on_person_detected(self, callback):
        self.movement_analizer.on_person_detected = callback

    def set_on_person_leaves(self, callback):
        self.movement_analizer.on_person_leaves = callback

    def showHeighResult(self, frame_left, frame_right, height, depth):
        if self.stereo_config.show_images:
            return showHeighResult(frame_left, frame_right, height, depth)


def person_detected(height):
    print("Person detected", height)


def person_leaves():
    print("Person leaves")


if __name__ == "__main__":
    stereo_config_file = sys.argv[1]
    height_daemon = HeightDaemon(stereo_config_file)
    height_daemon.set_on_person_detected(person_detected)
    height_daemon.set_on_person_leaves(person_leaves)
    height_daemon.run()
