
import cv2
import sys
import threading
import numpy as np

from stereo_lib.utils import loadStereoCameraConfig, startCameraArray
from stereo_lib.calibration import getStereoRectifier
from stereo_lib.triangulation import find_depth
from stereo_lib.face3d import LivePlot3d
from face_mesh import FeaturesExtractor, FaceFeatures
from time import sleep


def get_coordinaties_3d(keypoints, pixel_size, center_pt):
    kpts_displaced = keypoints - center_pt
    coords_3d = kpts_displaced
    coords_3d[:, 0] = coords_3d[:, 0] * pixel_size
    coords_3d[:, 1] = coords_3d[:, 1] * pixel_size
    return coords_3d[:, 0], coords_3d[:, 1]


class PointCloudExtractor:
    def __init__(self, config_file, name):
        self.stereo_config = loadStereoCameraConfig(config_file)
        #  cam_center_left, cam_center_right = cam_centers
        self.f_length = min(self.stereo_config.left_camera.fpx,
                            self.stereo_config.right_camera.fpx)
        self.cams = startCameraArray(self.stereo_config.left_camera,
                                     self.stereo_config.right_camera,
                                     self.stereo_config)
        rectifier = getStereoRectifier(self.stereo_config.stereo_map_file)
        self.cams.rectifier = rectifier

        self.features_left = FeaturesExtractor()
        self.features_right = FeaturesExtractor()

        self.keep_loop = True
        self.name = name
        self.thread = None
        self.plot3d_1 = LivePlot3d()
        #  self.plot3d_2 = LivePlot3d()

    def run(self):
        self.cams.start()
        coords = []
        while self.cams.isOpened() and self.keep_loop:
            frame_left, frame_right = self.cams.get_frames()
            succes_left, frame_left = frame_left
            succes_right, frame_right = frame_right

            if not succes_right or not succes_left:
                print("Ignoring empty camera frame.")
                continue

            left_kpts = self.features_left.extract_keypts(frame_left)
            right_kpts = self.features_right.extract_keypts(frame_right)

            if not left_kpts[0] or not right_kpts[0]:
                continue

            depth = find_depth(left_kpts[2], right_kpts[2],
                               self.stereo_config.cam_separation,
                               self.f_length)
            px_size = self.stereo_config.depth_to_pixel_size * depth
            x, y = get_coordinaties_3d(left_kpts[2], px_size,
                                       left_kpts[1].nose)
            z = depth - depth[2]
            #  self.plot3d_1.clean()
            #  self.plot3d_1.plot(x, y, z, marker="o")
            #  self.plot3d_1.plot(x, y, depth, marker="o")

            eye1 = np.array((x[33], y[33], z[33]))
            eye2 = np.array((x[263], y[263], z[263]))
            nose = np.array((x[2], y[2], z[2]))
            eye_middle = np.mean((eye1, eye2), axis=0)
            #  mouth_1 = np.array((x[13], y[13], z[13]))
            #  mouth_2 = np.array((x[14], y[14], z[14]))
            #  mouth = np.mean((mouth_1, mouth_2), axis=0)
            v1 = eye_middle - nose
            v2 = eye2 - eye_middle
            v3 = np.cross(v1, v2)

            v1 = v1 / np.linalg.norm(v1)
            v2 = v2 / np.linalg.norm(v2)
            v3 = v3 / np.linalg.norm(v3)

            base = np.array((v2, -v1, v3)).T
            M = np.linalg.inv(base)
            xyz = np.array((x, y, z)).T
            new_coords = np.array([M@coord for coord in xyz])

            new_coords[:, 0] = (new_coords[:, 0] - new_coords[:, 0].min()) / \
                (new_coords[:, 0].max() - new_coords[:, 0].min())
            new_coords[:, 1] = (new_coords[:, 1] - new_coords[:, 1].min()) / \
                (new_coords[:, 1].max() - new_coords[:, 1].min())
            new_coords[:, 2] = (new_coords[:, 2] - new_coords[:, 2].min()) / \
                (new_coords[:, 2].max() - new_coords[:, 2].min())

            new_coords = new_coords.T
            coords.append(new_coords)
            #  self.plot3d_1.plot(*(new_coords.T), marker='o')
        coords = np.array(coords)
        face = np.median(coords, axis=0)
        np.savetxt(f"./embedings/{self.name}.pc", face)

        self.cams.close()
        self.keep_loop = False

    def start(self):
        self.thread = threading.Thread(target=self.run)
        self.thread.start()

    def close(self):
        self.keep_loop = False

    def showImages(self, frame_left, frame_right):
        if self.stereo_config.show_images:
            cv2.imshow("frame right", cv2.resize(
                frame_right, (np.array(frame_right.shape[:2][::-1])*1.5).astype(int)))
            cv2.imshow("frame left", cv2.resize(
                frame_left, (np.array(frame_left.shape[:2][::-1])*1.5).astype(int)))
            if cv2.waitKey(5) & 0xFF == 27:
                return True
            return False


if __name__ == "__main__":
    json_config = sys.argv[1]
    name = sys.argv[2]
    pc_extractor = PointCloudExtractor(json_config, name)
    pc_extractor.run()
