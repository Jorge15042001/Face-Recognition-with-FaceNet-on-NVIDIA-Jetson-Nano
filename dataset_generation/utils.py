
from .cameraArray import CamArray
from typing import List

import json
from dataclasses import dataclass

from datetime import datetime


def get_now_str():
    return datetime.utcnow().strftime('%Y-%m-%d__%H:%M:%S.%f')


@dataclass
class CameraConfig:
    idx: int
    fpx: float()
    center: List[float]


@dataclass
class StereoConfig:
    left_camera: CameraConfig
    right_camera: CameraConfig

    cam_separation: float
    stereo_map_file: str
    depth_to_pixel_size: float
    show_images: bool

    save_input_images: bool
    save_rectified_images: bool
    save_predictions: bool

    save_input_images_path: str
    save_rectified_images_path: str
    save_predictions_file: bool


def startCameraArray(left_camera: CameraConfig,
                     right_camera: CameraConfig,
                     streo_config: StereoConfig) -> CamArray:
    if streo_config.save_input_images:
        return CamArray((left_camera.idx, right_camera.idx),
                        save_frames_to=streo_config.save_input_images_path)

    return CamArray((left_camera.idx, right_camera.idx))


def loadStereoCameraConfig(json_fname: str) -> StereoConfig:
    json_file = open(json_fname)
    stero_config = json.load(json_file)

    sep = stero_config["separation"]
    stereo_map_file = stero_config["stereo_map_file"]
    depth_to_pixel_size = stero_config["depth_to_pixel_size"]
    show_images = stero_config["show_images"]

    save_input_imgs = stero_config["save_input_images"]
    save_rectified_imgs = stero_config["save_rectified_imgs"]
    save_predictions = stero_config["save_predictions"]

    save_input_imgs_path = stero_config["save_input_images_path"]
    save_rectified_imgs_path = stero_config["save_rectified_images_path"]
    save_predictions_file = stero_config["save_predictions_file"]

    left_camera = CameraConfig(
        stero_config["left_camera"]["idx"],
        stero_config["left_camera"]["fpx"],
        stero_config["left_camera"]["center"]
    )

    right_camera = CameraConfig(
        stero_config["right_camera"]["idx"],
        stero_config["right_camera"]["fpx"],
        stero_config["right_camera"]["center"]
    )

    return StereoConfig(left_camera, right_camera, sep,
                        stereo_map_file, depth_to_pixel_size,
                        show_images, save_input_imgs, save_rectified_imgs,
                        save_predictions, save_input_imgs_path,
                        save_rectified_imgs_path, save_predictions_file)
