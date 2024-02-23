
from .cameraArray import CamArray, StereoConfig, CameraConfig
from typing import List

import json
from dataclasses import dataclass


def is_binocular_cam(stereoConfig: StereoConfig):
    return stereoConfig.left_camera.idx == stereoConfig.right_camera.idx


def startCameraArray(stereo_config: StereoConfig) -> CamArray:
    if stereo_config.save_images:
        return CamArray((stereo_config.left_camera,
                         stereo_config.right_camera),
                        save_frames_to=stereo_config.save_path)

    return CamArray((stereo_config.left_camera,
                     stereo_config.right_camera))


def loadStereoCameraConfig(json_fname: str) -> StereoConfig:
    json_file = open(json_fname)
    stero_config = json.load(json_file)

    sep = stero_config["separation"]
    stereo_map_file = stero_config["stereo_map_file"]
    depth_to_pixel_size = stero_config["depth_to_pixel_size"]
    show_images = stero_config["show_images"]

    save_imgs = stero_config["save_images"]

    save_imgs_path = stero_config["save_path"]

    left_camera = CameraConfig(
        stero_config["left_camera"]["idx"],
        stero_config["left_camera"]["fpx"],
        (stero_config["left_camera"]["center"][0],
         stero_config["left_camera"]["center"][1]),
        (stero_config["left_camera"]["resolution"][0],
         stero_config["left_camera"]["resolution"][1]),
        stero_config["left_camera"]["fps"]
    )

    right_camera = CameraConfig(
        stero_config["right_camera"]["idx"],
        stero_config["right_camera"]["fpx"],
        (stero_config["right_camera"]["center"][0],
         stero_config["right_camera"]["center"][1]),
        (stero_config["right_camera"]["resolution"][0],
         stero_config["right_camera"]["resolution"][1]),
        stero_config["right_camera"]["fps"]
    )

    return StereoConfig(left_camera, right_camera, sep,
                        stereo_map_file, depth_to_pixel_size,
                        show_images, save_imgs,  save_imgs_path)
