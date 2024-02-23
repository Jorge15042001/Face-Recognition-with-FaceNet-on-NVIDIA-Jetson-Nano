import numpy as np
import cv2
import glob
import json
import sys

from dataclasses import dataclass
#  from traits.api import Tuple
from typing import Tuple

from .cameraArray import StereoConfig
from .calibration import getStereoRectifier
from .utils import loadStereoCameraConfig, is_binocular_cam


def has_resolution(img, res):
    """Checks if an image have a given resolution

    Parameters:
        img (np.ndarray): image
        res (tuple[int,int]): resolution

    Returns:
        bool: true if img's resolution is res
    """
    img_h, img_w, _ = img.shape
    return res[0] == img_w and res[1] == img_h


def findChessboard(img, chessboardSize, termination_criteria, show=False):
    """Get the coordinates of a chessboard calibration pattern

    Parameters:
        img (np.ndarray): chessboard calibration pattern image
        chessboardSize (type[int,int]): (cols, rows) of the chessboard calbration pattern
        termination_criteria ( ): opencv findChessBoardCorners termination criteria
        show (bool): if true the found corners will be shown in a window, the window will hold excetution  for 1 seconds or until any key is pressed
    Returns:
        bool: whether or not a pattern was found in the image
        corners: list of cordinates found
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(
        gray, chessboardSize, cv2.CALIB_CB_EXHAUSTIVE)
    if not ret:
        return (False, None)
    if show:
        new_img = img.copy()
        cv2.drawChessboardCorners(new_img, chessboardSize, corners, ret)
        cv2.imshow("chessboard", new_img)
        cv2.waitKey(1000)

    corners = cv2.cornerSubPix(
        gray, corners, (4, 4), (-1, -1), termination_criteria)
    return (True, corners)


def createObjp(chessboardSize):
    """Create matrix with indeces for chessboard coordinates as requiered by opencv

    Parameters:
        chessboardSize (tuple[int,int]): (cols, rows) of the chessboard calibration pattern

    Returns:
        np.ndarray: index matrix
    """
    objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboardSize[0],
                           0:chessboardSize[1]].T.reshape(-1, 2)
    return objp


def extractChessboardCoordinates(chessboardSize, frameRes, imgsL, imgsR):
    """Get the chessboard coordinates from a list of left and right images

    Parameters:
        chessboardSize (tuple[int,int]): (cols, rows) of the calibration pattern used
        frameRes (tuple[int,int]): (height, width) of the images used for calibration
        imgsL (list[np.ndarray]): list of images of the calibration pattern captured from the left camera
        imgsR (list[np.ndarray]): list of images of the calibration pattern captured from the right camera

    Returns:
        list[np.ndarray]: list of object points as required by opencv
        list[np.ndarray]: list of corners found in the left images
        list[np.ndarray]: list of corners found in the right images

    """
    termination_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                            30, 0.01)

    objp = createObjp(chessboardSize)

    objpoints = []  # 3d point in real world space
    imgpointsL = []  # 2d points in image plane.
    imgpointsR = []  # 2d points in image plane.

    for imgL, imgR in zip(imgsL, imgsR):

        assert has_resolution(
            imgL, frameRes), f"Left image doesn't match resolution {frameRes}"
        assert has_resolution(
            imgR, frameRes), f"Right image doesn't match resolution {frameRes}"

        retL, cornersL = findChessboard(imgL, chessboardSize,
                                        termination_criteria)
        retR, cornersR = findChessboard(imgR, chessboardSize,
                                        termination_criteria)
        if not retL or not retR:
            # if not able to find corners for any of the images
            # then skip the pair of images
            continue
        # If found, add object points, image points (after refining them)

        objpoints.append(objp)

        imgpointsL.append(cornersL)
        imgpointsR.append(cornersR)
    return (objpoints, imgpointsL, imgpointsR)


def cameraCalibration(objpoints, imgpoints, frameRes):
    """Perform individual camera calibration
    Parameters:
        objpoints(list[np.ndarray]): list of index matrices as required by opencv
        imgpoints(list[np.ndarray]): list of coordinates found
    Returns:
        CalibrationResult: calibration output

    """
    cameraMatrix = cv2.initCameraMatrix2D(objpoints, imgpoints,
                                          frameRes, 0)
    #  newCameraMatrix, _ = cv2.getOptimalNewCameraMatrix(cameraMatrix, dist,
    #                                                     frameRes, 1, frameRes)
    return CalibrationResult(frameRes, imgpoints, cameraMatrix, None, None)


@dataclass
class CalibrationResult:
    frameSize: Tuple[int, int]
    img_points: np.ndarray
    cameraMatrix: np.ndarray
    newCameraMatrix: np.ndarray
    distMatrix: np.ndarray


def cameraCalibrationFromImages(cb_shape, frameRes, imagesLeft, imagesRight):
    """Perform initial camera calibration

    Parameters:
        cb_shape (tuple[int,int]): (cols, rows) of the calibration pattern
        frameRes (tuple[int,int]): (width, height) of the calibration images
        imgsL (list[np.ndarray]): list of images of the calibration pattern captured from the left camera
        imgsR (list[np.ndarray]): list of images of the calibration pattern captured from the right camera
    """
    objpoints, imgpointsL, imgpointsR = extractChessboardCoordinates(
        cb_shape, frameRes, imagesLeft, imagesRight)

    left_calib = cameraCalibration(objpoints, imgpointsL, frameRes)
    right_calib = cameraCalibration(objpoints, imgpointsR, frameRes)

    return (objpoints, left_calib, right_calib)


def saveCameraParameters(filename: str, projMatL: np.ndarray,
                         projMatR: np.ndarray):
    """Save camara parameters in json file

    Parameters:
        filename (str): json filename
        calib_left (CalibrationResult): left calibration result
        calib_right (CalibrationResult): right calibration result
    """
    json_file = open(filename, "r")
    config = json.load(json_file)
    json_file.close()

    config["left_camera"]["fpx"] = projMatL[0, 0]
    config["right_camera"]["fpx"] = projMatR[0, 0]

    config["left_camera"]["center"] = (projMatL[0, 2],
                                       projMatL[1, 2],)
    config["right_camera"]["center"] = (projMatR[0, 2],
                                        projMatR[1, 2],)
    new_config: str = json.dumps(config, indent=4)
    json_file = open(filename, "w")
    json_file.write(new_config)
    json_file.close()


def rectifyImages(rectifier, imgsL, imgsR):
    """Recrify frames list of images

    Parameters:
        rectifier (): function that returns a rectified stereo pair given a unrectified pair
        imgsL (list[np.ndarray]): list of images of the calibration pattern captured from the left camera
        imgsR (list[np.ndarray]): list of images of the calibration pattern captured from the right camera

    Returns:
        list[np.ndarray]: left images rectified
        list[np.ndarray]: right images rectified
    """
    imgsL = []
    imgsR = []
    for imgL, imgR in zip(imagesLeft, imagesRight):
        imgL, imgR = rectifier(imgL, imgR)
        imgsL.append(imgL)
        imgsR.append(imgR)
    return imgsL, imgsR


def stereoCalibration(objpoints: np.array,
                      calib_left: CalibrationResult,
                      calib_rigth: CalibrationResult,
                      calib_file: str) -> tuple[np.ndarray, np.ndarray]:
    """Perform stereo calibration
    Parameters:
        objpoints(list[np.ndarray]): list of index matrices as required by opencv
        calib_left (CalibrationResult): left calibration result
        calib_right (CalibrationResult): right calibration result
        calib_file (str): stereo map file
    Returns:
        np.ndarray: left projection matrix
        np.ndarray: right projection matrix

    """

    flags = cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_SAME_FOCAL_LENGTH
    # Here we fix the intrinsic camara matrixes so that only Rot, Trns,
    # Emat and Fmat are calculated.
    # Hence intrinsic parameters are the same

    criteria_stereo = (cv2.TERM_CRITERIA_EPS +
                       cv2.TERM_CRITERIA_MAX_ITER, 1000, 0.00001)

    # This step is performed to transformation between the two cameras and calculate Essential and Fundamenatl matrix
    _, newCamMtxL, distL, newCamMtxR, distR, rot, trans, _, _ = cv2.stereoCalibrate(
        objpoints, calib_left.img_points, calib_rigth.img_points,
        calib_left.cameraMatrix, None,
        calib_rigth.cameraMatrix, None,
        calib_rigth.frameSize, criteria=criteria_stereo, flags=flags)

    # ######### Stereo Rectification #########################################

    rectL, rectR, projMtxL, projMtxR, Q, roi_L, roi_R = cv2.stereoRectify(
        newCamMtxL, distL, newCamMtxR, distR, calib_rigth.frameSize,
        rot, trans, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)

    stereoMapL = cv2.initUndistortRectifyMap(newCamMtxL, distL, rectL,
                                             projMtxL, calib_rigth.frameSize,
                                             cv2.CV_16SC2)
    stereoMapR = cv2.initUndistortRectifyMap(newCamMtxR, distR, rectR,
                                             projMtxR, calib_rigth.frameSize,
                                             cv2.CV_16SC2)

    cv_file = cv2.FileStorage(calib_file, cv2.FILE_STORAGE_WRITE)

    cv_file.write('stereoMapL_x', stereoMapL[0])
    cv_file.write('stereoMapL_y', stereoMapL[1])
    cv_file.write('stereoMapR_x', stereoMapR[0])
    cv_file.write('stereoMapR_y', stereoMapR[1])

    cv_file.release()

    return projMtxL, projMtxR


if __name__ == "__main__":
    """
    Main calibration pipeline
    """

    images_path = sys.argv[1]
    streo_map_file = sys.argv[2]
    streo_config_file = sys.argv[3]

    stero_config: StereoConfig = loadStereoCameraConfig(streo_config_file)

    imagesPathLeft = sorted(glob.glob(f'{images_path}/imageL*.png'))
    imagesPathRight = sorted(glob.glob(f'{images_path}/imageR*.png'))

    imagesLeft = list(map(cv2.imread, imagesPathLeft))
    imagesRight = list(map(cv2.imread, imagesPathRight))

    img_resolution = stero_config.left_camera.resolution
    if is_binocular_cam(stero_config):
        img_resolution = (stero_config.left_camera.resolution[0]//2,
                          stero_config.left_camera.resolution[1])

    calib_result = cameraCalibrationFromImages(
        (9, 6), img_resolution, imagesLeft, imagesRight)

    projMatL, projMatR = stereoCalibration(*calib_result, streo_map_file)
    print(calib_result[1].cameraMatrix)
    #  print(calib_result[1].newCameraMatrix)
    print(calib_result[2].cameraMatrix)
    #  print(calib_result[2].newCameraMatrix)
    print(projMatL)
    print(projMatR)

    saveCameraParameters(streo_config_file, projMatL, projMatR)
