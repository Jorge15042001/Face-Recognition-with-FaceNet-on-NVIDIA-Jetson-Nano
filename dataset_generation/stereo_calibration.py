import numpy as np
import cv2
import glob
import json
import sys

from dataclasses import dataclass
#  from traits.api import Tuple
from typing import Tuple
from .calibration import getStereoRectifier


def has_resolution(img, res):
    img_h, img_w, _ = img.shape
    return res[0] == img_w and res[1] == img_h


def findChessboard(img, chessboardSize, termination_criteria):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(
        gray, chessboardSize, cv2.CALIB_CB_EXHAUSTIVE)
    if not ret:
        return (False, None)
    corners = cv2.cornerSubPix(
        gray, corners, (11, 11), (-1, -1), termination_criteria)
    return (True, corners)


def createObjp(chessboardSize):
    objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboardSize[0],
                           0:chessboardSize[1]].T.reshape(-1, 2)
    return objp


def extractChessboardCoordinates(chessboardSize, frameRes, imgsL, imgsR):
    termination_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                            30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
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
    _, cameraMatrix, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints,
                                                      frameRes, None, None)
    newCameraMatrix, _ = cv2.getOptimalNewCameraMatrix(cameraMatrix, dist,
                                                       frameRes, 1, frameRes)
    return CalibrationResult(frameRes, imgpoints, cameraMatrix, newCameraMatrix, dist)

################ FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS #############################


@dataclass
class CalibrationResult:
    frameSize: Tuple[int, int]
    img_points: np.array
    cameraMatrix: np.array
    newCameraMatrix: np.array
    distMatrix: np.array
    #  roi: np.array


def cameraCalibrationFromImages(cb_shape, frameRes, imagesLeft, imagesRight):
    objpoints, imgpointsL, imgpointsR = extractChessboardCoordinates(
        cb_shape, frameRes, imagesLeft, imagesRight)

    left_calib = cameraCalibration(objpoints, imgpointsL, frameRes)
    right_calib = cameraCalibration(objpoints, imgpointsR, frameRes)

    return (objpoints, left_calib, right_calib)


def saveCameraParameters(filename: str, calib_left: CalibrationResult,
                         calib_right: CalibrationResult):
    json_file = open(filename, "r")
    config = json.load(json_file)
    json_file.close()

    config["left_camera"]["fpx"] =  calib_left. newCameraMatrix[0, 0]
    config["right_camera"]["fpx"] = calib_right.newCameraMatrix[0, 0]

    config["left_camera"]["center"] = (calib_left.newCameraMatrix[0, 2],
                                       calib_left.newCameraMatrix[1, 2],)
    config["right_camera"]["center"] = (calib_right.newCameraMatrix[0, 2],
                                        calib_right.newCameraMatrix[1, 2],)
    new_config:str = json.dumps(config, indent=4)
    json_file = open(filename, "w")
    json_file.write(new_config)
    json_file.close()


def rectifyImages(rectifier, imgsL, imgsR):
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
                      calib_file: str):

    flags = 0
    flags |= cv2.CALIB_FIX_INTRINSIC
    # Here we fix the intrinsic camara matrixes so that only Rot, Trns,
    # Emat and Fmat are calculated.
    # Hence intrinsic parameters are the same

    criteria_stereo = (cv2.TERM_CRITERIA_EPS +
                       cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # This step is performed to transformation between the two cameras and calculate Essential and Fundamenatl matrix
    _, newCamMtxL, distL, newCamMtxR, distR, rot, trans, _, _ = cv2.stereoCalibrate(
        objpoints, calib_left.img_points, calib_rigth.img_points,
        calib_left.newCameraMatrix, calib_left.distMatrix,
        calib_rigth.newCameraMatrix, calib_rigth.distMatrix,
        calib_rigth.frameSize, criteria_stereo, flags)

    # ######### Stereo Rectification #########################################

    rectifyScale = 1
    rectL, rectR, projMtxL, projMtxR, Q, roi_L, roi_R = cv2.stereoRectify(
        newCamMtxL, distL, newCamMtxR, distR, calib_rigth.frameSize,
        rot, trans, rectifyScale, (0, 0))

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


if __name__ == "__main__":

    images_path = sys.argv[1]
    streo_map_file = sys.argv[2]
    streo_config_file = sys.argv[3]

    imagesPathLeft = sorted(glob.glob(f'{images_path}/imageL*.png'))
    imagesPathRight = sorted(glob.glob(f'{images_path}/imageR*.png'))

    imagesLeft = list(map(cv2.imread, imagesPathLeft))
    imagesRight = list(map(cv2.imread, imagesPathRight))

    calib_result = cameraCalibrationFromImages((8, 6), (640, 480),
                                               imagesLeft, imagesRight)
    stereoCalibration(*calib_result, streo_map_file)

    # find camera matrix of rectified images
    rectifier = getStereoRectifier(streo_map_file)

    rectifiedL, rectifiedR = rectifyImages(rectifier, imagesLeft, imagesRight)
    calib_rectified_result = cameraCalibrationFromImages((8, 6), (640, 480),
                                                         rectifiedL, rectifiedR)
    print(calib_result[1].cameraMatrix)
    print(calib_result[1].newCameraMatrix)
    print(calib_result[2].cameraMatrix)
    print(calib_result[2].newCameraMatrix)
    print(calib_rectified_result[1].cameraMatrix)
    print(calib_rectified_result[1].newCameraMatrix)
    print(calib_rectified_result[2].cameraMatrix)
    print(calib_rectified_result[2].newCameraMatrix)
    saveCameraParameters(streo_config_file,
                         calib_result[1], calib_result[2])
