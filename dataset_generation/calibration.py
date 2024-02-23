import cv2
# TODO: change file name


def getStereoRectifier(calib_file):
    """Build rectifier from stereo map file
    Parameters:
        calib_file (str): file name of the stereo map file generated with calibration procedure
    Returns:
        (np.ndarray, np.ndarray)->(np.ndarray,np.ndarray) rectify function takes 2 unrectified images and returns those images calibrated
    """

    # Camera parameters to undistort and rectify images
    cv_file = cv2.FileStorage()
    cv_file.open(calib_file, cv2.FileStorage_READ)

    stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
    stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
    stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
    stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()

    def undistortRectify(frameL, frameR):

        # Undistort and rectify images
        undistortedL = cv2.remap(frameL, stereoMapL_x, stereoMapL_y,
                                 cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
        undistortedR = cv2.remap(frameR, stereoMapR_x, stereoMapR_y,
                                 cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

        return undistortedL, undistortedR

    return undistortRectify
