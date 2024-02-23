import numpy as np


def find_depth_from_disparities(right_points, left_points,
                                baseline=16, f_pixel=829.4):
    """compute depth from a list of x coordinates
    Parameters:
        left_points(np.ndarray): 1xn or nx1 numpy array containing keypoints x coordinates as viewed from the left camera
        right_points(np.ndarray): 1xn or nx1 numpy array containing keypoints x coordinates as viwed from the right camera
        baseline (float): distance between cameras
        f_pixel (float): focal length of the stereo vision system in pixel units
    """

    x_right = np.array(right_points)
    x_left = np.array(left_points)

    # CALCULATE THE DISPARITY:
    # Displacement between left and right frames [pixels]
    disparity = np.abs(x_left-x_right)

    # CALCULATE DEPTH z:
    zDepth = (baseline*f_pixel)/disparity  # Depth in [cm]

    return np.mean(zDepth)
