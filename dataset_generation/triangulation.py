import numpy as np


def find_depth_from_disparities(right_points, left_points, baseline, f_pixel):

    x_right = np.array(right_points)
    x_left = np.array(left_points)

    # CALCULATE THE DISPARITY:
    # Displacement between left and right frames [pixels]
    disparity = np.abs(x_left-x_right)

    # CALCULATE DEPTH z:
    zDepth = (baseline*f_pixel)/disparity  # Depth in [cm]

    return np.mean(zDepth)


def find_depth(right_points, left_points, baseline, f_pixel):
    disparity = np.abs(left_points-right_points)
    disparity = disparity[:, 0]

    # CALCULATE DEPTH z:
    zDepth = (baseline*f_pixel)/disparity  # Depth in [cm]

    return zDepth
