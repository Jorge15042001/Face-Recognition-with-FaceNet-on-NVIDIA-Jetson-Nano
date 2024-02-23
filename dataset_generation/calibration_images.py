import cv2
from .cameraArray import CamArray
from datetime import datetime
from .utils import loadStereoCameraConfig, startCameraArray
import sys

#  images_dir = "./images"
#  if not os.path.exists(images_dir):
#      os.makedirs(images_dir)


def save_images(images_dir, img1, img2):
    """Save a stereo image pair with timestamp
    Parameters:
        images_dir (str): path to folder where the image pair should be saved
        img1 (np.ndarray): left image
        img2 (np.ndarray): right image
     """
    now = datetime.now()
    timestamp = now.strftime("%m.%d.%Y_%H:%M:%S.%f")
    cv2.imwrite(f'{images_dir}/imageL__{timestamp}.png', img1)
    cv2.imwrite(f'{images_dir}/imageR__{timestamp}.png', img2)
    print("images saved!")


def preview_and_save(images_dir, img1, img2):
    """Preview stereo pair before saving
    Parameters:
        images_dir (str): path to folder where the image pair should be saved
        img1 (np.ndarray): left image
        img2 (np.ndarray): right image
    """
    while True:
        k2 = cv2.waitKey(5)
        if k2 == ord("q"):
            break
        if k2 == ord("s"):
            save_images(images_dir, img1, img2)
            break


def capture_images(cams: CamArray, images_dir: str):
    """Live view stero pair and capture images\
    With the windows focused press 's' to preview image
        press 's' again to save previewd image
        press 'q' to discard previewed image
    press 'q' to stop capturing images

    Parameters:
        cams(CamArray): camera devices
        images_dir (str): path where images will be saved
    """
    while cams.isOpened():
        frame_left, frame_right = cams.get_frames()
        _, img_left = frame_left
        _, img_right = frame_right

        k = cv2.waitKey(5)

        if k == ord('q'):
            break

        if k == ord('s'):  # preview the image
            preview_and_save(images_dir, img_left, img_right)

        cv2.imshow('Img 1', img_left)
        cv2.imshow('Img 2', img_right)


if __name__ == "__main__":

    # TODO: handle command line parameters correctly
    stereo_config_file = sys.argv[1]
    images_dir = sys.argv[2]

    stereo_config = loadStereoCameraConfig(stereo_config_file)
    print(stereo_config)
    #  cams = loadCamArrayFromJson(stereo_config_file)
    cams = startCameraArray(stereo_config)
    cams.start()
    capture_images(cams, images_dir)
    cams.close()
