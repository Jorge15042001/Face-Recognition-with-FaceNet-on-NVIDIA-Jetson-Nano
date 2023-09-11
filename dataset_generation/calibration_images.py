import cv2
from .cameraArray import CamArray
from datetime import datetime
from .utils import loadStereoCameraConfig, startCameraArray
import sys

#  images_dir = "./images"
#  if not os.path.exists(images_dir):
#      os.makedirs(images_dir)


def flip_correction(img):
    return img[::-1][:, ::-1]


def save_images(images_dir, img1, img2):
    now = datetime.now()
    timestamp = now.strftime("%m.%d.%Y_%H:%M:%S.%f")
    cv2.imwrite(f'{images_dir}/imageL__{timestamp}.png', img1)
    cv2.imwrite(f'{images_dir}/imageR__{timestamp}.png', img2)
    print("images saved!")


def preview_and_save(images_dir, img1, img2):
    while True:
        k2 = cv2.waitKey(5)
        if k2 == ord("q"):
            break
        if k2 == ord("s"):
            save_images(images_dir, img1, img2)
            break


def capture_images(cams: CamArray, images_dir):
    if len(cams.cams) != 2:
        print(f"No support for {len(cams.cams)} cameras, only use 2 cameras ")
        exit(-1)

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

    stereo_config_file = sys.argv[1]
    images_dir = sys.argv[2]

    stereo_config = loadStereoCameraConfig(stereo_config_file)
    #  cams = loadCamArrayFromJson(stereo_config_file)
    cams = startCameraArray(stereo_config.left_camera,
                            stereo_config.right_camera)
    cams.start()
    capture_images(cams, images_dir)
    cams.close()
