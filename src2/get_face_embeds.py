import cv2
import sys
import threading
import numpy as np

from stereo_lib.utils import loadStereoCameraConfig, startCameraArray
from stereo_lib.calibration import getStereoRectifier
from face_mesh import FeaturesExtractor, FaceFeatures
import traceback
from keras_vggface.vggface import VGGFace
from keras_vggface import utils
from tensorflow.compat.v1 import disable_eager_execution, GPUOptions, Session, ConfigProto, get_default_graph, train
#  from speechbrain.pretrained import SpeakerRecognition
import copy
import os
import time

verification_threshold = 0.75

disable_eager_execution()

json_config = sys.argv[1]


#  audio_path = sys.argv[2]
def crop_face(img, face_points, width_multiplier=1.1, height_multiplier=1.3):
    x_ktps = face_points[:, 0]
    y_ktps = face_points[:, 1]

    x_max, x_min = x_ktps.max(), x_ktps.min()
    y_max, y_min = y_ktps.max(), y_ktps.min()
    #  print(x_max, x_min)
    #  print(y_max, y_min)
    x_center, y_center = (x_max + x_min)/2, (y_max + y_min)/2
    width, height = x_center - x_min, y_center - y_min
    width *= width_multiplier
    height *= height_multiplier

    x_start, x_end = int(x_center - width), int(x_center + width)
    y_start, y_end = int(y_center - height), int(x_center + height)

    #  print(x_start, x_end)
    #  print(y_start, y_end)
    x_start, x_end = max(x_start, 0), min(x_end, img.shape[1]-1)
    y_start, y_end = max(y_start, 0), min(y_end, img.shape[0]-1)
    croped_face = img.copy()[y_start:y_end, x_start:x_end]
    #  print(img.shape)
    return croped_face


class FaceRecognitionVGGFace:
    def __init__(self):
        gpu_options = GPUOptions(per_process_gpu_memory_fraction=0.333)
        self.session = Session(config=ConfigProto(gpu_options=gpu_options))
        self.vggface = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

    def img_to_embedding(self, face):
        #  x = face/255.
        x = face.astype('float32')
        print(x.shape)

        x = np.expand_dims(x, axis=0)
        x = utils.preprocess_input(x, version=1)
        preds = self.session.run(self.vggface.predict(x))
        preds = utils.decode_predictions(preds)
        print(preds.shape)
        return preds


class FaceRecognitionFaceNet:
    def __init__(self):
        gpu_options = GPUOptions(per_process_gpu_memory_fraction=0.333)
        self.session = Session(config=ConfigProto(gpu_options=gpu_options))
        self.net = self.load_face_recognition()
        self.images_placeholder = get_default_graph().get_tensor_by_name("input:0")
        self.embeddings = get_default_graph().get_tensor_by_name("embeddings:0")
        self.phase_train_placeholder = get_default_graph().get_tensor_by_name("phase_train:0")
        self.embedding_size = self.embeddings.get_shape()[1]
        self.dict_embeddings = {}

    # Loading the face recognition model and initialising the tensors with default values.
    def load_face_recognition(self):
        model_path = "../Models/FaceRecognition/"
        saver = train.import_meta_graph(os.path.join(
            model_path, "model-20180204-160909.meta"))
        saver.restore(self.session, os.path.join(
            model_path, "model-20180204-160909.ckpt-266000"))

    # Image to embedding conversion
    def img_to_embedding(self, img, image_size):
        start_time = time.time()
        # Creation of the image tensor
        image = np.zeros((1, image_size, image_size, 3))
        # Convert the image to rgb if it is in greyscale
        if img.ndim == 2:
            imagen = copy.deepcopy(img)
            w, h = imagen.shape
            img = np.empty((w, h, 3), dtype=np.uint8)
            img[:, :, 0] = img[:, :, 1] = img[:, :, 2] = imagen
        # Pre - whitening to image
        std_adj = np.maximum(np.std(img), 1.0 / np.sqrt(img.size))
        img = np.multiply(np.subtract(img, np.mean(img)), 1 / std_adj)
        image[0, :, :, :] = img
        # Conversion to embedding
        feed_dict = {self.images_placeholder: image,
                     self.phase_train_placeholder: False}
        emb_array = np.zeros((1, self.embedding_size))
        emb_array[0, :] = self.session.run(
            self.embeddings, feed_dict=feed_dict)
        elapsed_time = time.time() - start_time
        fps = 1 / elapsed_time
        print(f"fps: {fps}")
        return np.squeeze(emb_array)

    @staticmethod
    def is_same(emb1, emb2):
        diff = np.subtract(emb1, emb2)
        diff = np.sum(np.square(diff))
        return diff < verification_threshold, diff


class StereoVideoAnalizer:
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

        self.face_recognizer_fn = FaceRecognitionFaceNet()
        self.face_recognizer_vgg = FaceRecognitionVGGFace()
        self.keep_loop = True
        self.name = name
        self.thread = None

    def run(self):
        face_embedings = []
        self.cams.start()
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

            face_left = crop_face(frame_left, left_kpts[2])
            face_right = crop_face(frame_right, right_kpts[2])
            try:
                left_embeding = self.face_recognizer_fn.img_to_embedding(
                    cv2.resize(face_left, (160, 160)), 160)
                right_embeding = self.face_recognizer_fn.img_to_embedding(
                    cv2.resize(face_right, (160, 160)), 160)
                self.face_recognizer_vgg.img_to_embedding(
                    cv2.resize(face_left, (224, 224)))

                same_person = FaceRecognitionFaceNet.is_same(
                    left_embeding, right_embeding)
                if (same_person):
                    face_embedings.append(left_embeding)
                    face_embedings.append(right_embeding)
                terminate = self.showImages(face_left, face_right)
                if terminate:
                    break
            except Exception as e:
                #  print(e.with_traceback())
                traceback.print_exc()
                break

        self.cams.close()
        self.keep_loop = False

        face_embedings = np.array(face_embedings)
        face_embeding_final = np.median(face_embedings, axis=0)

        failed = 0
        for emb in face_embedings:
            same = FaceRecognitionFaceNet.is_same(emb, face_embeding_final)
            if not same:
                failed += 1
        print(failed/len(face_embedings))

        np.savetxt(f"./embedings/{self.name}_face.emb", face_embeding_final)
        np.savetxt(f"./embedings/{self.name}_face.embs", face_embedings)

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
    name = sys.argv[2]
    stereo_view = StereoVideoAnalizer(json_config, name)
    stereo_view.start()
    stereo_view.thread.join()
