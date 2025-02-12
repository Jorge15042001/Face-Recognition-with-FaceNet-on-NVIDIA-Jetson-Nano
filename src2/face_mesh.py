import cv2
import numpy as np
import mediapipe as mp
from dataclasses import dataclass

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh


@dataclass
class FaceFeatures:
    eye1: np.array
    eye2: np.array
    mouth: np.array
    nose: np.array

    def normalize(self, width, height):
        scale = np.array([width, height])
        normalized = FaceFeatures(self.eye1/scale,
                                  self.eye2/scale,
                                  self.mouth/scale,
                                  self.nose/scale)
        return normalized

    def __str__(self):
        return f"{self.eye1};{self.eye2};{self.mouth};{self.nose}"

class FeaturesExtractor:
    def __init__(self, save_pred_to=None):
        self.save_pred_to = save_pred_to
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=4,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def extract_keypts(self, frame):
        frame_height, frame_width, _ = frame.shape

        frame.flags.writeable = False  # TODO: does this improve performance?
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.face_mesh.process(frame)

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame.flags.writeable = True

        if not results.multi_face_landmarks:
            return (False, None)

        if len(results.multi_face_landmarks) != 1:
            # only give return when there is just one person detected
            return (False, None, None, None, None)

        rel_coords = np.array(list(map(lambda pt: (pt.x, pt.y),
                                   results.multi_face_landmarks[0].landmark)))
        #  coords = (coords*np.array((frame_width, frame_height))).astype(int)
        img_coords = (rel_coords*np.array((frame_width, frame_height)))
        #  mp_drawing.draw_landmarks(
        #      frame,
        #      results.pose_landmarks,
        #      mp_pose.POSE_CONNECTIONS,
        #      landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        #  )
        mouth = np.mean([img_coords[13, :], img_coords[14, :]], axis=0)
        mouth_rel = np.mean([rel_coords[13, :], rel_coords[14, :]], axis=0)

        features = FaceFeatures(
            img_coords[33, :], img_coords[263, :], mouth, img_coords[2, :])
        features_rel = FaceFeatures(
            rel_coords[33, :], rel_coords[263, :], mouth_rel, rel_coords[2, :])

        return (True, features, img_coords, rel_coords, features_rel)
