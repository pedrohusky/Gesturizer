import numpy as np
import mediapipe as mp

class FaceGestureRecognition:
    def __init__(self, main_app):
        self.main_app = main_app

        # Initialize mediapipe face mesh module
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh()

    @staticmethod
    def detect_eye_closure(landmarks):
        left_eye_top = np.array([landmarks[33].x, landmarks[33].y])
        left_eye_bottom = np.array([landmarks[157].x, landmarks[157].y])
        right_eye_top = np.array([landmarks[263].x, landmarks[263].y])
        right_eye_bottom = np.array([landmarks[387].x, landmarks[387].y])

        left_eye_distance = np.linalg.norm(left_eye_top - left_eye_bottom)
        right_eye_distance = np.linalg.norm(right_eye_top - right_eye_bottom)

        # If both eyes are notably closed, classify as "Both eyes closed"
        re_closure_threshold = 0.0115
        le_closure_threshold = 0.0425
        # print(f"LE D: {left_eye_distance}")
        # print(f"RE D: {right_eye_distance}")
        if left_eye_distance < le_closure_threshold and right_eye_distance < re_closure_threshold:
            return "Both eyes closed"

        return "Eyes open"
