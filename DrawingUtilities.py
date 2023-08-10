import cv2
import numpy as np


class DrawingUtilities:
    def __init__(self, mp_hands):
        self.mp_hands = mp_hands
        # Define the eye-related landmarks' indices
        self.LEFT_EYE_INDICES = list(range(33, 161))
        self.RIGHT_EYE_INDICES = list(range(263, 387))
        # Initialization code if needed
        pass

    def draw_volume_meter(self, image, landmarks, gesture_distance, tooltip):
        thumb_tip = landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        index_tip = landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]

        thumb_position = (int(thumb_tip.x * image.shape[1]), int(thumb_tip.y * image.shape[0]))
        index_position = (int(index_tip.x * image.shape[1]), int(index_tip.y * image.shape[0]))

        line_center = np.array(
            [(thumb_position[0] + index_position[0]) / 2, (thumb_position[1] + index_position[1]) / 2])
        line_length = np.linalg.norm(np.array(index_position) - np.array(thumb_position))

        # Calculate the percentage of gesture_distance and map it to line_length
        volume_distance = gesture_distance * line_length
        if volume_distance > line_length:
            volume_distance = line_length
            if volume_distance > gesture_distance:
                volume_distance = gesture_distance

        # Calculate the text position at the center of the line
        text_position = (int(line_center[0] - 60), int(line_center[1] + 10))  # Adjust position as needed

        # Draw the volume meter line
        cv2.line(image, thumb_position, index_position, (0, 255, 0), 2)
        cv2.circle(image, tuple(line_center.astype(int)), 5, (0, 255, 0), -1)
        cv2.circle(image, thumb_position, 5, (0, 0, 255), -1)
        cv2.circle(image, index_position, 5, (0, 0, 255), -1)
        cv2.putText(image, f"{tooltip}: {volume_distance:.0f}%", text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (255, 255, 255), 2)

    @staticmethod
    def draw_eye_landmarks(image, landmarks, connections=None):
        if connections:
            for connection in connections:
                start_idx, end_idx = connection
                if start_idx in self.LEFT_EYE_INDICES or start_idx in self.RIGHT_EYE_INDICES:
                    cv2.line(image, (
                        int(landmarks[start_idx].x * image.shape[1]), int(landmarks[start_idx].y * image.shape[0])),
                             (int(landmarks[end_idx].x * image.shape[1]), int(landmarks[end_idx].y * image.shape[0])),
                             (255, 0, 0), 1)
        for landmark in landmarks:
            if landmark in self.LEFT_EYE_INDICES or landmark in self.RIGHT_EYE_INDICES:
                cv2.circle(image, (int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])), 1, (0, 0, 255),
                           -1)

    @staticmethod
    def highlight_gesture(fingertip_landmarks, landmarks, frame):
        for fingertip in fingertip_landmarks:
            landmark = landmarks.landmark[fingertip]
            cv2.circle(frame, (int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])), 6,
                       (255, 0, 0), -1)
