import time
import cv2
import mediapipe as mp
import threading

import numpy as np
import pyautogui
from DrawingUtilities import DrawingUtilities
from HandGestureRecognition import HandGestureRecognition
from SoundPlayback import SoundPlayback
from SystemConfigurationSetter import SystemConfigurationSetter
from WindowManagement import WindowManagement

# In your global variables section, add:
GESTURE_DURATION_THRESHOLD = 1  # time in seconds
MOVEMENT_DELAY = 0.1  # Delay in seconds between window movements

# Initialize mediapipe hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

mp_drawing_styles = mp.solutions.drawing_styles
# Initialize mediapipe drawing module
mp_drawing = mp.solutions.drawing_utils


def map_value(x, in_min, in_max, out_min, out_max):
    """Map value from one range to another."""
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


class MainApplication:
    def __init__(self):
        self.hand_gesture_recognition = HandGestureRecognition(self, mp_hands)
        self.window_management = WindowManagement(self)
        self.sound_playback = SoundPlayback()
        self.drawing_utilities = DrawingUtilities(mp_hands)
        self.system_settings = SystemConfigurationSetter()

        # Global variables to maintain state
        self.hand_states = {}  # Dictionary to store hand-specific states for each hand
        self.positions_buffer = []

    # Initialize or update the hand_states dictionary structure
    @staticmethod
    def initialize_hand_states(hand_id, hand_states):
        if hand_id not in hand_states:
            hand_states[hand_id] = {
                'start_time': None,
                'listening_mode': False,
                'previous_angle': None,
                'previous_time': None,
                'claw_gesture_start_time': None,
                'scroll_gesture_start_time': None,
                'pointing_gesture_start_time': None,
                'last_movement_time': None,
                'prev_wrist_position': None,
                'open': False,
                'tracking_loss_counter': 0,
                'last_trigger_time': 0,
                'previous_click_time': None,
                'thumb_overlapping': None,
                'thumb_overlapping_time': None
            }

        return hand_states[hand_id]

    # Compute the palm position of a hand given its landmarks
    @staticmethod
    def compute_palm_position(landmarks_list):
        landmarks = landmarks_list.landmark

        # Assuming the palm's position is the average of the wrist and middle finger MCP
        palm_x = (landmarks[mp_hands.HandLandmark.WRIST].x + landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x) / 2
        palm_y = (landmarks[mp_hands.HandLandmark.WRIST].y + landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y) / 2
        return (palm_x, palm_y)

    @staticmethod
    def update_tracking_loss(hand_detected, hand_id, hand_states):
        TRACKING_LOSS_THRESHOLD = 5  # Adjust this based on your needs
        if not hand_detected:
            hand_states[hand_id]['tracking_loss_counter'] += 1
            if hand_states[hand_id]['tracking_loss_counter'] > TRACKING_LOSS_THRESHOLD:
                print(f"Deleted hand {hand_id}")
                del hand_states[hand_id]
        else:
            hand_states[hand_id]['tracking_loss_counter'] = 0

    def cancel_gesture_listening(self, hand_label):
        self.hand_states[hand_label]['listening_mode'] = False
        self.hand_states[hand_label]['start_time'] = None
        self.sound_playback.play_sound("./sounds/wrist_twisted.wav")

    @staticmethod
    def enhance_rgb_image(image):
        # Split the image into its color channels
        r, g, b = cv2.split(image)

        # Apply CLAHE to each channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        r = clahe.apply(r)
        g = clahe.apply(g)
        b = clahe.apply(b)

        # Merge the channels back together
        enhanced = cv2.merge([r, g, b])

        # Apply Gaussian blur to reduce noise (optional)
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)

        return blurred

    def run(self):
        # Start capturing video from the first camera device
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # Mirror (flip horizontally) the frame
            image = cv2.flip(image, 1)
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            hands_detected = results.multi_hand_landmarks

            if hands_detected:
                for hand_id, landmarks in enumerate(hands_detected):

                    multiheadness = results.multi_handedness[hand_id].classification[0]
                    # Correctly determine the hand_id using multi_handedness attribute
                    hand_label = multiheadness.index
                    hand_left_or_right = multiheadness.label

                    wrist_position = (int(landmarks.landmark[mp_hands.HandLandmark.WRIST].x * image.shape[1]),
                                      int(landmarks.landmark[mp_hands.HandLandmark.WRIST].y * image.shape[0]))

                    orientation = self.hand_gesture_recognition.orientation(
                        (landmarks.landmark[0].x, landmarks.landmark[0].y),
                        (landmarks.landmark[9].x, landmarks.landmark[9].y))

                    hand_size = self.hand_gesture_recognition.calculate_hand_size(landmarks.landmark)

                    self.initialize_hand_states(hand_label, self.hand_states)
                    self.update_tracking_loss(True, hand_label, self.hand_states)

                    # self.hand_gesture_recognition.is_index_pointing(landmarks, frame)

                    mp_drawing.draw_landmarks(
                        image,
                        landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

                    gesture_status, gesture, gesture_distance = self.hand_gesture_recognition.recognize_gesture(
                        landmarks.landmark, hand_label, orientation, hand_size)

                    if gesture == "Volume":
                        # Draws the highlight in the gesture
                        fingertip_landmarks = [mp_hands.HandLandmark.THUMB_TIP,
                                               mp_hands.HandLandmark.INDEX_FINGER_TIP]

                        volume_level = map_value(gesture_distance, 0.04 * hand_size[0], 0.5 * hand_size[0], 0, 100)
                        volume_level = max(0, min(100, volume_level))

                        self.drawing_utilities.draw_volume_meter(image, landmarks, volume_level, "Volume")

                        self.drawing_utilities.highlight_gesture(fingertip_landmarks, landmarks, image)

                        self.system_settings.set_volume_based_on_distance(gesture_distance, hand_size)
                    elif gesture == "Brightness":
                        # Draws the highlight in the gesture
                        fingertip_landmarks = [mp_hands.HandLandmark.THUMB_TIP,
                                               mp_hands.HandLandmark.INDEX_FINGER_TIP]

                        brightness_level = map_value(gesture_distance, 0.04 * hand_size[0], 0.5 * hand_size[0], 0, 100)
                        brightness_level = max(0, min(100, brightness_level))

                        self.drawing_utilities.draw_volume_meter(image, landmarks, brightness_level, "Brightness")
                        self.drawing_utilities.highlight_gesture(fingertip_landmarks, landmarks, image)

                        thread = threading.Thread(target=self.system_settings.set_brightness_based_on_distance,
                                                  args=(gesture_distance, hand_size))
                        thread.start()
                        gesture_status = f""
                    elif gesture == "Wrist twist":
                        self.cancel_gesture_listening(hand_label)
                    elif gesture == "Claw":
                        # If claw_gesture_start_time is None, initialize it to the current time
                        if not self.hand_states[hand_label]['claw_gesture_start_time']:
                            self.hand_states[hand_label]['claw_gesture_start_time'] = time.time()
                            self.hand_states[hand_label]['last_movement_time'] = time.time()

                        # Check if the gesture has been maintained for the threshold duration
                        if time.time() - self.hand_states[hand_label][
                            'claw_gesture_start_time'] >= GESTURE_DURATION_THRESHOLD:

                            # Draws the highlight in the gesture
                            fingertip_landmarks = [mp_hands.HandLandmark.THUMB_TIP,
                                                   mp_hands.HandLandmark.INDEX_FINGER_TIP,
                                                   mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                                                   mp_hands.HandLandmark.RING_FINGER_TIP,
                                                   mp_hands.HandLandmark.PINKY_TIP]

                            self.drawing_utilities.highlight_gesture(fingertip_landmarks, landmarks, image)

                            wrist_position = (int(landmarks.landmark[mp_hands.HandLandmark.WRIST].x * image.shape[1]),
                                              int(landmarks.landmark[mp_hands.HandLandmark.WRIST].y * image.shape[0]))

                            # Now here we calculate the new x and y based on the wrist position
                            x, y = self.window_management.calculate_x_y_using_wrist(wrist_position, image, hand_label)
                            # break
                            if time.time() - self.hand_states[hand_label]['last_movement_time'] > MOVEMENT_DELAY:
                                threading.Thread(target=self.window_management.move_current_window, args=(x, y)).start()
                                self.hand_states[hand_label]['last_movement_time'] = time.time()
                    elif gesture == "Index pointed":

                        thread = threading.Thread(target=self.system_settings.move_cursor, args=(gesture_distance,))
                        thread.start()

                        # Visual feedback for pointing
                        cv2.circle(image,
                                   (int(gesture_distance[0] * image.shape[1]),
                                    int(gesture_distance[1] * image.shape[0])),
                                   10,
                                   (0, 255, 0), -1)

                    elif gesture == "Scroll":
                        fingertip_landmarks = [mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                                               mp_hands.HandLandmark.INDEX_FINGER_TIP]

                        self.drawing_utilities.highlight_gesture(fingertip_landmarks, landmarks, image)

                    # Display text relative to the wrist_position of each hand
                    cv2.putText(image, orientation, (wrist_position[0], wrist_position[1] + 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (255, 0, 255), 2, cv2.LINE_AA)
                    cv2.putText(image, hand_left_or_right, (wrist_position[0], wrist_position[1] + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (255, 0, 255), 2, cv2.LINE_AA)
                    cv2.putText(image, gesture_status, wrist_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2,
                                cv2.LINE_AA)

            if hands_detected:
                # Extract detected hand IDs
                detected_hand_ids = [hand_id for hand_id, _ in enumerate(hands_detected)]
                # Check each hand_id in self.hand_states
                for hand_id in list(self.hand_states.keys()):
                    if hand_id not in detected_hand_ids:
                        self.update_tracking_loss(False, hand_id, self.hand_states)
            # Display the frame
            cv2.imshow('Volume Control Using Pinch with pycaw (clamped)', image)

            # Exit loop on pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = MainApplication()
    with mp_hands.Hands(
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
        app.run()
