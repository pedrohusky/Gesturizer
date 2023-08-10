import ctypes
import time

import numpy as np
import pyautogui


class HandGestureRecognition:
    def __init__(self, main_app, mp_hands):
        self.main_app = main_app
        self.mp_hands = mp_hands
        self.median_y_start = None
        self.TWIST_THRESHOLD = 10  # Adjust based on sensitivity required
        self.TWIST_TIME_THRESHOLD = 0.5  # Time within which twist should occur
        self.GESTURE_DURATION_THRESHOLD = 1
        # Initialization code if needed
        pass



    @staticmethod
    def orientation(coordinate_landmark_0, coordinate_landmark_9):
        x0 = coordinate_landmark_0[0]
        y0 = coordinate_landmark_0[1]

        x9 = coordinate_landmark_9[0]
        y9 = coordinate_landmark_9[1]

        if abs(x9 - x0) < 0.05:  # since tan(0) --> ∞
            m = 1000000000
        else:
            m = abs((y9 - y0) / (x9 - x0))

        if 0 <= m <= 1:
            if x9 > x0:
                return "Right"
            else:
                return "Left"
        if m > 1:
            if y9 < y0:  # since, y decreases upwards
                return "Up"
            else:
                return "Down"

    @staticmethod
    def compute_angle(p1, p2, p3):
        """Compute the angle formed by three points."""
        vector_a = p2 - p1
        vector_b = p3 - p1
        cosine_angle = np.dot(vector_a, vector_b) / (np.linalg.norm(vector_a) * np.linalg.norm(vector_b))
        angle = np.arccos(cosine_angle)
        return np.degrees(angle)

    def is_hand_claw(self, landmarks, orientation, hand_current_states, hand_size):
        # Get the landmarks for the fingertips and the last joint before the tip for each finger
        thumb_tip = np.array(
            [landmarks[self.mp_hands.HandLandmark.THUMB_TIP].x, landmarks[self.mp_hands.HandLandmark.THUMB_TIP].y])
        thumb_last_joint = np.array(
            [landmarks[self.mp_hands.HandLandmark.THUMB_IP].x, landmarks[self.mp_hands.HandLandmark.THUMB_IP].y])

        index_tip = np.array(
            [landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].x,
             landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y])
        index_last_joint = np.array(
            [landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_DIP].x,
             landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_DIP].y])

        middle_tip = np.array(
            [landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x,
             landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y])
        middle_last_joint = np.array(
            [landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_DIP].x,
             landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y])

        ring_tip = np.array(
            [landmarks[self.mp_hands.HandLandmark.RING_FINGER_TIP].x,
             landmarks[self.mp_hands.HandLandmark.RING_FINGER_TIP].y])
        ring_last_joint = np.array(
            [landmarks[self.mp_hands.HandLandmark.RING_FINGER_DIP].x,
             landmarks[self.mp_hands.HandLandmark.RING_FINGER_DIP].y])

        pinky_tip = np.array(
            [landmarks[self.mp_hands.HandLandmark.PINKY_TIP].x, landmarks[self.mp_hands.HandLandmark.PINKY_TIP].y])
        pinky_last_joint = np.array(
            [landmarks[self.mp_hands.HandLandmark.PINKY_DIP].x, landmarks[self.mp_hands.HandLandmark.PINKY_DIP].y])

        distances = [
            np.linalg.norm(thumb_tip - thumb_last_joint),
            np.linalg.norm(index_tip - index_last_joint),
            np.linalg.norm(middle_tip - middle_last_joint),
            np.linalg.norm(ring_tip - ring_last_joint),
            np.linalg.norm(pinky_tip - pinky_last_joint)
        ]

        # A threshold to determine if the fingertip is closer to the last joint indicating a claw gesture
        claw_threshold = 0.05 * hand_size[0]  # Adjust based on experimentation

        if all(distance < claw_threshold for distance in distances) and orientation == "Up" and all(
                state in hand_current_states for state in
                ['Hand open', 'Index stretched', 'Middle stretched', 'Ring stretched', 'Pinky stretched']):
            return True
        else:
            return False

    def scroll(self, landmarks, hand_current_states, hand_id):
        hand_states = self.main_app.hand_states
        index_tip_y = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y
        middle_tip_y = landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
        median_y_current = (index_tip_y + middle_tip_y) / 2
        index_stretched = 'Index stretched' in hand_current_states
        middle_stretched = 'Middle stretched' in hand_current_states
        other_fingers_closed = all(
            state not in hand_current_states for state in ['Ring stretched', 'Pinky stretched', 'Thumb stretched'])

        # Check if both index and middle finger tips are close
        fingers_close = abs(index_tip_y - middle_tip_y) < 0.04  # Define CLOSE_THRESHOLD based on your needs

        if index_stretched and middle_stretched and fingers_close and other_fingers_closed:
            current_time = time.time()

            if hand_states[hand_id]['scroll_gesture_start_time'] is None:
                hand_states[hand_id]['scroll_gesture_start_time'] = current_time
                self.median_y_start = median_y_current

            distance = median_y_current - self.median_y_start
            direction = "Downwards" if distance > 0 else "Upwards"
            print(f"Distance: {distance} - Direction: {direction}")

            # Check if the gesture has been maintained for at least 1.5 seconds
            if current_time - hand_states[hand_id]['scroll_gesture_start_time'] >= 0.1:
                distance = median_y_current - self.median_y_start
                direction = "Downwards" if distance > 0 else "Upwards"
                hand_states[hand_id]['scroll_gesture_start_time'] = None  # Reset for the next gesture
                scroll_amount = int(distance * 100)  # You may want to adjust this factor based on your use case
                dw_data = scroll_amount * 120  # Scroll amount
                ctypes.windll.user32.mouse_event(0x0800, 0, 0, ctypes.c_int32(dw_data), 0)
                return f"Gesture Detected: {direction}", "Scroll", distance
            else:
                return f"Gesture Detected: ", "Scroll", 0

        else:
            hand_states[hand_id]['scroll_gesture_start_time'] = None  # Reset if the specific gesture is not maintained
            return "", "", None

    def detect_click(self, landmarks, hand_current_states, hand_states, hand_id, hand_size):
        thumb_tip = landmarks[self.mp_hands.HandLandmark.THUMB_TIP]
        index_mcp = landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_DIP]

        thumb_overlapping = thumb_tip.x > index_mcp.x and abs(thumb_tip.y - index_mcp.y) < 0.2 * hand_size[0]

        click_detected = False
        double_click_detected = False

        fingers_closed = all(state in hand_current_states for state in ['Index stretched']) and all(
            state not in hand_current_states for state in ['Pinky stretched', 'Middle stretched', 'Ring stretched'])

        if not hand_states[hand_id]['thumb_overlapping'] and thumb_overlapping and fingers_closed:
            hand_states[hand_id]['thumb_overlapping'] = True
            hand_states[hand_id]['thumb_overlapping_time'] = time.time()
        elif hand_states[hand_id]['thumb_overlapping'] and not thumb_overlapping:
            time_difference = time.time() - hand_states[hand_id]['thumb_overlapping_time']
            if time_difference <= 0.5:
                click_detected = True
            hand_states[hand_id]['thumb_overlapping'] = False
            hand_states[hand_id]['thumb_overlapping_time'] = None

        # Detect double click if two consecutive clicks are detected
        if click_detected and hand_states[hand_id]['previous_click_time'] is not None:
            time_difference = time.time() - hand_states[hand_id]['previous_click_time']
            if time_difference <= 0.5:
                double_click_detected = True
            hand_states[hand_id]['previous_click_time'] = None
        elif click_detected:
            hand_states[hand_id]['previous_click_time'] = time.time()

        if double_click_detected:
            print('Double Click!')
            pyautogui.doubleClick()  # Simulate double-click
        elif click_detected:
            print('Click!')
            pyautogui.click()  # Simulate single-click

    def recognize_gesture(self, landmarks, hand_id, orientation, hand_size):
        hand_states = self.main_app.hand_states
        hand_current_states = self.is_hand_open(landmarks, hand_id, hand_size)
        # print(hand_current_states)

        gesture_status, gesture, value = self.scroll(landmarks, hand_current_states, hand_id)
        if gesture == "Scroll":
            return gesture_status, gesture, value

        # Check for wrist twist first
        wrists = self.detect_wrist_twist(landmarks, hand_id, hand_size)
        if "Wrist twist detected!" in wrists and hand_states[hand_id]['listening_mode']:
            return "Listening mode cancelled", "Wrist twist", None

        if self.is_hand_claw(landmarks, orientation, hand_current_states, hand_size):
            return "Claw", "Claw", None

        is_pinching, pinching_distance = self.is_pinching(landmarks, hand_id, orientation, hand_current_states,
                                                          hand_size)
        if is_pinching:

            # Check if pinky is stretched
            pinky_tip_y = landmarks[self.mp_hands.HandLandmark.PINKY_TIP].y
            pinky_mcp_y = landmarks[self.mp_hands.HandLandmark.PINKY_MCP].y

            if pinky_tip_y < pinky_mcp_y:
                # Detect the brightness gesture
                return f"", "Brightness", pinching_distance

            return "", "Volume", pinching_distance
        else:
            hand_states[hand_id]['listening_mode'] = False

        return "", "", None
        is_pointed, pointed_gesture, pointed_value = self.is_index_pointing(landmarks, orientation, hand_current_states,
                                                                            hand_size, hand_id)
        if pointed_gesture == "Index pointed":
            return "Index Pointed", "Index pointed", pointed_value

        self.detect_click(landmarks, hand_current_states, hand_states, hand_id, hand_size)

    def calculate_hand_size(self, landmarks):
        finger_lengths = [np.linalg.norm(
            np.array([landmarks[mcp].x, landmarks[mcp].y]) - np.array([landmarks[tip].x, landmarks[tip].y]))
            for tip, mcp in zip([self.mp_hands.HandLandmark.THUMB_TIP, self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
                                 self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                                 self.mp_hands.HandLandmark.RING_FINGER_TIP,
                                 self.mp_hands.HandLandmark.PINKY_TIP],
                                [self.mp_hands.HandLandmark.THUMB_MCP, self.mp_hands.HandLandmark.INDEX_FINGER_MCP,
                                 self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
                                 self.mp_hands.HandLandmark.RING_FINGER_MCP,
                                 self.mp_hands.HandLandmark.PINKY_MCP])]

        palm_width = np.linalg.norm(
            np.array([landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_MCP].x,
                      landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_MCP].y]) -
            np.array(
                [landmarks[self.mp_hands.HandLandmark.PINKY_MCP].x, landmarks[self.mp_hands.HandLandmark.PINKY_MCP].y]))

        return sum(finger_lengths), palm_width

    def is_hand_open(self, landmarks, hand_id, hand_size):
        hand_states = self.main_app.hand_states

        # Define distances from tips of fingers to the center of the palm
        palm_center = np.array(
            [(landmarks[self.mp_hands.HandLandmark.WRIST].x + landmarks[
                self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x) / 2,
             (landmarks[self.mp_hands.HandLandmark.WRIST].y + landmarks[
                 self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y) / 2])

        tips = [self.mp_hands.HandLandmark.THUMB_TIP, self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
                self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP, self.mp_hands.HandLandmark.RING_FINGER_TIP,
                self.mp_hands.HandLandmark.PINKY_TIP]

        finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

        distances = [np.linalg.norm(
            np.array([landmarks[tip].x, landmarks[tip].y]) - palm_center) for tip in tips]

        states = []

        # Check if the hand is open, partially open, closed, or partially closed
        avg_distance = sum(distances) / len(distances)
        if avg_distance < 0.1 * hand_size[0]:
            states.append("Hand closed")
        elif avg_distance < 0.15 * hand_size[0]:
            states.append("Hand partially closed")
        elif avg_distance < 0.25 * hand_size[0]:
            states.append("Hand partially open")
        else:
            states.append("Hand open")

        # Check for individual finger stretches
        for i, distance in enumerate(distances):
            if finger_names[i] == 'Thumb':
                if distance > 0.18 * hand_size[0]:
                    states.append(f"{finger_names[i]} stretched")
            else:
                if distance > 0.20 * hand_size[0]:  # Adjust this threshold as needed
                    states.append(f"{finger_names[i]} stretched")

        # Update the hand state information
        hand_states[hand_id]['states'] = states

        return states

    def is_pinching(self, landmarks, hand_id, orientation, hand_current_states, hand_size):
        hand_states = self.main_app.hand_states

        # Extract landmark positions
        thumb_tip = np.array(
            [landmarks[self.mp_hands.HandLandmark.THUMB_TIP].x, landmarks[self.mp_hands.HandLandmark.THUMB_TIP].y])
        index_tip = np.array(
            [landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].x,
             landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y])
        middle_tip = np.array(
            [landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x,
             landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y])
        ring_tip = np.array(
            [landmarks[self.mp_hands.HandLandmark.RING_FINGER_TIP].x,
             landmarks[self.mp_hands.HandLandmark.RING_FINGER_TIP].y])
        pinky_tip = np.array(
            [landmarks[self.mp_hands.HandLandmark.PINKY_TIP].x, landmarks[self.mp_hands.HandLandmark.PINKY_TIP].y])

        distance_tip = np.linalg.norm(thumb_tip - index_tip)

        # Check distance from index to other fingers
        dist_index_middle = np.linalg.norm(index_tip - middle_tip)
        dist_index_ring = np.linalg.norm(index_tip - ring_tip)
        dist_index_pinky = np.linalg.norm(index_tip - pinky_tip)

        # Check if index is overlapping or very close to the thumb
        is_index_near_thumb = distance_tip < 0.03 * hand_size[0]

        # Check if index is closer to other fingers than to the thumb
        is_index_closer_to_others = dist_index_middle < distance_tip or dist_index_ring < distance_tip or dist_index_pinky < distance_tip

        # Check if middle finger tip is close to index tip
        is_middle_near_index = dist_index_middle < 0.07 * hand_size[0]
        # If conditions are met, it's not a pinch
        if (is_index_near_thumb or is_index_closer_to_others or is_middle_near_index) and not hand_states[hand_id][
            'listening_mode']:
            return False, 0

        # Check for initial pinch to activate listening mode

        if (distance_tip < 0.06 * hand_size[0] and not hand_states[hand_id][
            'listening_mode'] and orientation == "Up" and
                (all(
                    state in hand_current_states for state in
                    ['Hand open', 'Thumb stretched', 'Index stretched', 'Middle stretched', 'Ring stretched',
                     'Pinky stretched']) or
                 all(state in hand_current_states for state in
                     ['Hand partially open', 'Thumb stretched', 'Index stretched']) and not all(
                            state in hand_current_states for state in
                            ['Middle stretched', 'Ring stretched', 'Pinky stretched']))):

            if hand_states[hand_id]['start_time'] is None:
                hand_states[hand_id]['start_time'] = time.time()
            elif time.time() - hand_states[hand_id]['start_time'] > self.GESTURE_DURATION_THRESHOLD:
                self.main_app.sound_playback.play_sound("./sounds/volume_recognized.mp3")
                hand_states[hand_id]['listening_mode'] = True
                hand_states[hand_id]['start_time'] = time.time()
                print("Started listening to volume changes.")
        else:
            hand_states[hand_id]['start_time'] = None

        # If in listening mode and fingers are not curled, return True
        if hand_states[hand_id]['listening_mode']:
            return True, distance_tip

        return False, 0

    def is_index_pointing(self, landmarks, orientation, hand_current_states, hand_size, hand_id):
        hand_states = self.main_app.hand_states
        index_tip_y = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y
        middle_tip_y = landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
        median_y_current = (index_tip_y + middle_tip_y) / 2
        index_stretched = 'Index stretched' in hand_current_states and 'Thumb stretched' in hand_current_states
        hand_partially_closed = 'Hand partially open' in hand_current_states
        other_fingers_closed = all(
            state not in hand_current_states for state in ['Ring stretched', 'Middle stretched', 'Pinky stretched'])

        # Check if both index and middle finger tips are close
        fingers_close = abs(index_tip_y - middle_tip_y) < 0.45 * hand_size[
            0]  # Define CLOSE_THRESHOLD based on your needs

        if index_stretched and hand_partially_closed and fingers_close and other_fingers_closed:
            current_time = time.time()

            if hand_states[hand_id]['pointing_gesture_start_time'] is None:
                hand_states[hand_id]['pointing_gesture_start_time'] = current_time
                self.median_y_start = median_y_current

            index_tip = np.array(
                [landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].x,
                 landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y])

            return f"Gesture Detected:", "Index pointed", index_tip

        else:
            hand_states[hand_id][
                'pointing_gesture_start_time'] = None  # Reset if the specific gesture is not maintained
            return "", "", None

    def detect_wrist_twist(self, landmarks, hand_id, hand_size):
        hand_states = self.main_app.hand_states

        # Ensure that the hand_id exists in hand_states dictionary
        if hand_id not in hand_states:
            hand_states[hand_id] = {'previous_angle': None, 'previous_time': None}

        wrist = np.array([landmarks[self.mp_hands.HandLandmark.WRIST].x, landmarks[self.mp_hands.HandLandmark.WRIST].y])
        middle_mcp = np.array([landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x,
                               landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y])
        middle_tip = np.array([landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x,
                               landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y])

        current_angle = self.compute_angle(wrist, middle_mcp, middle_tip)
        current_time = time.time()

        if hand_states[hand_id]['previous_angle'] is not None:
            angle_difference = abs(current_angle - hand_states[hand_id]['previous_angle'])
            time_difference = current_time - hand_states[hand_id]['previous_time']

            if angle_difference > self.TWIST_THRESHOLD * hand_size[0] and time_difference <= self.TWIST_TIME_THRESHOLD * \
                    hand_size[0]:
                print(f"Wrist twist detected! Time: {time.time()}")
                # Reset to avoid multiple detections
                hand_states[hand_id]['previous_angle'] = None
                hand_states[hand_id]['previous_time'] = None
                return f"Wrist twist detected!"

        hand_states[hand_id]['previous_angle'] = current_angle
        hand_states[hand_id]['previous_time'] = current_time
        return f"No wrist twist"
