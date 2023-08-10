import pyautogui
import pygetwindow as gw

class WindowManagement:
    def __init__(self, main_app):
        self.main_app = main_app
        # Initialization code if needed
        pass

    def calculate_x_y_using_wrist(self, wrist_position, frame, hand_id):
        positions_buffer = self.main_app.positions_buffer
        mapped_x, mapped_y = self.map_video_to_screen(wrist_position[0], wrist_position[1],
                                                      frame.shape[1],
                                                      frame.shape[0])

        positions_buffer.append((mapped_x, mapped_y))
        if len(positions_buffer) > 5:  # Take average over the last 5 positions for smoothing
            positions_buffer.pop(0)

        average_position = (sum([pos[0] for pos in positions_buffer]) / len(positions_buffer),
                            sum([pos[1] for pos in positions_buffer]) / len(positions_buffer))

        # Get the current window's position
        current_window_position = gw.getActiveWindow()._rect.topleft
        new_x, new_y = self.interpolate_movement(current_window_position, average_position, hand_id=hand_id)
        return new_x, new_y

    @staticmethod
    def move_current_window(x, y):
        """Function to move the current active window."""
        current_window = gw.getActiveWindow()  # Get the currently active window
        if current_window:
            current_window.moveTo(x, y)

    @staticmethod
    def map_video_to_screen(x, y, frame_width, frame_height):
        """
        Map x and y coordinates from video dimensions to screen dimensions.
        """
        screen_width, screen_height = pyautogui.size()
        mapped_x = int(x / frame_width * screen_width)
        mapped_y = int(y / frame_height * screen_height)
        return mapped_x, mapped_y

    def interpolate_movement(self, old_pos, new_pos, factor=2.5, hand_id=None):
        """
        Interpolate between old and new position and multiply the difference by a factor.
        """
        prev_wrist_position = self.main_app.hand_states[hand_id]['prev_wrist_position']

        if prev_wrist_position:
            dx = (new_pos[0] - prev_wrist_position[0]) * factor
            dy = (new_pos[1] - prev_wrist_position[1]) * factor
            new_x = old_pos[0] + dx
            new_y = old_pos[1] + dy
        else:
            new_x, new_y = old_pos

        self.main_app.hand_states[hand_id]['prev_wrist_position'] = new_pos

        return int(new_x), int(new_y)
