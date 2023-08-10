import time
import threading
from ctypes import POINTER, cast

import pyautogui
import pygame
import screen_brightness_control as sbc
from comtypes import CLSCTX_ALL
from pycaw.api.endpointvolume import IAudioEndpointVolume
from pycaw.utils import AudioUtilities


class SystemConfigurationSetter:
    def __init__(self):
        self.last_volume_set_time = 0
        self.width_height = pyautogui.size()

        """Set system volume using pycaw."""
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(
            IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        self.volume = cast(interface, POINTER(IAudioEndpointVolume))
        pygame.mixer.init()
        pass

    def move_cursor(self, position):
        pyautogui.moveTo((position[0] * 1.1) * self.width_height[0], (position[1] * 1.1) * self.width_height[1])

    @staticmethod
    def map_value(x, in_min, in_max, out_min, out_max):
        """Map value from one range to another."""
        return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

    def set_volume_based_on_distance(self, distance, hand_size):
        # Map the distance to volume level and clamp between 0 and 100
        volume_level = self.map_value(distance, 0.04 * hand_size[0], 0.5 * hand_size[0], 0, 100)
        volume_level = max(0, min(100, volume_level))

        # Set system volume
        self.set_volume_threaded(volume_level)

    def set_brightness_based_on_distance(self, distance, hand_size):
        # Extract landmark positions for thumb and index fingertips

        # Map distance to brightness level (adjust the range values based on your preference)
        brightness = self.map_value(distance, 0.04 * hand_size[0], 0.5 * hand_size[0], 0, 100)

        # Ensure brightness is within [0, 100]
        brightness = max(0, min(100, brightness))

        # Set brightness
        sbc.set_brightness(display=0,
                           value=brightness)  # change display=0 if you want to set brightness for a different screen
        return brightness

    def set_volume_threaded(self, volume_level):
        current_time = time.time()

        if current_time - self.last_volume_set_time > 0.25:
            thread = threading.Thread(target=self.set_volume, args=(volume_level,))
            thread.start()
            self.last_volume_set_time = current_time

    def set_volume(self, volume_level):
        time.sleep(1)

        # Convert volume level from 0-100 to the range -65.25 to 0.0

        min_volume, max_volume, _ = self.volume.GetVolumeRange()
        volume_level = self.map_value(volume_level, 0, 100, min_volume, max_volume)
        self.volume.SetMasterVolumeLevel(volume_level, None)
