import threading
import time

import pygame


class SoundPlayback:
    def __init__(self):
        self.last_played_sound_time = 0
        # Initialization code if needed
        pass

    @staticmethod
    def play_mp3(file_path):
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()

    # Start the play_mp3 function in a separate thread
    def play_sound(self, file_path):
        current_time = time.time()

        if current_time - self.last_played_sound_time > 1:
            thread = threading.Thread(target=self.play_mp3, args=(file_path,))
            thread.start()
            self.last_played_sound_time = current_time

    # [Include other necessary methods related to sound playback
