import cupy as cp
import numpy as np
import time
import threading
from queue import Queue, Full, Empty



class ImgGenerator: 
    def __init__(self, width, height, framerate):
        self.framerate = framerate
        
        self.generator_thread = threading.Thread(target=self.generator_loop, daemon=True)
        self.generator_thread.start()
        


        self.queue = Queue(maxsize=10)

    def generator_loop(self):
        framerate = 1.0 / self.framerate
        next_time = time.perf_counter()

        while True:
        # --- do work here ---
            print("tick", time.perf_counter())

            # --- pacing ---
            next_time += framerate
            sleep_time = next_time - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # loop overran target period
                next_time = time.perf_counter()



if __name__ == "__main__":
    img_generator = ImgGenerator(1800, 1800, 60)
    while img_generator.generator_thread.is_alive():
        time.sleep(1)
    print("Generator thread stopped")