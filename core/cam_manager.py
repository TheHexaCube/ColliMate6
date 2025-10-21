import pypylon.pylon as pylon
import os
import threading
import core.img_processing as proc
from queue import Queue, Full, Empty
import time
import numpy as np

FRAME_QUEUE_SIZE = 10

class CamManager:
    def __init__(self, debug_cam=False):
        self._debug_cam = debug_cam
        if self._debug_cam:
            os.environ["PYLON_CAMEMU"] = "1"

        self.tl_factory = pylon.TlFactory.GetInstance()
        self.available_cams = self.tl_factory.EnumerateDevices()
        self.selected_cam = None

        self._capture_thread = None
        self._process_thread = None
        self._stop_event = threading.Event()

        self._raw_frame = Queue(maxsize=FRAME_QUEUE_SIZE)
        self._processed_frame = Queue(maxsize=FRAME_QUEUE_SIZE)
        self._avg_frame = Queue(maxsize=FRAME_QUEUE_SIZE)

        self.raw_drop_ctr = 0
        self.processed_drop_ctr = 0
        self.avg_drop_ctr = 0

        self.raw2mono_ms = -1.0
        self.raw2mono_n = 0

        self.mono2rgb_ms = -1.0
        self.mono2rgb_n = 0

        self.proc2avg_ms = -1.0
        self.proc2avg_n = 0
        

    def connect_cam(self, cam_index):
        if 0 <= cam_index < len(self.available_cams):
            self.selected_cam = pylon.InstantCamera(self.tl_factory.CreateDevice(self.available_cams[cam_index]))
            self.selected_cam.Open()
            self.selected_cam.PixelFormat.Value = "BayerRG12"

            if self._debug_cam and "0815" in self.selected_cam.GetDeviceInfo().GetSerialNumber():
                print("Debug camera detected")
                self.selected_cam.Width = 2048
                self.selected_cam.Height = 1536          
                self.selected_cam.AcquisitionFrameRateEnable = True
                self.selected_cam.AcquisitionFrameRate = 55

            self.selected_cam.ExposureTime.Value = 100
            self.selected_cam.Gain.Value = 0
            print(f"Camera connected: {self.selected_cam.GetDeviceInfo().GetModelName()} [Ser.-No.: {self.selected_cam.GetDeviceInfo().GetSerialNumber()}]")
            
            return True
        return False
    
    def disconnect_cam(self):
        if self._capture_thread and self._capture_thread.is_alive():
            self._stop_event.set()
            self._capture_thread.join()

        if self.selected_cam is not None:
            self.selected_cam.Close()
            self.selected_cam = None

        print(f"Camera disconnected")
        return True

    def start_capture(self):
        if self._capture_thread and self._capture_thread.is_alive():
            return
        self._stop_event.clear()
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()
        self._process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self._process_thread.start()

    def stop_capture(self):
        if self._capture_thread and self._capture_thread.is_alive():
            self._stop_event.set()
            self._capture_thread.join(timeout=1.0)
            if self._capture_thread.is_alive():
                print("Warning: capture thread did not exit cleanly")
        if self._process_thread and self._process_thread.is_alive():
            self._process_thread.join(timeout=1.0)
            if self._process_thread.is_alive():
                print("Warning: process thread did not exit cleanly")

    
    def _capture_loop(self):
        print("Capture loop started")
        while not self._stop_event.is_set():
            if self.selected_cam is not None:
                self.selected_cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
                while not self._stop_event.is_set():
                    if self.selected_cam.IsGrabbing():
                        image = self.selected_cam.RetrieveResult(5000)
                        if image.GrabSucceeded():
                            try:                                
                                self._raw_frame.put_nowait(image.GetArray())
                            except Full:
                                print("Raw frame queue is full, dropping frame") 
                                self.raw_drop_ctr += 1
                    else:
                        break
                self.selected_cam.StopGrabbing()
            else:
                break

    def _process_loop(self):
        alpha = 0.9  # smoothing factor for moving average
        print("Process loop started")
        while not self._stop_event.is_set():
            
            try:
                frame = self._raw_frame.get()
            except Empty:
                continue

            t0 = time.perf_counter()
            processed_frame = proc.raw_to_mono(frame)
            t1 = time.perf_counter()

            dt_ms = (t1 - t0) * 1000.0  # convert to milliseconds

            # Initialize on first frame
            if self.raw2mono_ms < 0:
                self.raw2mono_ms = dt_ms
            else:
                # exponential moving average in milliseconds
                self.raw2mono_ms = alpha * self.raw2mono_ms + (1 - alpha) * dt_ms

    

            try:
                self._processed_frame.put_nowait(processed_frame)
            except Full:
                print("Processed frame queue is full, dropping frame")
                self.processed_drop_ctr += 1

            # # convert queue to ndarray
            # processed_frames = np.array(list(self._processed_frame.queue))

            # t2 = time.perf_counter()
            # avg_frame = proc.average_frames(processed_frames)
            # t3 = time.perf_counter()
            # dt_ms = (t3 - t2) * 1000.0  # convert to milliseconds
            # if self.proc2avg_ms < 0:
            #     self.proc2avg_ms = dt_ms
            # else:
            #     # exponential moving average in milliseconds
            #     self.proc2avg_ms = 0.9 * self.proc2avg_ms + (1 - 0.9) * dt_ms

            # try:
            #     self._avg_frame.put_nowait(avg_frame)
            #     self._processed_frame.get_nowait()
            # except Full:
            #     print("Average frame queue is full, dropping frame")
            #     self.avg_drop_ctr += 1

            t2 = time.perf_counter()
            
            frames = list(self._processed_frame.queue)
            self._processed_frame.get_nowait()
            if frames:
                count = min(10, len(frames))
                # create contiguous 3D array from last `count` frames
                arr = np.ascontiguousarray(np.stack(frames[-count:], axis=0), dtype=np.float32)
                avg_frame = proc.average_frames(arr, count)
            else:
                avg_frame = processed_frame
            t3 = time.perf_counter()
            dt_ms = (t3 - t2) * 1000.0  # convert to milliseconds
            if self.proc2avg_ms < 0:
                self.proc2avg_ms = dt_ms
            else:
                # exponential moving average in milliseconds
                self.proc2avg_ms = 0.9 * self.proc2avg_ms + (1 - 0.9) * dt_ms

            try:
                self._avg_frame.put(avg_frame)                 
            except Full:
                print("Average frame queue is full, dropping frame")
                self.avg_drop_ctr += 1

            
            


            
           
            


            
                 
       
    
        
    def get_available_cams(self):
        self.available_cams = self.tl_factory.EnumerateDevices()
        return [(index, device.GetModelName(), device.GetSerialNumber()) for index, device in enumerate(self.available_cams)] 

    def get_resolution(self):
        return self.selected_cam.Width.Value, self.selected_cam.Height.Value

    def get_frame_test(self, type="avg"):
        print(f"Getting {type} frame")
        try: 
            if type == "avg":
                frame = self._avg_frame.get()
                print(f"Got {type} frame")
            elif type == "proc":
                frame = self._processed_frame.get()
                print(f"Got {type} frame")
            elif type == "raw":
                frame = self._raw_frame.get()
                print(f"Got {type} frame")
        except Empty:
            print(f"{type} frame queue is empty")
            return None
        else:
            t2 = time.perf_counter()
            frame = proc.mono_to_rgb(frame) 
            t3 = time.perf_counter()
            dt_ms = (t3 - t2) * 1000.0  # convert to milliseconds
            if self.mono2rgb_ms < 0:
                self.mono2rgb_ms = dt_ms
            else:
                # exponential moving average in milliseconds
                self.mono2rgb_ms = 0.9 * self.mono2rgb_ms + (1 - 0.9) * dt_ms
            return frame
        
        
    def get_time_avg(self, type):
        if type == "avg":
            return self.proc2avg_ms
        elif type == "proc":
            return self.raw2mono_ms
        elif type == "mono":
            return self.mono2rgb_ms

    def get_drop_ctr(self, type="avg"):
        if type == "avg":
            return self.avg_drop_ctr
        elif type == "processed":
            return self.processed_drop_ctr
        elif type == "raw":
            return self.raw_drop_ctr
        

    def set_exposure_time(self, exposure_time):
        if self.selected_cam is not None:
            self.selected_cam.ExposureTime.Value = exposure_time
            print(f"Exposure time set to {exposure_time}")

    def get_exposure_time(self):
        if self.selected_cam is not None:
            return self.selected_cam.ExposureTime.Value
        return 0
       
