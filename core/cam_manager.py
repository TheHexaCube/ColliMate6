import pypylon.pylon as pylon
import os
import threading
import core.img_processing as proc
from queue import Queue, Full, Empty
import time
import numpy as np
from core.gpu import GPUQueue, run_demosaic, average_gpuqueue
import cupy as cp

FRAME_QUEUE_SIZE = 8
FRAME_SIZE = (1536, 2048)

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

        self._raw_frame = GPUQueue(max_size=FRAME_QUEUE_SIZE, shape=FRAME_SIZE, dtype=cp.uint16)
        self._processed_frame = GPUQueue(max_size=FRAME_QUEUE_SIZE, shape=(FRAME_SIZE[1], FRAME_SIZE[0], 3), dtype=cp.float32)
        self._avg_frame = GPUQueue(max_size=FRAME_QUEUE_SIZE, shape=(FRAME_SIZE[1], FRAME_SIZE[0], 3), dtype=cp.float32)

        self.raw_drop_ctr = 0
        self.processed_drop_ctr = 0
        self.avg_drop_ctr = 0

        

    def connect_cam(self, cam_index):
        if 0 <= cam_index < len(self.available_cams):
            self.selected_cam = pylon.InstantCamera(self.tl_factory.CreateDevice(self.available_cams[cam_index]))
            self.selected_cam.Open()
            self.selected_cam.PixelFormat.Value = "BayerRG12"

            if self._debug_cam and "0815" in self.selected_cam.GetDeviceInfo().GetSerialNumber():
                print("Debug camera detected")
                self.selected_cam.Width = FRAME_SIZE[0]
                self.selected_cam.Height = FRAME_SIZE[1]          
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
                                #self._raw_frame.put(cp.asarray(image.GetArray()))
                                image_array = cp.asarray(image.GetArray())
                                result = cp.zeros((FRAME_SIZE[1], FRAME_SIZE[0], 3), dtype=cp.float32)
                                run_demosaic(image_array, result)                            
                                self._processed_frame.put(result)

                                avg = cp.empty(self._processed_frame.shape, dtype=cp.float32)
                                average_gpuqueue(self._processed_frame, avg)

                                print(avg.min(), avg.max())
                                self._avg_frame.put(avg)

                             



                            except Full:
                                print(f"camera frame queue is full, dropping frame") 
                                self.raw_drop_ctr += 1
                    else:
                        break
                self.selected_cam.StopGrabbing()
            else:
                break

    
    
        
    def get_available_cams(self):
        self.available_cams = self.tl_factory.EnumerateDevices()
        return [(index, device.GetModelName(), device.GetSerialNumber()) for index, device in enumerate(self.available_cams)] 

    def get_resolution(self):
        return self.selected_cam.Width.Value, self.selected_cam.Height.Value   

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

    def get_frame(self, type="rgb"):
        if type == "rgb":            
            return self._processed_frame.get_view().ravel()
        elif type == "raw":
            return self._raw_frame.get_view().ravel()
        elif type == "avg":
            if self._processed_frame.is_full():
                self._processed_frame.discard_frame()
        
            return self._avg_frame.get_view().ravel()
        
