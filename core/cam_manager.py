import pypylon.pylon as pylon
import os
import threading
import core.img_processing as proc
from queue import Queue, Full, Empty
import time
import numpy as np
from core.gpu import GPUQueue, run_demosaic, average_gpuqueue_windowed
import cupy as cp
from utils.profiling import PerformanceProfiler, TimingContext
from core.overlay_manager import OverlayManager

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

        # Queues and buffers are initialized after camera connect when resolution is known
        self._raw_frame = None
        self._processed_frame = None
        self._avg_frame = None

        self._rgb_buffer = None
        self._avg_buffer = None

        self._avg_window = 8

        self.raw_drop_ctr = 0
        self.processed_drop_ctr = 0
        self.avg_drop_ctr = 0

        # Performance profiling
        self.profiler = PerformanceProfiler(window_size=100)
        self._profiling_enabled = True

        # Overlay manager for GPU-accelerated drawing
        self.overlay_manager = OverlayManager()

        

    def connect_cam(self, cam_index):
        if 0 <= cam_index < len(self.available_cams):
            self.selected_cam = pylon.InstantCamera(self.tl_factory.CreateDevice(self.available_cams[cam_index]))
            self.selected_cam.Open()
            self.selected_cam.PixelFormat.Value = "BayerRG12"

            if self._debug_cam and "0815" in self.selected_cam.GetDeviceInfo().GetSerialNumber():
                print("Debug camera detected")
                # FRAME_SIZE is (H, W); Basler expects Width (W) and Height (H)
                self.selected_cam.Width = FRAME_SIZE[1]
                self.selected_cam.Height = FRAME_SIZE[0]
                self.selected_cam.AcquisitionFrameRateEnable = True
                self.selected_cam.AcquisitionFrameRate = 55

            self.selected_cam.ExposureTime.Value = 100
            self.selected_cam.Gain.Value = 0
            print(f"Camera connected: {self.selected_cam.GetDeviceInfo().GetModelName()} [Ser.-No.: {self.selected_cam.GetDeviceInfo().GetSerialNumber()}]")
            
            # Initialize GPU queues and buffers based on actual resolution
            h = int(self.selected_cam.Height.Value)
            w = int(self.selected_cam.Width.Value)

            self._raw_frame = GPUQueue(max_size=FRAME_QUEUE_SIZE, shape=(h, w), dtype=cp.uint16)
            self._processed_frame = GPUQueue(max_size=FRAME_QUEUE_SIZE, shape=(h, w, 3), dtype=cp.float32)
            self._avg_frame = GPUQueue(max_size=FRAME_QUEUE_SIZE, shape=(h, w, 3), dtype=cp.float32)

            self._rgb_buffer = cp.empty((h, w, 3), dtype=cp.float32)
            self._avg_buffer = cp.empty((h, w, 3), dtype=cp.float32)

            # clamp window to queue capacity
            self._avg_window = max(1, min(self._avg_window, FRAME_QUEUE_SIZE))
            
            # Initialize overlay manager with camera resolution
            self.overlay_manager.set_resolution(w, h)
            
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
        if self.selected_cam is None:
            return
        try:
            self.selected_cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            while not self._stop_event.is_set() and self.selected_cam.IsGrabbing():
                result = self.selected_cam.RetrieveResult(5000)
                try:
                    if result.GrabSucceeded():
                        # Raw to GPU
                        image_array = cp.asarray(result.GetArray())

                        # Demosaic into reusable RGB buffer
                        if self._profiling_enabled:
                            with TimingContext(self.profiler, "demosaic", use_cuda_sync=True):
                                run_demosaic(image_array, self._rgb_buffer)
                        else:
                            run_demosaic(image_array, self._rgb_buffer)

                        # Enqueue processed RGB; overwrite oldest if full
                        if self._processed_frame.put_overwrite(self._rgb_buffer):
                            self.processed_drop_ctr += 1

                        # Compute windowed average over last K frames into reusable buffer
                        try:
                            if self._profiling_enabled:
                                with TimingContext(self.profiler, "averaging", use_cuda_sync=True):
                                    average_gpuqueue_windowed(self._processed_frame, self._avg_buffer, self._avg_window)
                            else:
                                average_gpuqueue_windowed(self._processed_frame, self._avg_buffer, self._avg_window)
                            
                            # Render overlay and composite onto averaged frame
                            self.overlay_manager.render_overlay()
                            composited = self.overlay_manager.composite_onto_frame(self._avg_buffer)
                            
                            # Put composited frame into queue
                            if self._avg_frame.put_overwrite(composited):
                                self.avg_drop_ctr += 1
                        except ValueError:
                            # No frames yet in processed queue
                            pass
                finally:
                    result.Release()
        finally:
            if self.selected_cam.IsGrabbing():
                self.selected_cam.StopGrabbing()

    
    
        
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
            return self._avg_frame.get_view().ravel()

    def set_avg_window(self, k: int):
        self._avg_window = max(1, min(int(k), FRAME_QUEUE_SIZE))

    def get_avg_window(self) -> int:
        return self._avg_window

    def get_profiling_stats(self):
        """Get profiling statistics for demosaic and averaging operations."""
        demosaic_stats = self.profiler.get_stats("demosaic")
        averaging_stats = self.profiler.get_stats("averaging")
        return {
            "demosaic": {
                "avg_ms": demosaic_stats[0],
                "min_ms": demosaic_stats[1],
                "max_ms": demosaic_stats[2],
                "ops_per_sec": demosaic_stats[3],
                "sample_count": demosaic_stats[4],
                "p50_ms": demosaic_stats[5],
                "p95_ms": demosaic_stats[6]
            },
            "averaging": {
                "avg_ms": averaging_stats[0],
                "min_ms": averaging_stats[1],
                "max_ms": averaging_stats[2],
                "ops_per_sec": averaging_stats[3],
                "sample_count": averaging_stats[4],
                "p50_ms": averaging_stats[5],
                "p95_ms": averaging_stats[6]
            }
        }

    def reset_profiling_counters(self):
        """Reset profiling frequency counters."""
        self.profiler.reset_counters()

    def reset_profiling_all(self):
        """Reset all profiling data including timing samples (min/max/avg)."""
        self.profiler.reset_all()

    def set_profiling_enabled(self, enabled: bool):
        """Enable or disable profiling (affects performance slightly when enabled)."""
        self._profiling_enabled = enabled
        if not enabled:
            self.profiler.clear()

    def is_profiling_enabled(self) -> bool:
        return self._profiling_enabled

    def get_overlay_manager(self) -> OverlayManager:
        """Get the overlay manager for adding/removing primitives."""
        return self.overlay_manager
        
