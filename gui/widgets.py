import dearpygui.dearpygui as dpg
import numpy as np
from queue import Empty, Full
import time
import threading
import cupy as cp

from camera.cam_manager import CamManager

class CameraSelectorWidget:
    def __init__(self, cam_manager, start_callback=None, stop_callback=None):
        self.cam_manager = cam_manager
        self.available_cams = self.cam_manager.get_available_cams()
        self.selected_index = None
        self.start_callback = start_callback
        self.stop_callback = stop_callback


        # Create camera selector section
        with dpg.group(horizontal=True):              
            dpg.add_text("Select Camera:")
            self.cam_info = [f"{i}: {model} [Ser.-No.: {serial}]" for i, model, serial in self.available_cams]
            self.cam_combo = dpg.add_combo(self.cam_info, callback=self.cam_combo_callback, width=400)

            # Create connect button
            self.connect_button = dpg.add_button(label="Connect", callback=self.connect_button_callback, enabled=False)

        self.exposure_slider = dpg.add_slider_int(label="Exposure Time [µs]", width=200,min_value=27, max_value=1000, default_value=105, callback=self.exposure_time_callback)
        self.avg_window_slider = dpg.add_slider_int(label="Average Window (frames)", width=200, min_value=1, max_value=8, default_value=8, callback=self.avg_window_callback)

                

    def cam_combo_callback(self, sender, app_data):
        print(f"Sender: {sender}, Camera selected: {app_data}")       

        selected_label = app_data
        self.selected_index = self.cam_info.index(selected_label)            
    
        dpg.configure_item(self.connect_button, enabled=True)    


    def connect_button_callback(self): 
        if self.cam_manager.connect_cam(self.selected_index):
            dpg.configure_item(self.connect_button, label="Disconnect", callback=self.disconnect_button_callback)
            dpg.configure_item(self.exposure_slider, enabled=True, default_value=self.cam_manager.get_exposure_time())
            # clamp and reflect avg window to slider max (queue size)
            dpg.configure_item(self.avg_window_slider, enabled=True, default_value=self.cam_manager.get_avg_window())
            self.cam_manager.start_capture()
            if self.start_callback is not None:
                self.start_callback()
        else:
            dpg.configure_item(self.connect_button, label="Connect", callback=self.connect_button_callback)
            print("Failed to connect camera")

    def disconnect_button_callback(self):
        if self.cam_manager.disconnect_cam():
            dpg.configure_item(self.connect_button, label="Connect", callback=self.connect_button_callback)
            if self.stop_callback is not None:
                self.stop_callback()
        else:
            dpg.configure_item(self.connect_button, label="Disconnect", callback=self.disconnect_button_callback)
            print("Failed to disconnect camera")

    def exposure_time_callback(self, sender, app_data):
        print(f"Sender: {sender}, Exposure time: {app_data}")
        self.cam_manager.set_exposure_time(app_data)

    def avg_window_callback(self, sender, app_data):
        self.cam_manager.set_avg_window(app_data)

        

class VideoStreamWidget: 
    def __init__(self, cam_manager: CamManager):
        self.cam_manager = cam_manager
        self.update_video_texture_stop_event = threading.Event()
        self.update_video_texture_thread = None


        with dpg.plot(label="Video Stream", width=600, height=400, equal_aspects=True):
            dpg.add_plot_axis(dpg.mvXAxis, label="Pixels")
            dpg.add_plot_axis(dpg.mvYAxis, label="Pixels", tag="vid_y_axis")
            # image series will be added in start_display after texture creation



    def update_video_texture(self):
        print("update_video_texture thread started")
        
        while not self.update_video_texture_stop_event.is_set():
            try:             
                # Get frame (already composited with overlay in cam_manager)
                temp_frame = self.cam_manager.get_frame(type="avg")
                
                # Display frame directly (already 1D raveled)
                dpg.set_value("video_texture", cp.asnumpy(temp_frame))
              
            except Empty:                 
                time.sleep(0.01)
                continue

            except Exception as e:
                print(f"Error in update_video_texture: {e}")
                continue
                
        print("update_video_texture thread exiting")

    def start_display(self):
        if self.update_video_texture_thread is None or not self.update_video_texture_thread.is_alive():
            # Create or recreate texture and image series based on current camera resolution
            w, h = self.cam_manager.get_resolution()
            with dpg.texture_registry():
                texture = np.zeros((h, w, 3), dtype=np.float32)
                if dpg.does_item_exist("video_texture"):
                    dpg.delete_item("video_texture")
                dpg.add_raw_texture(width=w, height=h, format=dpg.mvFormat_Float_rgb, tag="video_texture", default_value=texture)

            if dpg.does_item_exist("video_image_series"):
                dpg.delete_item("video_image_series")
            dpg.add_image_series("video_texture", bounds_min=[0, 0], bounds_max=[w, h], parent="vid_y_axis", tag="video_image_series")

            self.update_video_texture_thread = threading.Thread(target=self.update_video_texture, daemon=True)
            self.update_video_texture_stop_event.clear()
            self.update_video_texture_thread.start()

    def stop_display(self):
        print("Stopping video display...")
        self.update_video_texture_stop_event.set()
        if self.update_video_texture_thread is not None and self.update_video_texture_thread.is_alive():
            print("Waiting for update_video_texture thread to join...")
            self.update_video_texture_thread.join()
            print("update_video_texture thread joined successfully")
        
        # Clean up DearPyGui resources
        if dpg.does_item_exist("video_image_series"):
            dpg.delete_item("video_image_series")
        if dpg.does_item_exist("video_texture"):
            dpg.delete_item("video_texture")
           
    def cleanup(self):
        self.stop_display()


class ProfilingWidget:
    """Widget to display performance profiling information."""
    
    # Control profiling display at module level
    PROFILING_DISPLAY_ENABLED = True  # Set to False to hide profiling section
    
    def __init__(self, cam_manager: CamManager):
        self.cam_manager = cam_manager
        self.update_stats_stop_event = threading.Event()
        self.update_stats_thread = None
        
        if not self.PROFILING_DISPLAY_ENABLED:
            return
        
        with dpg.collapsing_header(label="Performance Profiling", default_open=True):
            with dpg.group(horizontal=True):
                dpg.add_text("Profiling:")
                self.profiling_toggle = dpg.add_checkbox(
                    label="Enabled", 
                    default_value=self.cam_manager.is_profiling_enabled(),
                    callback=self.toggle_profiling_callback
                )
                dpg.add_button(label="Reset Counters", callback=self.reset_counters_callback)
            
            dpg.add_separator()
            
            # Demosaic stats
            with dpg.group(horizontal=True):
                dpg.add_text("Demosaic:", color=(150, 200, 255))
                with dpg.group(horizontal=False):
                    dpg.add_text("Avg: 0.00 ms | P50: 0.00 ms | P95: 0.00 ms", tag="prof_demosaic_avg")
                    dpg.add_text("Min: 0.00 ms | Max: 0.00 ms", tag="prof_demosaic_minmax")
                    dpg.add_text("Rate: 0.0 ops/s | Samples: 0", tag="prof_demosaic_rate")
                
                
                
                # Averaging stats
                dpg.add_text("Averaging:", color=(150, 255, 200))
                with dpg.group(horizontal=False):
                    dpg.add_text("Avg: 0.00 ms | P50: 0.00 ms | P95: 0.00 ms", tag="prof_averaging_avg")
                    dpg.add_text("Min: 0.00 ms | Max: 0.00 ms", tag="prof_averaging_minmax")
                    dpg.add_text("Rate: 0.0 ops/s | Samples: 0", tag="prof_averaging_rate")
    
    def toggle_profiling_callback(self, sender, app_data):
        self.cam_manager.set_profiling_enabled(app_data)
    
    def reset_counters_callback(self):
        self.cam_manager.reset_profiling_all()
    
    def update_stats_loop(self):
        """Background thread to update profiling statistics display."""
        while not self.update_stats_stop_event.is_set():
            try:
                if self.cam_manager.is_profiling_enabled():
                    stats = self.cam_manager.get_profiling_stats()
                    
                    # Update demosaic stats
                    dpg.set_value("prof_demosaic_avg", 
                                  f"Avg: {stats['demosaic']['avg_ms']:.3f} ms | P50: {stats['demosaic']['p50_ms']:.3f} ms | P95: {stats['demosaic']['p95_ms']:.3f} ms")
                    dpg.set_value("prof_demosaic_minmax", 
                                  f"Min: {stats['demosaic']['min_ms']:.3f} ms | Max: {stats['demosaic']['max_ms']:.3f} ms")
                    dpg.set_value("prof_demosaic_rate", 
                                  f"Rate: {stats['demosaic']['ops_per_sec']:.1f} ops/s | Samples: {stats['demosaic']['sample_count']}")
                    
                    # Update averaging stats
                    dpg.set_value("prof_averaging_avg", 
                                  f"Avg: {stats['averaging']['avg_ms']:.3f} ms | P50: {stats['averaging']['p50_ms']:.3f} ms | P95: {stats['averaging']['p95_ms']:.3f} ms")
                    dpg.set_value("prof_averaging_minmax", 
                                  f"Min: {stats['averaging']['min_ms']:.3f} ms | Max: {stats['averaging']['max_ms']:.3f} ms")
                    dpg.set_value("prof_averaging_rate", 
                                  f"Rate: {stats['averaging']['ops_per_sec']:.1f} ops/s | Samples: {stats['averaging']['sample_count']}")
                
                time.sleep(0.1)  # Update at 10 Hz
            except Exception as e:
                print(f"Error updating profiling stats: {e}")
                time.sleep(0.1)
    
    def start_display(self):
        if not self.PROFILING_DISPLAY_ENABLED:
            return
        
        if self.update_stats_thread is None or not self.update_stats_thread.is_alive():
            self.update_stats_thread = threading.Thread(target=self.update_stats_loop, daemon=True)
            self.update_stats_stop_event.clear()
            self.update_stats_thread.start()
    
    def stop_display(self):
        if not self.PROFILING_DISPLAY_ENABLED:
            return
        
        self.update_stats_stop_event.set()
        if self.update_stats_thread is not None and self.update_stats_thread.is_alive():
            self.update_stats_thread.join(timeout=1.0)
    
    def cleanup(self):
        self.stop_display()
    