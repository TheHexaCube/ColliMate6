import dearpygui.dearpygui as dpg
import numpy as np
from queue import Empty, Full
import time
import threading
import cupy as cp

from core.cam_manager import CamManager

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

                

    def cam_combo_callback(self, sender, app_data):
        print(f"Sender: {sender}, Camera selected: {app_data}")       

        selected_label = app_data
        self.selected_index = self.cam_info.index(selected_label)            
    
        dpg.configure_item(self.connect_button, enabled=True)    


    def connect_button_callback(self): 
        if self.cam_manager.connect_cam(self.selected_index):
            dpg.configure_item(self.connect_button, label="Disconnect", callback=self.disconnect_button_callback)
            dpg.configure_item(self.exposure_slider, enabled=True, default_value=self.cam_manager.get_exposure_time())
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

        

class VideoStreamWidget: 
    def __init__(self, cam_manager: CamManager):
        self.cam_manager = cam_manager
        self.update_video_texture_stop_event = threading.Event()
        self.update_video_texture_thread = None


        with dpg.texture_registry():
            texture = np.zeros((2048, 1536, 3), dtype=np.float32)
            dpg.add_raw_texture(width=2048, height=1536, format=dpg.mvFormat_Float_rgb, tag="video_texture", default_value=texture)

        with dpg.group():
            dpg.add_text("Framerate: 0.00 fps", tag="framerate_text")
            dpg.add_text("Frame Drop: 0", tag="frame_drop_text")

            with dpg.plot(label="Video Stream", width=600, height=400, equal_aspects=True):
                dpg.add_plot_axis(dpg.mvXAxis, label="Pixels")
                dpg.add_plot_axis(dpg.mvYAxis, label="Pixels", tag="vid_y_axis")
                dpg.add_image_series("video_texture", bounds_min=[0, 0], bounds_max=[2048, 1536], parent="vid_y_axis", tag="video_image_series")



    def update_video_texture(self):
        print("update_video_texture thread started")
        while not self.update_video_texture_stop_event.is_set():
            try:             
                temp_frame = self.cam_manager.get_frame(type="avg")          
                           
                print(f"temp_frame shape: {temp_frame.shape}")
                print(f"frame type: {type(temp_frame)}")
                dpg.set_value("video_texture", cp.asnumpy(temp_frame))
                        #dpg.set_value("framerate_text", f"Proc. to Avg.: {self.cam_manager.get_time_avg(type='avg'):.2f} ms, Raw to Mono: {self.cam_manager.get_time_avg(type='proc'):.2f} ms, Mono to RGB: {self.cam_manager.get_time_avg(type='mono'):.2f} ms")
                        #dpg.set_value("frame_drop_text", f"Raw Drop: {self.cam_manager.get_drop_ctr(type='raw')}, Processed Drop: {self.cam_manager.get_drop_ctr(type='processed')}, Avg Drop: {self.cam_manager.get_drop_ctr(type='avg')}")
                   
              
            except Empty:                 
                time.sleep(0.01)
                continue

            except Exception as e:
                print(f"Error in update_video_texture: {e}")
                continue
                
        print("update_video_texture thread exiting")

    def start_display(self):
        if self.update_video_texture_thread is None or not self.update_video_texture_thread.is_alive():
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
           
    def cleanup(self):
        self.stop_display()
        

    