import dearpygui.dearpygui as dpg
from core.cam_manager import CamManager
import ctypes
import threading
from .widgets import CameraSelectorWidget, VideoStreamWidget
from queue import Empty
import time

class MainWindow: 
    def __init__(self):
        ctypes.windll.user32.SetProcessDPIAware()
        dpg.create_context()

        self.init_registries()
        self.cam_manager = CamManager(debug_cam=True)    


        dpg.create_viewport(title="Main Window", width=1280, height=600)
        dpg.setup_dearpygui()

        #dpg.set_viewport_vsync(False)



        
        self.create_main_window()
        dpg.set_primary_window("primary", True)

        





        
    def init_registries(self):

        with dpg.handler_registry():
            dpg.add_key_press_handler(callback=self.key_press_callback)
        
        with dpg.font_registry():
            font_figtree = dpg.add_font("assets\\fonts\\Figtree-Regular.ttf", 16)
            font_clearsans = dpg.add_font("assets\\fonts\\ClearSans-Regular.ttf", 16)

        dpg.bind_font(font_clearsans)

        with dpg.theme() as disabled_theme:
            with dpg.theme_component(dpg.mvButton, enabled_state=False):
                dpg.add_theme_color(dpg.mvThemeCol_Text, [120, 120, 120, 255])
                dpg.add_theme_color(dpg.mvThemeCol_Button, [51, 51, 51, 255])
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, [51, 51, 51, 255])
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, [51, 51, 51, 255])
        
        dpg.bind_theme(disabled_theme)




    def create_main_window(self):
        # Create the main window first
        with dpg.window(label="ColliMate6 Autocollimator", width=1260, height=580, tag="primary"):
            self.cam_selector_widget = CameraSelectorWidget(self.cam_manager, start_callback=self.start_callback, stop_callback=self.stop_callback)

            self.video_stream_widget = VideoStreamWidget(self.cam_manager)

            # make main window primary
        

        

    def key_press_callback(self, sender, app_data):
        if (dpg.is_key_down(dpg.mvKey_LControl)):
            if (dpg.is_key_down(dpg.mvKey_P)):
                dpg.show_metrics()
            elif (dpg.is_key_down(dpg.mvKey_S)):
                dpg.show_style_editor()
            elif (dpg.is_key_down(dpg.mvKey_F)):
                dpg.show_font_manager()


    def start_callback(self):        
        self.video_stream_widget.start_display()

    def stop_callback(self):
        self.video_stream_widget.stop_display()

    def run(self):
        dpg.show_viewport()

        while dpg.is_dearpygui_running():
            dpg.render_dearpygui_frame()
        
        self.cleanup()

    def cleanup(self):
       
        self.cam_manager.stop_capture()
        self.cam_manager.disconnect_cam()

        self.video_stream_widget.stop_display()

        dpg.destroy_context()
       