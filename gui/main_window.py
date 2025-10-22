import dearpygui.dearpygui as dpg
from camera.cam_manager import CamManager
import ctypes
import threading
from .widgets import CameraSelectorWidget, VideoStreamWidget, ProfilingWidget
from queue import Empty
import time
from visualization.primitives import Cross, Circle, Line
from analysis.roi_analyzer import ROIAnalyzer
import numpy as np


class MainWindow:
    def __init__(self):
        ctypes.windll.user32.SetProcessDPIAware()
        dpg.create_context()

        self.init_registries()
        self.cam_manager = CamManager(debug_cam=True)
        self.overlay = self.cam_manager.get_overlay_manager()

        # ROI Analyzer for dedicated pixel extraction thread
        self.roi_analyzer = ROIAnalyzer(self.cam_manager, update_rate_hz=10.0)
        self.roi_analyzer.set_callback(self.on_roi_data)  # Optional callback

        # Primitives will be added after camera connects
        self.crosshair = None
        self.roi_line = None

        dpg.create_viewport(title="Main Window", width=1280, height=600)
        dpg.setup_dearpygui()

        # dpg.set_viewport_vsync(False)

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
        with dpg.window(
            label="ColliMate6 Autocollimator", width=1260, height=580, tag="primary"
        ):
            self.cam_selector_widget = CameraSelectorWidget(
                self.cam_manager,
                start_callback=self.start_callback,
                stop_callback=self.stop_callback,
            )

            self.profiling_widget = ProfilingWidget(self.cam_manager)

            self.video_stream_widget = VideoStreamWidget(self.cam_manager)

            # make main window primary

    def key_press_callback(self, sender, app_data):
        if dpg.is_key_down(dpg.mvKey_LControl):
            if dpg.is_key_down(dpg.mvKey_P):
                dpg.show_metrics()
            elif dpg.is_key_down(dpg.mvKey_S):
                dpg.show_style_editor()
            elif dpg.is_key_down(dpg.mvKey_F):
                dpg.show_font_manager()

    def start_callback(self):
        # Add overlay primitives after camera is connected
        if self.crosshair is None:
            w, h = self.cam_manager.get_resolution()
            cx, cy = w // 2, h // 2

            # Create crosshair at center
            self.crosshair = Cross(
                center_x=cx,
                center_y=cy,
                size=200,
                color=(1.0, 0.0, 0.0, 0.6),  # Red, 80% opaque
                thickness=3,
                rotation=(np.pi / 4),
            )
            self.overlay.add_primitive(self.crosshair)

            # Add a circle around it for visibility
            circle = Circle(
                center_x=cx,
                center_y=cy,
                radius=150,
                color=(1.0, 0.0, 0.0, 0.6),  # Red, 60% opaque
                thickness=3,
                filled=False,
            )
            self.overlay.add_primitive(circle)

            # Create ROI line for analysis
            self.roi_line = Line(
                start_x=100,
                start_y=cy,
                end_x=w - 100,
                end_y=cy,
                color=(0.0, 1.0, 0.0, 0.6),  # Green, 60% opaque
                thickness=5,
            )
            self.overlay.add_primitive(self.roi_line)

            # Add ROI to analyzer
            self.roi_analyzer.add_roi("horizontal_line", self.roi_line)

            print(f"Added {self.overlay.get_primitive_count()} overlay primitives")

        self.video_stream_widget.start_display()
        self.profiling_widget.start_display()
        self.roi_analyzer.start()  # Start ROI analysis thread

    def stop_callback(self):
        self.video_stream_widget.stop_display()
        self.profiling_widget.stop_display()
        self.roi_analyzer.stop()  # Stop ROI analysis thread

    def run(self):
        dpg.show_viewport()

        while dpg.is_dearpygui_running():
            dpg.render_dearpygui_frame()

        self.cleanup()

    def on_roi_data(self, results):
        """
        Callback function called by ROI analyzer with latest results.
        This runs in the ROI analyzer thread.
        """
        # Example: Print statistics for each ROI
        for roi_name, data in results.items():
            print(
                f"{roi_name}: mean={data['mean']:.4f}, peak@{data['peak_position']}, peak_val={data['peak_value']:.4f}"
            )

        # You can also:
        # - Log data to file
        # - Update GUI elements (use dpg.set_value())
        # - Trigger actions based on thresholds
        # - Store data for further analysis

    def cleanup(self):
        self.roi_analyzer.stop()

        self.cam_manager.stop_capture()
        self.cam_manager.disconnect_cam()

        self.video_stream_widget.stop_display()
        self.profiling_widget.cleanup()

        dpg.destroy_context()
