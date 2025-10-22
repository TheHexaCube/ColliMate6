"""
ROI Analyzer - Dedicated thread for extracting and analyzing pixel data from Line ROIs.
"""

import threading
import time
import cupy as cp
from typing import Optional, Callable, Dict, Any
from visualization.primitives import Line


class ROIAnalyzer:
    """
    Manages real-time ROI pixel extraction and analysis in a separate thread.
    """

    def __init__(self, cam_manager, update_rate_hz: float = 10.0):
        """
        Args:
            cam_manager: CamManager instance to get frames from
            update_rate_hz: How often to extract and analyze ROI data (Hz)
        """
        self.cam_manager = cam_manager
        self.update_interval = 1.0 / update_rate_hz

        self.roi_lines = []  # List of Line primitives to analyze
        self.analysis_thread = None
        self.stop_event = threading.Event()

        # Storage for latest results
        self._latest_results = {}
        self._results_lock = threading.Lock()

        # Optional callback for real-time notifications
        self.callback = None

    def add_roi(self, name: str, line: Line):
        """Add a Line ROI to analyze."""
        self.roi_lines.append((name, line))

    def remove_roi(self, name: str):
        """Remove a Line ROI by name."""
        self.roi_lines = [(n, l) for n, l in self.roi_lines if n != name]

    def clear_rois(self):
        """Remove all ROIs."""
        self.roi_lines.clear()

    def set_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """
        Set callback function to be called with results each iteration.
        Callback receives: dict with keys = ROI names, values = analysis results
        """
        self.callback = callback

    def _analysis_loop(self):
        """Main analysis loop running in separate thread."""
        while not self.stop_event.is_set():
            try:
                # Get averaged frame from camera
                frame_1d = self.cam_manager.get_frame(type="avg")
                if frame_1d is None:
                    time.sleep(self.update_interval)
                    continue

                # Reshape to 3D
                w, h = self.cam_manager.get_resolution()
                frame_3d = frame_1d.reshape((h, w, 3))

                # Analyze each ROI
                results = {}
                for name, line in self.roi_lines:
                    if not line.visible:
                        continue

                    # Extract pixels with thickness
                    roi_pixels = line.extract_pixels_with_thickness(frame_3d)

                    # Calculate statistics
                    profile = cp.mean(roi_pixels, axis=1)  # Average across thickness
                    intensity_1d = cp.mean(
                        profile, axis=1
                    )  # Average across RGB channels

                    results[name] = {
                        "roi_pixels": roi_pixels,  # Full ROI data (on GPU)
                        "profile": profile,  # Profile along line (N, 3)
                        "intensity": intensity_1d,  # 1D intensity profile (N,)
                        "mean": float(cp.mean(roi_pixels)),
                        "std": float(cp.std(roi_pixels)),
                        "min": float(cp.min(roi_pixels)),
                        "max": float(cp.max(roi_pixels)),
                        "peak_position": int(cp.argmax(intensity_1d)),
                        "peak_value": float(cp.max(intensity_1d)),
                    }

                # Store results
                with self._results_lock:
                    self._latest_results = results

                # Call callback if set
                if self.callback:
                    self.callback(results)

                time.sleep(self.update_interval)

            except Exception as e:
                print(f"ROI analysis error: {e}")
                time.sleep(self.update_interval)

    def get_latest_results(self, roi_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get latest analysis results.

        Args:
            roi_name: If specified, return results for that ROI only.
                     If None, return all results.

        Returns:
            Dictionary of results
        """
        with self._results_lock:
            if roi_name:
                return self._latest_results.get(roi_name, {})
            return self._latest_results.copy()

    def start(self):
        """Start the analysis thread."""
        if self.analysis_thread and self.analysis_thread.is_alive():
            print("ROI analyzer already running")
            return

        self.stop_event.clear()
        self.analysis_thread = threading.Thread(target=self._analysis_loop, daemon=True)
        self.analysis_thread.start()
        print(f"ROI analyzer started (update rate: {1/self.update_interval:.1f} Hz)")

    def stop(self):
        """Stop the analysis thread."""
        self.stop_event.set()
        if self.analysis_thread and self.analysis_thread.is_alive():
            self.analysis_thread.join(timeout=1.0)
            if self.analysis_thread.is_alive():
                print("Warning: ROI analyzer thread did not exit cleanly")
        print("ROI analyzer stopped")

    def is_running(self) -> bool:
        """Check if analysis thread is running."""
        return self.analysis_thread is not None and self.analysis_thread.is_alive()
