"""Quick test of ROI Analyzer without full GUI."""

import sys
sys.path.insert(0, 'D:\\EDA\\projects\\hexacube\\autocollimator\\ColliMate6')

import cupy as cp
import time
from core.primitives import Line
from core.roi_analyzer import ROIAnalyzer

# Mock CamManager for testing
class MockCamManager:
    def __init__(self):
        self.frame_counter = 0
        
    def get_resolution(self):
        return 640, 480
    
    def get_frame(self, type="avg"):
        # Generate a synthetic frame with a moving bright spot
        h, w = 480, 640
        frame = cp.zeros((h, w, 3), dtype=cp.float32)
        
        # Moving bright spot
        spot_x = int(320 + 200 * cp.sin(self.frame_counter * 0.1))
        spot_y = 240
        
        y, x = cp.ogrid[:h, :w]
        dist = cp.sqrt((x - spot_x)**2 + (y - spot_y)**2)
        frame[:, :, 0] = cp.exp(-dist**2 / 500)  # Red channel
        
        self.frame_counter += 1
        return frame.ravel()

# Test
print("Testing ROI Analyzer...")

cam_mock = MockCamManager()
analyzer = ROIAnalyzer(cam_mock, update_rate_hz=5.0)

# Create ROI line
w, h = cam_mock.get_resolution()
line = Line(
    start_x=100, start_y=h//2,
    end_x=w-100, end_y=h//2,
    color=(0, 1, 0, 0.7),
    thickness=5
)

# Add to analyzer
analyzer.add_roi("test_line", line)

# Define callback
def on_data(results):
    for name, data in results.items():
        print(f"  {name}: mean={data['mean']:.4f}, peak@{data['peak_position']:3d}, peak_val={data['peak_value']:.4f}")

analyzer.set_callback(on_data)

# Start
print("Starting analyzer (will run for 3 seconds)...")
analyzer.start()

# Let it run
time.sleep(3)

# Stop
print("\nStopping analyzer...")
analyzer.stop()

# Get final results
final = analyzer.get_latest_results("test_line")
print(f"\nFinal results:")
print(f"  Mean: {final['mean']:.4f}")
print(f"  Peak at position: {final['peak_position']}")
print(f"  Peak value: {final['peak_value']:.4f}")

print("\nTest complete!")

