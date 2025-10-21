"""
Practical example: Using Line primitive as ROI with camera frames.

This demonstrates how to:
1. Create a Line ROI and add it to the overlay
2. Extract pixel values from camera frames
3. Perform real-time analysis
4. Update line position dynamically
"""

import sys
sys.path.insert(0, 'D:\\EDA\\projects\\hexacube\\autocollimator\\ColliMate6')

import cupy as cp
import numpy as np
from core.primitives import Line

# Simulating camera frame (replace with actual cam_manager.get_frame())
def get_simulated_frame(width=640, height=480):
    """Simulate a camera frame for demonstration."""
    frame = cp.zeros((height, width, 3), dtype=cp.float32)
    
    # Simulate some interesting features
    # 1. Bright spot in center
    cy, cx = height // 2, width // 2
    y, x = cp.ogrid[:height, :width]
    dist = cp.sqrt((x - cx)**2 + (y - cy)**2)
    frame[:, :, 0] = cp.exp(-dist**2 / 5000)  # Red bright spot
    frame[:, :, 1] = cp.exp(-dist**2 / 5000) * 0.5  # Green
    
    # 2. Gradient background
    frame[:, :, 2] = cp.linspace(0, 0.3, width)[None, :]  # Blue gradient
    
    return frame


# Example 1: Basic ROI extraction
print("=" * 70)
print("Example 1: Basic Line ROI for Intensity Profile")
print("=" * 70)

# Create a horizontal line through center
line1 = Line(start_x=50, start_y=240, end_x=590, end_y=240,
             color=(1, 0, 0, 0.7),  # Semi-transparent red
             thickness=1)

# Get frame and extract pixels
frame = get_simulated_frame()
pixels = line1.extract_pixels(frame)

print(f"Line: ({line1.start_x}, {line1.start_y}) -> ({line1.end_x}, {line1.end_y})")
print(f"Extracted {pixels.shape[0]} samples with {pixels.shape[1]} channels")
print(f"Red channel:   min={cp.min(pixels[:, 0]):.4f}, max={cp.max(pixels[:, 0]):.4f}")
print(f"Green channel: min={cp.min(pixels[:, 1]):.4f}, max={cp.max(pixels[:, 1]):.4f}")
print(f"Blue channel:  min={cp.min(pixels[:, 2]):.4f}, max={cp.max(pixels[:, 2]):.4f}")

# Find peak position
intensity = cp.mean(pixels, axis=1)  # Average across RGB channels
peak_pos = cp.argmax(intensity)
peak_value = intensity[peak_pos]
print(f"\nPeak intensity: {peak_value:.4f} at sample {peak_pos}")


# Example 2: Thick line for noise reduction
print("\n" + "=" * 70)
print("Example 2: Thick Line ROI with Averaging for Noise Reduction")
print("=" * 70)

line2 = Line(start_x=50, start_y=240, end_x=590, end_y=240,
             color=(0, 1, 0, 0.7),  # Green
             thickness=11)  # Thicker line

# Extract with thickness
roi = line2.extract_pixels_with_thickness(frame)
print(f"ROI shape: {roi.shape} (samples × thickness × channels)")

# Average across thickness to reduce noise
profile_averaged = cp.mean(roi, axis=1)  # Shape: (N, 3)
print(f"Averaged profile shape: {profile_averaged.shape}")

# Compare noise (simulate by looking at std across thickness)
std_across_thickness = cp.std(roi, axis=1)  # Variation across thickness
mean_std = cp.mean(std_across_thickness)
print(f"Average std across thickness: {mean_std:.6f}")
print(f"  (Lower = more consistent, better noise reduction)")


# Example 3: Multiple ROIs for comparison
print("\n" + "=" * 70)
print("Example 3: Multiple ROIs for Comparative Analysis")
print("=" * 70)

# Top, middle, bottom horizontal lines
lines = [
    Line(100, 120, 540, 120, (1, 0, 0, 0.5), thickness=3),  # Top
    Line(100, 240, 540, 240, (0, 1, 0, 0.5), thickness=3),  # Middle
    Line(100, 360, 540, 360, (0, 0, 1, 0.5), thickness=3),  # Bottom
]

print("Comparing intensity profiles at different Y positions:")
for i, line in enumerate(lines):
    pixels = line.extract_pixels(frame)
    mean_intensity = cp.mean(pixels)
    max_intensity = cp.max(pixels)
    print(f"  Line {i+1} (y={int(line.start_y)}): mean={mean_intensity:.4f}, max={max_intensity:.4f}")


# Example 4: Dynamic ROI - simulate scanning
print("\n" + "=" * 70)
print("Example 4: Dynamic ROI - Scanning Across Image")
print("=" * 70)

line_scanner = Line(320, 0, 320, 479, (1, 1, 0, 0.7), thickness=1)

print("Scanning vertical line across image (every 100 pixels)...")
scan_results = []

for x_pos in range(100, 541, 100):
    # Update line position
    line_scanner.start_x = x_pos
    line_scanner.end_x = x_pos
    
    # Extract and analyze
    pixels = line_scanner.extract_pixels(frame)
    mean_val = cp.mean(pixels)
    max_val = cp.max(pixels)
    
    scan_results.append((x_pos, float(mean_val), float(max_val)))
    print(f"  x={x_pos:3d}: mean={mean_val:.4f}, max={max_val:.4f}")

# Find optimal position (highest max value)
optimal_x = max(scan_results, key=lambda x: x[2])
print(f"\nOptimal X position: {optimal_x[0]} (max intensity: {optimal_x[2]:.4f})")


# Example 5: Real-time integration pattern
print("\n" + "=" * 70)
print("Example 5: Real-Time Integration Pattern (Pseudocode)")
print("=" * 70)

example_code = """
# In your actual application with camera manager:

from core.primitives import Line
import cupy as cp

# Setup
cam_manager = CamManager()
overlay_mgr = cam_manager.get_overlay_manager()

# Create ROI line
line = Line(x1=100, y1=240, x2=540, y2=240, 
            color=(1, 0, 0, 0.7), thickness=5)
overlay_mgr.add_primitive(line)  # Visible in GUI

# In your processing loop
def process_frame():
    # Get averaged frame (already on GPU as 1D array)
    frame_1d = cam_manager.get_frame(type="avg")
    
    # Reshape to 3D for ROI extraction
    h, w = cam_manager.get_resolution()
    frame_3d = frame_1d.reshape((h, w, 3))
    
    # Extract ROI pixels
    roi_pixels = line.extract_pixels_with_thickness(frame_3d)
    
    # Analysis (all on GPU)
    profile = cp.mean(roi_pixels, axis=1)  # Average across thickness
    mean_intensity = cp.mean(profile)
    peak_pos = cp.argmax(cp.mean(profile, axis=1))
    
    # Optional: transfer to CPU for logging/display
    stats = {
        'mean': float(mean_intensity),
        'peak_position': int(peak_pos),
        'profile': cp.asnumpy(profile)  # Only if needed on CPU
    }
    
    return stats

# Dynamic ROI adjustment based on analysis
def auto_adjust_roi(stats):
    if stats['mean'] < 0.5:
        # Low signal - increase line thickness
        line.thickness = min(line.thickness + 2, 21)
    elif stats['peak_position'] < 100:
        # Peak is off-center, shift line
        line.start_x -= 10
        line.end_x -= 10
"""

print(example_code)


# Example 6: Statistical monitoring
print("\n" + "=" * 70)
print("Example 6: Statistical Monitoring Over Time")
print("=" * 70)

line_monitor = Line(200, 240, 440, 240, (0, 1, 1, 0.7), thickness=7)

print("Simulating 5 frames of monitoring...")
history = []

for frame_num in range(5):
    # Get frame (in practice, this would be a new camera frame each iteration)
    frame = get_simulated_frame()
    
    # Add some random noise to simulate temporal variation
    noise = cp.random.randn(*frame.shape) * 0.01
    frame = cp.clip(frame + noise, 0, 1)
    
    # Extract and analyze
    roi = line_monitor.extract_pixels_with_thickness(frame)
    
    stats = {
        'frame': frame_num,
        'mean': float(cp.mean(roi)),
        'std': float(cp.std(roi)),
        'min': float(cp.min(roi)),
        'max': float(cp.max(roi)),
    }
    
    history.append(stats)
    print(f"  Frame {frame_num}: mean={stats['mean']:.4f}, "
          f"std={stats['std']:.4f}, min={stats['min']:.4f}, max={stats['max']:.4f}")

# Calculate statistics over time
mean_over_time = [s['mean'] for s in history]
temporal_std = float(np.std(mean_over_time))
print(f"\nTemporal variation (std of means): {temporal_std:.6f}")


print("\n" + "=" * 70)
print("Summary: Line ROI Features")
print("=" * 70)
print("""
✓ GPU-accelerated pixel extraction
✓ Center-line or full-thickness sampling
✓ Real-time analysis capabilities
✓ Dynamic ROI repositioning
✓ Multiple simultaneous ROIs
✓ Statistical analysis (mean, std, min, max, profiles)
✓ Noise reduction via thickness averaging
✓ Zero CPU-GPU transfer when frame already on GPU
✓ Integration with overlay system for visual feedback
""")

