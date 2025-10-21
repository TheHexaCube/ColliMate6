"""Test Line primitive as Region of Interest for pixel extraction."""

import sys
sys.path.insert(0, 'D:\\EDA\\projects\\hexacube\\autocollimator\\ColliMate6')

import cupy as cp
import numpy as np
from core.primitives import Line

# Create a test image with a gradient
h, w = 480, 640
test_image = cp.zeros((h, w, 3), dtype=cp.float32)

# Create horizontal gradient (left=0, right=1)
for x in range(w):
    test_image[:, x, 0] = x / w  # Red channel

# Create vertical gradient (top=0, bottom=1)  
for y in range(h):
    test_image[y, :, 1] = y / h  # Green channel

# Blue channel: diagonal pattern
test_image[:, :, 2] = 0.5

print("Test Image created: shape =", test_image.shape)
print(f"  Red: horizontal gradient (0 to 1)")
print(f"  Green: vertical gradient (0 to 1)")
print(f"  Blue: constant 0.5\n")

# Test 1: Horizontal line - should show red gradient
print("=" * 60)
print("Test 1: Horizontal line across center")
line_h = Line(start_x=50, start_y=240, end_x=590, end_y=240, 
              color=(1, 0, 0, 1), thickness=1)

pixels_h = line_h.extract_pixels(test_image)
print(f"  Extracted pixels shape: {pixels_h.shape}")
print(f"  Red channel:   min={cp.min(pixels_h[:, 0]):.3f}, max={cp.max(pixels_h[:, 0]):.3f}")
print(f"  Green channel: min={cp.min(pixels_h[:, 1]):.3f}, max={cp.max(pixels_h[:, 1]):.3f}")
print(f"  Blue channel:  min={cp.min(pixels_h[:, 2]):.3f}, max={cp.max(pixels_h[:, 2]):.3f}")
print(f"  Expected: Red varies 0->1, Green ~0.5, Blue ~0.5")

# Test 2: Vertical line - should show green gradient
print("\n" + "=" * 60)
print("Test 2: Vertical line down center")
line_v = Line(start_x=320, start_y=50, end_x=320, end_y=430,
              color=(0, 1, 0, 1), thickness=1)

pixels_v = line_v.extract_pixels(test_image)
print(f"  Extracted pixels shape: {pixels_v.shape}")
print(f"  Red channel:   min={cp.min(pixels_v[:, 0]):.3f}, max={cp.max(pixels_v[:, 0]):.3f}")
print(f"  Green channel: min={cp.min(pixels_v[:, 1]):.3f}, max={cp.max(pixels_v[:, 1]):.3f}")
print(f"  Blue channel:  min={cp.min(pixels_v[:, 2]):.3f}, max={cp.max(pixels_v[:, 2]):.3f}")
print(f"  Expected: Red ~0.5, Green varies 0->1, Blue ~0.5")

# Test 3: Diagonal line - should show both gradients
print("\n" + "=" * 60)
print("Test 3: Diagonal line from top-left to bottom-right")
line_d = Line(start_x=100, start_y=100, end_x=540, end_y=380,
              color=(0, 0, 1, 1), thickness=1)

pixels_d = line_d.extract_pixels(test_image)
print(f"  Extracted pixels shape: {pixels_d.shape}")
print(f"  Red channel:   min={cp.min(pixels_d[:, 0]):.3f}, max={cp.max(pixels_d[:, 0]):.3f}")
print(f"  Green channel: min={cp.min(pixels_d[:, 1]):.3f}, max={cp.max(pixels_d[:, 1]):.3f}")
print(f"  Blue channel:  min={cp.min(pixels_d[:, 2]):.3f}, max={cp.max(pixels_d[:, 2]):.3f}")
print(f"  Expected: Both Red and Green vary (diagonal through gradients)")

# Test 4: Line with thickness - extract full ROI
print("\n" + "=" * 60)
print("Test 4: Horizontal line WITH thickness=5")
line_thick = Line(start_x=50, start_y=240, end_x=590, end_y=240,
                  color=(1, 1, 0, 1), thickness=5)

pixels_thick = line_thick.extract_pixels_with_thickness(test_image)
print(f"  Extracted pixels shape: {pixels_thick.shape}")
print(f"  Shape breakdown: ({pixels_thick.shape[0]} samples along line, "
      f"{pixels_thick.shape[1]} pixels across thickness, {pixels_thick.shape[2]} channels)")

# Average across thickness to get 1D profile
profile = cp.mean(pixels_thick, axis=1)  # Average across thickness
print(f"\n  Profile along line (averaged across thickness):")
print(f"    Shape: {profile.shape}")
print(f"    Red:   min={cp.min(profile[:, 0]):.3f}, max={cp.max(profile[:, 0]):.3f}")
print(f"    Green: min={cp.min(profile[:, 1]):.3f}, max={cp.max(profile[:, 1]):.3f}")

# Test 5: Statistical analysis of ROI
print("\n" + "=" * 60)
print("Test 5: Statistical analysis of ROI")
mean_rgb = cp.mean(pixels_thick, axis=(0, 1))  # Mean across all pixels in ROI
std_rgb = cp.std(pixels_thick, axis=(0, 1))    # Std dev across all pixels in ROI

print(f"  Mean RGB values in ROI: {mean_rgb}")
print(f"  Std dev RGB in ROI: {std_rgb}")

# Test 6: Custom sampling (fewer samples)
print("\n" + "=" * 60)
print("Test 6: Custom sampling - only 10 samples along line")
pixels_sparse = line_h.extract_pixels(test_image, num_samples=10)
print(f"  Extracted pixels shape: {pixels_sparse.shape}")
print(f"  Red values at 10 samples: {pixels_sparse[:, 0]}")

print("\n" + "=" * 60)
print("✓ All tests completed successfully!")
print("\nUsage examples:")
print("  # Simple center-line extraction:")
print("  pixels = line.extract_pixels(frame)")
print("  mean_intensity = cp.mean(pixels, axis=0)")
print()
print("  # Full ROI with thickness:")
print("  roi = line.extract_pixels_with_thickness(frame)")
print("  profile = cp.mean(roi, axis=1)  # Avg across thickness")
print("  total_avg = cp.mean(roi)  # Overall average")

