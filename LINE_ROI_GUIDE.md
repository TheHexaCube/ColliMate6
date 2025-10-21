# Line Primitive as Region of Interest (ROI)

The `Line` primitive now supports extracting pixel values from underneath the line, allowing it to act as a configurable Region of Interest for analysis.

## Features

### 1. `extract_pixels()` - Center-line extraction
Extracts pixel values along the center of the line.

**Signature:**
```python
def extract_pixels(source_frame: cp.ndarray, num_samples: int = None) -> cp.ndarray
```

**Returns:** CuPy array of shape `(N, C)` where:
- `N` = number of samples along the line
- `C` = number of channels (e.g., 3 for RGB)

**Example:**
```python
# Create a line ROI
line = Line(start_x=100, start_y=200, end_x=500, end_y=200, 
            color=(1, 0, 0, 1), thickness=3)

# Extract pixels from camera frame
pixels = line.extract_pixels(rgb_frame)  # Shape: (400, 3)

# Analyze the data
mean_intensity = cp.mean(pixels, axis=0)  # Average RGB along line
max_red = cp.max(pixels[:, 0])            # Maximum red value
profile = pixels[:, 0]                     # Red channel intensity profile
```

### 2. `extract_pixels_with_thickness()` - Full ROI extraction
Extracts ALL pixels within the line's thickness by sampling perpendicular to the line direction.

**Signature:**
```python
def extract_pixels_with_thickness(source_frame: cp.ndarray, num_samples: int = None) -> cp.ndarray
```

**Returns:** CuPy array of shape `(N, T, C)` where:
- `N` = number of samples along the line
- `T` = thickness of the line
- `C` = number of channels

**Example:**
```python
# Create a thick line ROI
line = Line(start_x=100, start_y=200, end_x=500, end_y=200,
            color=(1, 0, 0, 1), thickness=5)

# Extract full ROI including thickness
roi = line.extract_pixels_with_thickness(rgb_frame)  # Shape: (400, 5, 3)

# Analyze the data
profile = cp.mean(roi, axis=1)          # Avg across thickness: (400, 3)
total_avg = cp.mean(roi, axis=(0, 1))   # Overall RGB average: (3,)
std_dev = cp.std(roi, axis=(0, 1))      # Standard deviation: (3,)

# Get intensity profile along line (average across all channels and thickness)
intensity_profile = cp.mean(roi, axis=(1, 2))  # Shape: (400,)
```

## Use Cases

### 1. Edge Detection / Transition Analysis
```python
# Horizontal line across expected edge
line = Line(x1=0, y1=240, x2=639, y2=240, color=(1,1,0,1), thickness=1)
pixels = line.extract_pixels(frame)

# Find edge location
gradient = cp.diff(pixels[:, 0])  # Derivative of red channel
edge_location = cp.argmax(cp.abs(gradient))
```

### 2. Beam Profile Analysis
```python
# Vertical line through laser beam
line = Line(x1=320, y1=0, x2=320, y2=479, color=(0,1,0,1), thickness=1)
pixels = line.extract_pixels(frame)

# Get beam profile
intensity = cp.mean(pixels, axis=1)  # Average across RGB
peak_position = cp.argmax(intensity)
beam_width = cp.sum(intensity > cp.max(intensity) * 0.5)  # FWHM
```

### 3. Alignment/Calibration Line
```python
# Diagonal line for alignment check
line = Line(x1=0, y1=0, x2=639, y2=479, color=(1,0,1,1), thickness=3)
roi = line.extract_pixels_with_thickness(frame)

# Check uniformity
mean_intensity = cp.mean(roi)
std_intensity = cp.std(roi)
is_uniform = std_intensity < threshold
```

### 4. Real-time Profile Monitoring
```python
# In your capture loop
line = Line(x1=100, y1=240, x2=540, y2=240, color=(1,0,0,0.5), thickness=5)

while capturing:
    frame = cam_manager.get_frame(type="avg")
    frame_3d = frame.reshape((height, width, 3))
    frame_gpu = cp.asarray(frame_3d)
    
    # Extract ROI
    profile = line.extract_pixels(frame_gpu)
    
    # Real-time statistics
    current_mean = cp.mean(profile)
    current_std = cp.std(profile)
    
    # Update display or log data
    print(f"Profile: mean={current_mean:.3f}, std={current_std:.3f}")
```

## Parameters

### `num_samples` (optional)
Controls how many points to sample along the line.

- **`None` (default)**: Samples at every pixel along the line (high resolution)
- **Integer value**: Samples exactly that many points (for downsampling)

```python
# High resolution (default)
pixels_dense = line.extract_pixels(frame)  # e.g., (400, 3)

# Downsampled for faster processing
pixels_sparse = line.extract_pixels(frame, num_samples=50)  # (50, 3)
```

## Performance Notes

1. **GPU Accelerated**: All extraction happens on the GPU using CuPy operations
2. **No Memory Transfer**: If your frame is already on GPU, no CPU↔GPU transfer occurs
3. **Efficient Sampling**: Uses vectorized operations for fast extraction
4. **Zero-copy**: Uses CuPy indexing without creating intermediate copies

## Integration with Camera Manager

```python
# Get the overlay manager
overlay_mgr = cam_manager.get_overlay_manager()

# Create and add line ROI
line = Line(x1=100, y1=240, x2=540, y2=240, 
            color=(1, 0, 0, 0.7),  # Semi-transparent red
            thickness=5)
overlay_mgr.add_primitive(line)

# In your processing loop
def analyze_frame():
    # Get averaged frame from camera (already on GPU)
    frame_1d = cam_manager.get_frame(type="avg")
    frame_3d = frame_1d.reshape((height, width, 3))
    
    # Extract ROI pixels
    roi_pixels = line.extract_pixels_with_thickness(frame_3d)
    
    # Analyze
    profile = cp.mean(roi_pixels, axis=1)  # Average across thickness
    mean_val = cp.mean(profile)
    
    # Optional: transfer to CPU for further processing
    profile_cpu = cp.asnumpy(profile)
    
    return profile_cpu
```

## Data Access Patterns

The extracted arrays follow standard NumPy/CuPy conventions:

```python
# extract_pixels() returns (N, C)
pixels = line.extract_pixels(frame)
pixels[i, :]      # All channels at position i along line
pixels[:, c]      # Channel c for all positions
pixels[i, c]      # Single value at position i, channel c

# extract_pixels_with_thickness() returns (N, T, C)
roi = line.extract_pixels_with_thickness(frame)
roi[i, :, :]      # All thickness samples at position i (T, C)
roi[:, t, :]      # Specific thickness offset t for all positions (N, C)
roi[i, t, c]      # Single pixel value
```

## Tips

1. **Line thickness for noise reduction**: Use `extract_pixels_with_thickness()` with a larger thickness and average across it to reduce noise
2. **Dynamic ROI**: You can update line position in real-time by setting `line.start_x = new_value`
3. **Multiple lines**: Add multiple Line primitives to monitor different regions simultaneously
4. **Coordinate system**: Uses pixel coordinates where (0,0) is top-left
5. **Bounds safety**: All sampling is automatically clipped to image boundaries

## Tested Results

From `test_stuff/test_line_roi.py`:
- ✅ Horizontal line: Correctly extracts horizontal gradient
- ✅ Vertical line: Correctly extracts vertical gradient  
- ✅ Diagonal line: Correctly samples both gradients
- ✅ Thickness sampling: Extracts full ROI perpendicular to line
- ✅ Custom sampling: Sparse sampling works correctly
- ✅ Statistical analysis: Mean, std, min, max all function properly

