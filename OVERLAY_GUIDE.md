# GPU Overlay Primitives - Usage Guide

## Overview

The overlay system provides GPU-accelerated drawing of geometric primitives (lines, crosses, circles) with alpha transparency over camera frames. Primitives automatically track their state and only trigger re-rendering when parameters change, making the system very efficient.

## Quick Start

### Basic Usage

```python
# Get overlay manager from camera manager
overlay = cam_manager.get_overlay_manager()

# Create a crosshair at center
w, h = cam_manager.get_resolution()
crosshair = Cross(
    center_x=w//2, center_y=h//2,
    size=50,
    color=(1.0, 0.0, 0.0, 0.8),  # RGBA: red with 80% opacity
    thickness=2,
    rotation=0.0  # radians
)

# Add to overlay
overlay.add_primitive(crosshair)

# Modify later (automatically marks dirty and re-renders)
crosshair.center_x = 512
crosshair.rotation_deg = 45  # Can use degrees instead of radians
```

## Available Primitives

### Line
Straight line between two points.

```python
from core.primitives import Line

line = Line(
    start_x=100, start_y=100,
    end_x=200, end_y=200,
    color=(1.0, 1.0, 0.0, 0.7),  # Yellow, 70% opaque
    thickness=2,
    visible=True
)
```

**Properties:**
- `start_x`, `start_y`: Start point coordinates
- `end_x`, `end_y`: End point coordinates
- `thickness`: Line width in pixels
- `color`: RGBA tuple (0.0-1.0 range)
- `visible`: Show/hide primitive

### Cross
Crosshair with rotation support.

```python
from core.primitives import Cross

cross = Cross(
    center_x=1024, center_y=768,
    size=50,  # Half-length of each arm from center
    color=(1.0, 0.0, 0.0, 0.8),  # Red, 80% opaque
    thickness=2,
    rotation=0.0,  # Radians (0 = horizontal/vertical)
    visible=True
)

# Use degrees for convenience
cross.rotation_deg = 45
```

**Properties:**
- `center_x`, `center_y`: Center position
- `size`: Half-length of arms from center
- `thickness`: Line width in pixels
- `rotation`: Rotation in radians
- `rotation_deg`: Rotation in degrees (convenience property)
- `color`: RGBA tuple
- `visible`: Show/hide primitive

### Circle
Circle (filled or outline).

```python
from core.primitives import Circle

circle = Circle(
    center_x=512, center_y=512,
    radius=100,
    color=(0.0, 1.0, 0.0, 0.5),  # Green, 50% opaque
    thickness=3,
    filled=False,  # True for filled circle
    visible=True
)
```

**Properties:**
- `center_x`, `center_y`: Center position
- `radius`: Radius in pixels
- `thickness`: Edge width (for unfilled circles)
- `filled`: `True` for filled, `False` for outline
- `color`: RGBA tuple
- `visible`: Show/hide primitive

## Overlay Manager API

### Managing Primitives

```python
overlay = cam_manager.get_overlay_manager()

# Add primitive
overlay.add_primitive(my_primitive)

# Remove specific primitive
overlay.remove_primitive(my_primitive)

# Clear all primitives
overlay.clear_all()

# Get counts
total = overlay.get_primitive_count()
visible = overlay.get_visible_primitive_count()
```

### Resolution

The overlay buffer is automatically initialized when the camera connects. If you need to manually set/change resolution:

```python
overlay.set_resolution(width, height)
```

## Performance Characteristics

### Caching and Dirty Tracking

- **Automatic dirty tracking**: When you modify any primitive property, it's automatically marked dirty
- **Lazy rendering**: Overlay only re-renders when at least one primitive is dirty
- **Shared buffer**: All primitives render into a single RGBA overlay buffer
- **Alpha compositing**: Happens every frame (~0.1-0.3 ms), but primitive rendering only when needed

### Performance Numbers

- **Overlay rendering**: ~0.5-2 ms when primitives change (1-20 primitives)
- **Compositing**: ~0.1-0.3 ms per frame (always runs)
- **Memory**: ~25 MB for 2048×1536 RGBA overlay buffer

### Best Practices

1. **Minimize changes**: Only update primitives when needed
2. **Batch updates**: If updating multiple properties, update them all before the next frame
3. **Use visibility**: Hide primitives instead of removing/re-adding them
4. **Keep count reasonable**: System handles 1-20 primitives easily

## Advanced Usage

### Dynamic Animation

```python
import time
import math

# Rotating crosshair
while running:
    angle = time.time() * 0.5  # 0.5 rad/s
    crosshair.rotation = angle
    time.sleep(0.033)  # ~30 fps
```

### Conditional Visibility

```python
# Show/hide based on detection
if feature_detected:
    marker.visible = True
    marker.center_x = detected_x
    marker.center_y = detected_y
else:
    marker.visible = False
```

### Color Fading

```python
# Pulse effect
fade = (math.sin(time.time() * 2) + 1) / 2  # 0 to 1
circle.color = (1.0, fade, 0.0, 0.7)
```

## Color Reference

Colors use RGBA format with values in the 0.0-1.0 range:

```python
# Common colors (with 70% opacity)
RED = (1.0, 0.0, 0.0, 0.7)
GREEN = (0.0, 1.0, 0.0, 0.7)
BLUE = (0.0, 0.0, 1.0, 0.7)
YELLOW = (1.0, 1.0, 0.0, 0.7)
CYAN = (0.0, 1.0, 1.0, 0.7)
MAGENTA = (1.0, 0.0, 1.0, 0.7)
WHITE = (1.0, 1.0, 1.0, 0.7)
GRAY = (0.5, 0.5, 0.5, 0.7)

# Alpha channel (4th value)
FULLY_OPAQUE = 1.0
SEMI_TRANSPARENT = 0.5
BARELY_VISIBLE = 0.2
INVISIBLE = 0.0
```

## Example: Feature Detection Overlay

```python
def setup_detection_overlay(cam_manager):
    """Setup overlay for feature detection visualization."""
    overlay = cam_manager.get_overlay_manager()
    w, h = cam_manager.get_resolution()
    
    # Center crosshair for alignment
    crosshair = Cross(w//2, h//2, 30, (0.0, 1.0, 0.0, 0.6), thickness=1)
    overlay.add_primitive(crosshair)
    
    # Detection markers (initially hidden)
    markers = []
    for i in range(10):
        marker = Circle(
            0, 0, radius=15,
            color=(1.0, 0.0, 0.0, 0.8),
            thickness=2, filled=False,
            visible=False
        )
        overlay.add_primitive(marker)
        markers.append(marker)
    
    return {'crosshair': crosshair, 'markers': markers}

def update_detections(overlay_data, detections):
    """Update detection markers based on detected features."""
    markers = overlay_data['markers']
    
    # Show markers for detected features
    for i, detection in enumerate(detections[:len(markers)]):
        markers[i].center_x = detection['x']
        markers[i].center_y = detection['y']
        markers[i].visible = True
    
    # Hide unused markers
    for i in range(len(detections), len(markers)):
        markers[i].visible = False
```

## Troubleshooting

**Q: Primitives not showing up?**
- Check that `visible=True`
- Verify coordinates are within frame bounds
- Ensure color alpha > 0
- Check overlay manager is initialized (camera connected)

**Q: Performance issues?**
- Are you updating primitives every frame unnecessarily?
- Consider using visibility instead of add/remove
- Check profiling metrics for compositing overhead

**Q: Colors look wrong?**
- Remember RGBA values are 0.0-1.0, not 0-255
- Alpha=1.0 is fully opaque, 0.0 is transparent
- Check if frame brightness affects visibility

## See Also

- `test_stuff/test_overlay.py` - Example code with animations
- `core/primitives.py` - Primitive class definitions
- `core/overlay_manager.py` - Overlay management implementation
- `core/gpu.py` - GPU kernels for drawing and compositing

