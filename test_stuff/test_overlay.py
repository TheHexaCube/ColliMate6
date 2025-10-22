"""
Example script demonstrating the overlay primitives system.

This shows how to add various primitives (lines, crosses, circles) to the
camera display with alpha transparency.
"""

import time
import math
from visualization.primitives import Line, Cross, Circle


def add_test_primitives(cam_manager):
    """Add example primitives to the overlay for testing."""
    
    # Get the overlay manager
    overlay = cam_manager.get_overlay_manager()
    
    # Get camera resolution for positioning
    w, h = cam_manager.get_resolution()
    cx, cy = w // 2, h // 2
    
    # Create a crosshair at center (red, semi-transparent)
    crosshair = Cross(
        center_x=cx,
        center_y=cy,
        size=50,
        color=(1.0, 0.0, 0.0, 0.8),  # Red, 80% opaque
        thickness=2,
        rotation=0.0  # Can be changed dynamically
    )
    overlay.add_primitive(crosshair)
    
    # Create a circle around the crosshair (green, translucent)
    circle_outer = Circle(
        center_x=cx,
        center_y=cy,
        radius=100,
        color=(0.0, 1.0, 0.0, 0.5),  # Green, 50% opaque
        thickness=2,
        filled=False
    )
    overlay.add_primitive(circle_outer)
    
    # Create a smaller filled circle at center (blue, very translucent)
    circle_inner = Circle(
        center_x=cx,
        center_y=cy,
        radius=10,
        color=(0.0, 0.5, 1.0, 0.3),  # Light blue, 30% opaque
        thickness=1,
        filled=True
    )
    overlay.add_primitive(circle_inner)
    
    # Create diagonal lines in corners (yellow)
    corner_size = 30
    
    # Top-left corner
    line_tl = Line(
        start_x=10, start_y=10,
        end_x=10 + corner_size, end_y=10 + corner_size,
        color=(1.0, 1.0, 0.0, 0.7),  # Yellow, 70% opaque
        thickness=2
    )
    overlay.add_primitive(line_tl)
    
    # Top-right corner
    line_tr = Line(
        start_x=w - 10, start_y=10,
        end_x=w - 10 - corner_size, end_y=10 + corner_size,
        color=(1.0, 1.0, 0.0, 0.7),
        thickness=2
    )
    overlay.add_primitive(line_tr)
    
    # Bottom-left corner
    line_bl = Line(
        start_x=10, start_y=h - 10,
        end_x=10 + corner_size, end_y=h - 10 - corner_size,
        color=(1.0, 1.0, 0.0, 0.7),
        thickness=2
    )
    overlay.add_primitive(line_bl)
    
    # Bottom-right corner
    line_br = Line(
        start_x=w - 10, start_y=h - 10,
        end_x=w - 10 - corner_size, end_y=h - 10 - corner_size,
        color=(1.0, 1.0, 0.0, 0.7),
        thickness=2
    )
    overlay.add_primitive(line_br)
    
    print(f"Added {overlay.get_primitive_count()} test primitives to overlay")
    
    return {
        'crosshair': crosshair,
        'circle_outer': circle_outer,
        'circle_inner': circle_inner,
        'line_tl': line_tl,
        'line_tr': line_tr,
        'line_bl': line_bl,
        'line_br': line_br
    }


def animate_primitives(primitives, duration=10.0):
    """
    Animate the primitives for a few seconds to demonstrate dynamic updates.
    
    This function modifies primitive properties, which automatically marks
    them dirty and triggers re-rendering.
    """
    start_time = time.time()
    
    while time.time() - start_time < duration:
        elapsed = time.time() - start_time
        
        # Rotate the crosshair
        angle = elapsed * 0.5  # Rotate at 0.5 rad/s
        primitives['crosshair'].rotation = angle
        
        # Pulse the outer circle radius
        radius_offset = 20 * math.sin(elapsed * 2)
        primitives['circle_outer'].radius = 100 + radius_offset
        
        # Change crosshair color (fade between red and yellow)
        fade = (math.sin(elapsed * 3) + 1) / 2  # 0 to 1
        primitives['crosshair'].color = (1.0, fade, 0.0, 0.8)
        
        time.sleep(0.033)  # ~30 fps update rate
    
    print("Animation complete")


# Example of how to integrate this into your application:
# 
# 1. After connecting camera:
#    primitives = add_test_primitives(cam_manager)
# 
# 2. To animate (in a separate thread or loop):
#    animate_primitives(primitives)
# 
# 3. To modify a primitive at any time:
#    primitives['crosshair'].center_x = new_x
#    primitives['crosshair'].center_y = new_y
# 
# 4. To hide/show a primitive:
#    primitives['circle_outer'].visible = False
# 
# 5. To remove a primitive:
#    overlay = cam_manager.get_overlay_manager()
#    overlay.remove_primitive(primitives['circle_outer'])
# 
# 6. To clear all primitives:
#    overlay.clear_all()

