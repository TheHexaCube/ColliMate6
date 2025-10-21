"""
Overlay manager for GPU-accelerated primitive rendering.

Manages a collection of overlay primitives and handles efficient rendering
into a shared RGBA overlay buffer with dirty-flag based caching.
"""

import cupy as cp
from typing import List
from core.primitives import OverlayPrimitive
from core.gpu import composite_overlay


class OverlayManager:
    """Manages overlay primitives and handles rendering with caching."""
    
    def __init__(self):
        self.primitives: List[OverlayPrimitive] = []
        self.overlay_buffer = None
        self._width = 0
        self._height = 0
        self._composited_buffer = None
        
    def set_resolution(self, width: int, height: int):
        """
        Set or resize the overlay buffer resolution.
        
        Args:
            width: Width in pixels
            height: Height in pixels
        """
        if self._width == width and self._height == height and self.overlay_buffer is not None:
            return  # Already correct size
        
        self._width = width
        self._height = height
        
        # Create RGBA overlay buffer (H, W, 4)
        self.overlay_buffer = cp.zeros((height, width, 4), dtype=cp.float32)
        
        # Create composited output buffer (H, W, 3)
        self._composited_buffer = cp.empty((height, width, 3), dtype=cp.float32)
        
        # Mark all primitives dirty since resolution changed
        for prim in self.primitives:
            prim._mark_dirty()
    
    def add_primitive(self, primitive: OverlayPrimitive):
        """Add a primitive to the overlay."""
        if primitive not in self.primitives:
            self.primitives.append(primitive)
            primitive._mark_dirty()
    
    def remove_primitive(self, primitive: OverlayPrimitive):
        """Remove a primitive from the overlay."""
        if primitive in self.primitives:
            self.primitives.remove(primitive)
            # Mark remaining primitives dirty to trigger re-render
            for prim in self.primitives:
                prim._mark_dirty()
    
    def clear_all(self):
        """Remove all primitives from the overlay."""
        self.primitives.clear()
        if self.overlay_buffer is not None:
            self.overlay_buffer.fill(0.0)
    
    def render_overlay(self):
        """
        Render all dirty primitives into the overlay buffer.
        Only re-renders if at least one primitive is dirty.
        """
        if self.overlay_buffer is None:
            return  # No buffer allocated yet
        
        # Check if any primitive is dirty
        any_dirty = any(prim.is_dirty() for prim in self.primitives)
        
        if not any_dirty:
            return  # Use cached overlay
        
        # Clear overlay buffer (set all alpha to 0)
        self.overlay_buffer.fill(0.0)
        
        # Render each visible primitive
        for prim in self.primitives:
            if prim.visible:
                prim.render_to_buffer(self.overlay_buffer)
            prim.mark_clean()
    
    def composite_onto_frame(self, rgb_frame: cp.ndarray, output: cp.ndarray = None) -> cp.ndarray:
        """
        Alpha-composite the overlay onto an RGB frame.
        
        Args:
            rgb_frame: Input RGB frame (H, W, 3) on GPU
            output: Optional pre-allocated output buffer. If None, uses internal buffer.
        
        Returns:
            Composited RGB frame (H, W, 3)
        """
        if self.overlay_buffer is None:
            # No overlay, just return input (or copy to output)
            if output is not None:
                output[:] = rgb_frame
                return output
            return rgb_frame
        
        if output is None:
            output = self._composited_buffer
        
        # Composite overlay onto RGB frame
        composite_overlay(rgb_frame, self.overlay_buffer, output)
        
        return output
    
    def get_primitive_count(self) -> int:
        """Return number of primitives in the overlay."""
        return len(self.primitives)
    
    def get_visible_primitive_count(self) -> int:
        """Return number of visible primitives."""
        return sum(1 for prim in self.primitives if prim.visible)

