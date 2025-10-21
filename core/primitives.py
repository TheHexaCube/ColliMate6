"""
GPU-accelerated overlay primitives for real-time drawing.

Primitives track their state and only trigger re-rendering when parameters change.
All primitives render into a shared RGBA overlay buffer on the GPU.
"""

import math
from abc import ABC, abstractmethod
from typing import Tuple
import cupy as cp


class OverlayPrimitive(ABC):
    """Base class for overlay primitives with automatic dirty tracking."""
    
    def __init__(self, color: Tuple[float, float, float, float], visible: bool = True):
        """
        Args:
            color: RGBA color tuple (0.0-1.0 range)
            visible: Whether primitive should be rendered
        """
        self._color = tuple(color)
        self._visible = visible
        self._dirty = True
        self._params_hash = None
        
    @property
    def color(self) -> Tuple[float, float, float, float]:
        return self._color
    
    @color.setter
    def color(self, value: Tuple[float, float, float, float]):
        self._color = tuple(value)
        self._mark_dirty()
    
    @property
    def visible(self) -> bool:
        return self._visible
    
    @visible.setter
    def visible(self, value: bool):
        if self._visible != value:
            self._visible = value
            self._mark_dirty()
    
    def _mark_dirty(self):
        """Mark this primitive as needing re-render."""
        self._dirty = True
        
    def is_dirty(self) -> bool:
        """Check if primitive needs re-rendering."""
        return self._dirty
    
    def mark_clean(self):
        """Mark primitive as rendered (called after rendering)."""
        self._dirty = False
        self._params_hash = self._compute_hash()
    
    @abstractmethod
    def _compute_hash(self) -> int:
        """Compute hash of all parameters for change detection."""
        pass
    
    @abstractmethod
    def render_to_buffer(self, overlay_buffer: cp.ndarray):
        """Render this primitive into the RGBA overlay buffer on GPU."""
        pass


class Line(OverlayPrimitive):
    """Straight line primitive with configurable thickness."""
    
    def __init__(self, start_x: float, start_y: float, end_x: float, end_y: float,
                 color: Tuple[float, float, float, float], thickness: int = 1, visible: bool = True):
        super().__init__(color, visible)
        self._start_x = float(start_x)
        self._start_y = float(start_y)
        self._end_x = float(end_x)
        self._end_y = float(end_y)
        self._thickness = int(thickness)
    
    @property
    def start_x(self) -> float:
        return self._start_x
    
    @start_x.setter
    def start_x(self, value: float):
        self._start_x = float(value)
        self._mark_dirty()
    
    @property
    def start_y(self) -> float:
        return self._start_y
    
    @start_y.setter
    def start_y(self, value: float):
        self._start_y = float(value)
        self._mark_dirty()
    
    @property
    def end_x(self) -> float:
        return self._end_x
    
    @end_x.setter
    def end_x(self, value: float):
        self._end_x = float(value)
        self._mark_dirty()
    
    @property
    def end_y(self) -> float:
        return self._end_y
    
    @end_y.setter
    def end_y(self, value: float):
        self._end_y = float(value)
        self._mark_dirty()
    
    @property
    def thickness(self) -> int:
        return self._thickness
    
    @thickness.setter
    def thickness(self, value: int):
        self._thickness = int(value)
        self._mark_dirty()
    
    def _compute_hash(self) -> int:
        return hash((self._start_x, self._start_y, self._end_x, self._end_y,
                     self._thickness, self._color, self._visible))
    
    def render_to_buffer(self, overlay_buffer: cp.ndarray):
        """Render line into RGBA overlay buffer."""
        from core.gpu import draw_line
        if self._visible:
            draw_line(overlay_buffer, 
                     int(self._start_x), int(self._start_y),
                     int(self._end_x), int(self._end_y),
                     self._color, self._thickness)
    
    def extract_pixels(self, source_frame: cp.ndarray, num_samples: int = None) -> cp.ndarray:
        """
        Extract pixel values along the line from a source frame (Region of Interest).
        
        Args:
            source_frame: Source image (H, W, C) on GPU or CPU
            num_samples: Number of points to sample along the line. 
                        If None, samples at each pixel along the line (default)
        
        Returns:
            CuPy array of shape (N, C) where N is number of samples and C is channels.
            Each row contains the pixel values at that point along the line.
        
        Example:
            >>> line = Line(100, 200, 500, 200, (1,0,0,1), thickness=3)
            >>> pixels = line.extract_pixels(rgb_frame)  # Shape: (400, 3)
            >>> mean_intensity = cp.mean(pixels, axis=0)  # Average RGB along line
        """
        # Convert to CuPy if needed
        if not isinstance(source_frame, cp.ndarray):
            source_frame = cp.asarray(source_frame)
        
        h, w = source_frame.shape[:2]
        channels = source_frame.shape[2] if len(source_frame.shape) == 3 else 1
        
        # Calculate line length
        dx = self._end_x - self._start_x
        dy = self._end_y - self._start_y
        length = math.sqrt(dx**2 + dy**2)
        
        # Determine number of samples
        if num_samples is None:
            # Sample at each pixel along the line
            num_samples = max(int(length) + 1, 2)
        
        # Generate interpolation parameters (0.0 to 1.0)
        t = cp.linspace(0, 1, num_samples, dtype=cp.float32)
        
        # Calculate sample positions along the line
        sample_x = self._start_x + t * dx
        sample_y = self._start_y + t * dy
        
        # Round to nearest pixel and clip to image bounds
        sample_x = cp.clip(cp.round(sample_x).astype(cp.int32), 0, w - 1)
        sample_y = cp.clip(cp.round(sample_y).astype(cp.int32), 0, h - 1)
        
        # Extract pixels
        if len(source_frame.shape) == 3:
            pixels = source_frame[sample_y, sample_x, :]
        else:
            pixels = source_frame[sample_y, sample_x]
            pixels = pixels[:, cp.newaxis]  # Add channel dimension
        
        return pixels
    
    def extract_pixels_with_thickness(self, source_frame: cp.ndarray, num_samples: int = None) -> cp.ndarray:
        """
        Extract pixel values along the line INCLUDING thickness (perpendicular sampling).
        This samples across the line's thickness to get all pixels within the ROI.
        
        Args:
            source_frame: Source image (H, W, C) on GPU or CPU
            num_samples: Number of points to sample along the line length.
                        If None, samples at each pixel along the line (default)
        
        Returns:
            CuPy array of shape (N, thickness, C) where:
            - N is number of samples along the line
            - thickness is the line thickness
            - C is number of channels
        
        Example:
            >>> line = Line(100, 200, 500, 200, (1,0,0,1), thickness=5)
            >>> pixels = line.extract_pixels_with_thickness(rgb_frame)  # Shape: (400, 5, 3)
            >>> mean_across_thickness = cp.mean(pixels, axis=1)  # Average across width
            >>> profile_along_line = cp.mean(pixels, axis=(1,2))  # Avg intensity along line
        """
        # Convert to CuPy if needed
        if not isinstance(source_frame, cp.ndarray):
            source_frame = cp.asarray(source_frame)
        
        h, w = source_frame.shape[:2]
        channels = source_frame.shape[2] if len(source_frame.shape) == 3 else 1
        
        # Calculate line parameters
        dx = self._end_x - self._start_x
        dy = self._end_y - self._start_y
        length = math.sqrt(dx**2 + dy**2)
        
        # Determine number of samples along line
        if num_samples is None:
            num_samples = max(int(length) + 1, 2)
        
        # Calculate perpendicular direction (normal vector)
        if length > 0:
            perp_x = -dy / length  # Perpendicular to line
            perp_y = dx / length
        else:
            perp_x, perp_y = 0, 1
        
        # Sample positions along the line
        t = cp.linspace(0, 1, num_samples, dtype=cp.float32)
        center_x = self._start_x + t * dx
        center_y = self._start_y + t * dy
        
        # For each point along the line, sample perpendicular points for thickness
        half_thickness = self._thickness // 2
        thickness_offsets = cp.arange(-half_thickness, half_thickness + 1, dtype=cp.int32)
        
        # Create meshgrid for all samples
        center_x_grid = center_x[:, cp.newaxis]  # (num_samples, 1)
        center_y_grid = center_y[:, cp.newaxis]  # (num_samples, 1)
        offsets_grid = thickness_offsets[cp.newaxis, :]  # (1, thickness)
        
        # Calculate all sample positions
        sample_x = center_x_grid + offsets_grid * perp_x
        sample_y = center_y_grid + offsets_grid * perp_y
        
        # Round and clip to image bounds
        sample_x = cp.clip(cp.round(sample_x).astype(cp.int32), 0, w - 1)
        sample_y = cp.clip(cp.round(sample_y).astype(cp.int32), 0, h - 1)
        
        # Extract pixels
        if len(source_frame.shape) == 3:
            pixels = source_frame[sample_y, sample_x, :]  # (num_samples, thickness, channels)
        else:
            pixels = source_frame[sample_y, sample_x]
            pixels = pixels[:, :, cp.newaxis]  # Add channel dimension
        
        return pixels


class Cross(OverlayPrimitive):
    """Cross/crosshair primitive with rotation support."""
    
    def __init__(self, center_x: float, center_y: float, size: float,
                 color: Tuple[float, float, float, float], thickness: int = 1,
                 rotation: float = 0.0, visible: bool = True):
        """
        Args:
            center_x, center_y: Center position
            size: Half-length of each arm from center
            color: RGBA color
            thickness: Line thickness
            rotation: Rotation in radians (0 = horizontal/vertical)
            visible: Visibility flag
        """
        super().__init__(color, visible)
        self._center_x = float(center_x)
        self._center_y = float(center_y)
        self._size = float(size)
        self._thickness = int(thickness)
        self._rotation = float(rotation)
    
    @property
    def center_x(self) -> float:
        return self._center_x
    
    @center_x.setter
    def center_x(self, value: float):
        self._center_x = float(value)
        self._mark_dirty()
    
    @property
    def center_y(self) -> float:
        return self._center_y
    
    @center_y.setter
    def center_y(self, value: float):
        self._center_y = float(value)
        self._mark_dirty()
    
    @property
    def size(self) -> float:
        return self._size
    
    @size.setter
    def size(self, value: float):
        self._size = float(value)
        self._mark_dirty()
    
    @property
    def thickness(self) -> int:
        return self._thickness
    
    @thickness.setter
    def thickness(self, value: int):
        self._thickness = int(value)
        self._mark_dirty()
    
    @property
    def rotation(self) -> float:
        """Rotation in radians."""
        return self._rotation
    
    @rotation.setter
    def rotation(self, value: float):
        self._rotation = float(value)
        self._mark_dirty()
    
    @property
    def rotation_deg(self) -> float:
        """Rotation in degrees (convenience property)."""
        return math.degrees(self._rotation)
    
    @rotation_deg.setter
    def rotation_deg(self, value: float):
        self._rotation = math.radians(value)
        self._mark_dirty()
    
    def _compute_hash(self) -> int:
        return hash((self._center_x, self._center_y, self._size,
                     self._thickness, self._rotation, self._color, self._visible))
    
    def render_to_buffer(self, overlay_buffer: cp.ndarray):
        """Render rotated cross into RGBA overlay buffer."""
        from core.gpu import draw_cross
        if self._visible:
            draw_cross(overlay_buffer,
                      int(self._center_x), int(self._center_y),
                      self._size, self._color, self._thickness, self._rotation)


class Circle(OverlayPrimitive):
    """Circle primitive (filled or outline)."""
    
    def __init__(self, center_x: float, center_y: float, radius: float,
                 color: Tuple[float, float, float, float], thickness: int = 1,
                 filled: bool = False, visible: bool = True):
        super().__init__(color, visible)
        self._center_x = float(center_x)
        self._center_y = float(center_y)
        self._radius = float(radius)
        self._thickness = int(thickness)
        self._filled = bool(filled)
    
    @property
    def center_x(self) -> float:
        return self._center_x
    
    @center_x.setter
    def center_x(self, value: float):
        self._center_x = float(value)
        self._mark_dirty()
    
    @property
    def center_y(self) -> float:
        return self._center_y
    
    @center_y.setter
    def center_y(self, value: float):
        self._center_y = float(value)
        self._mark_dirty()
    
    @property
    def radius(self) -> float:
        return self._radius
    
    @radius.setter
    def radius(self, value: float):
        self._radius = float(value)
        self._mark_dirty()
    
    @property
    def thickness(self) -> int:
        return self._thickness
    
    @thickness.setter
    def thickness(self, value: int):
        self._thickness = int(value)
        self._mark_dirty()
    
    @property
    def filled(self) -> bool:
        return self._filled
    
    @filled.setter
    def filled(self, value: bool):
        if self._filled != value:
            self._filled = bool(value)
            self._mark_dirty()
    
    def _compute_hash(self) -> int:
        return hash((self._center_x, self._center_y, self._radius,
                     self._thickness, self._filled, self._color, self._visible))
    
    def render_to_buffer(self, overlay_buffer: cp.ndarray):
        """Render circle into RGBA overlay buffer."""
        from core.gpu import draw_circle
        if self._visible:
            draw_circle(overlay_buffer,
                       int(self._center_x), int(self._center_y),
                       self._radius, self._color, self._thickness, self._filled)

