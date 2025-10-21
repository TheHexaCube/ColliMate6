# Overlay System - CUDA Parameter Passing Issue - FIXED

## Problem
The GPU overlay drawing system was not working. Drawing kernels (cross, line, circle) were not writing any data to the overlay buffer, despite correct buffer allocation and kernel compilation.

## Root Cause
**CUDA parameter passing issue with Python native types**

When passing parameters to CUDA kernels using CuPy's RawKernel:
- Python's native `int` and `float` types were NOT being correctly interpreted by CUDA
- This caused parameter values to be misaligned or corrupted
- The kernels would execute but with incorrect parameter values

## Solution
**Explicit type casting using numpy dtypes**

All parameters passed to CUDA kernels MUST be explicitly cast to:
- `np.int32()` for integer parameters
- `np.float32()` for float parameters

### Before (BROKEN):
```python
kernel_draw_cross(
    (blocks,), (threads,),
    (overlay_1d, w, h, cx, cy, thickness, size, r, g, b, a, rotation)
)
```

### After (WORKING):
```python
kernel_draw_cross(
    (blocks,), (threads,),
    (overlay_1d, 
     np.int32(w), np.int32(h), np.int32(cx), np.int32(cy), np.int32(thickness),
     np.float32(size), np.float32(r), np.float32(g), np.float32(b), np.float32(a), np.float32(rotation))
)
```

## Testing
Confirmed working with test script `test_stuff/test_kernel.py`:
- ✓ Cross drawing: 237 pixels with alpha=1.0
- ✓ Rotation support: Functional
- ✓ All parameters correctly passed to CUDA

## Files Modified
- `core/gpu.py`: Added `np.int32()` and `np.float32()` casts to all drawing wrapper functions:
  - `draw_line()`
  - `draw_circle()`
  - `draw_cross()`
- `core/overlay_manager.py`: Removed debug print statements

## Lessons Learned
1. **Always explicitly cast parameters when calling CUDA kernels** - Python's type inference doesn't match CUDA's expectations
2. **Parameter alignment matters** - While grouping ints before floats can help, explicit casting is the real solution
3. **Test with minimal kernels first** - Isolated testing revealed the parameter count threshold where the issue occurred

## Status
✅ **RESOLVED** - GPU overlay system is now fully functional and ready for use.

