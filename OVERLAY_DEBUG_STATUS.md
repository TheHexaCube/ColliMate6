# Overlay System Debug Status

## Current Issue

The GPU overlay primitives system is implemented but the drawing kernels are not writing to the overlay buffer.

## What Works ✓

1. **Infrastructure is complete:**
   - Primitive classes (Line, Cross, Circle) with dirty tracking
   - OverlayManager with caching
   - Integration into CamManager capture loop
   - Alpha compositing kernel works
   
2. **Buffer access works:**
   - Manual writes to buffer work: `overlay[100, 100, :] = [1.0, 0.0, 0.0, 1.0]` ✓
   - Simple test kernels can write: `data[idx] = 0.5f` ✓
   - Kernel with RGBA indexing works: `overlay[idx*4+3] = 0.75f` ✓
   - Minimal cross kernel works: Single pixel at center ✓

3. **Array layout is correct:**
   - Buffer is C-contiguous
   - Shape: (H, W, 4)
   - Strides correct: (1600, 16, 4) for 100x100 image
   - `.ravel()` produces correct 1D view

## What Doesn't Work ✗

The actual draw_cross, draw_circle, and draw_line kernels do NOT write anything to the buffer.

- Kernel compiles without errors
- Kernel is called (blocks=40, threads=256 for 100x100 image)
- Parameters are passed correctly
- But max alpha stays at 0.0 after kernel execution

## Test Results

```python
# This WORKS:
simple_kernel = cp.RawKernel(r'''
extern "C" __global__
void test(float* data, int size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size) data[idx] = 0.5f;
}
''', 'test')
```

```python
# This WORKS:
minimal_cross = cp.RawKernel(r'''
extern "C" __global__
void minimal_cross(float* overlay, int w, int h, int cx, int cy) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= w*h) return;
    
    int px = idx % w;
    int py = idx / w;
    
    if (px == cx && py == cy) {
        overlay[idx*4+3] = 1.0f;
    }
}
''', 'minimal_cross')
```

```python
# This DOESN'T WORK:
kernel_draw_cross = cp.RawKernel(r'''
extern "C" __global__
void draw_cross(float* overlay, int width, int height,
                int cx, int cy, float size, ...) {
    // Complex logic with rotation, distance checks
    // Never writes anything!
}
''', 'draw_cross')
```

## Hypothesis

The complex conditional logic in the full kernels might have a bug that causes NO pixels to meet the drawing condition, even though mathematically they should.

Possible issues:
1. Type casting issues (`int` vs `float`)
2. `abs()` vs `fabsf()` behavior
3. Integer division in `thickness / 2`
4. Rotation math producing NaN or inf
5. Comparison operators behaving unexpectedly

## Next Steps

1. **Simplify the cross kernel** - Remove all complex logic, just draw a filled square
2. **Add debug output** - Have kernel write to a debug buffer showing which pixels checked
3. **Test without rotation** - Set rotation=0 and remove cos/sin entirely
4. **Use integer math only** - Avoid all floating point in conditionals
5. **Copy working minimal kernel** - Gradually add features until it breaks

## Files Modified

- `core/primitives.py` - Primitive classes ✓
- `core/overlay_manager.py` - Manager with debug output ✓
- `core/gpu.py` - Drawing kernels (NOT WORKING)
- `core/cam_manager.py` - Integration ✓
- `gui/widgets.py` - Simplified display ✓

## Workaround

Until kernels are fixed, you can:
1. Use CPU-based drawing (OpenCV, PIL)
2. Draw directly in Python before transferring to GPU
3. Use a different overlay approach (texture-based)

