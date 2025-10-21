"""Test if the draw kernels actually work."""

import sys
sys.path.insert(0, 'D:\\EDA\\projects\\hexacube\\autocollimator\\ColliMate6')

import cupy as cp
from core.gpu import draw_cross, draw_circle, draw_line

# Create a small test overlay buffer
h, w = 100, 100
overlay = cp.zeros((h, w, 4), dtype=cp.float32)

print(f"Initial overlay buffer - max alpha: {cp.max(overlay[:,:,3])}")
print(f"Overlay is C-contiguous: {overlay.flags.c_contiguous}")
print(f"Overlay is F-contiguous: {overlay.flags.f_contiguous}")
print(f"Overlay shape: {overlay.shape}, strides: {overlay.strides}")

# First try a simple raw kernel test
print(f"\n=== Testing raw kernel directly ===")
test_kernel = cp.RawKernel(r'''
extern "C" __global__
void test_write(float* data, int size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size) {
        data[idx] = 0.5f;
    }
}
''', 'test_write')

test_array = cp.zeros(100, dtype=cp.float32)
print(f"Before kernel: max = {cp.max(test_array)}")
test_kernel((1,), (100,), (test_array, 100))
cp.cuda.Stream.null.synchronize()
print(f"After kernel: max = {cp.max(test_array)}")

# Test if kernel can access the buffer at all
print(f"\n=== Testing if kernel threads run ===")
simple_kernel = cp.RawKernel(r'''
extern "C" __global__
void fill_alpha(float* overlay, int w, int h) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total = w * h;
    if (idx >= total) return;
    
    // Write to alpha channel (every 4th element starting from 3)
    overlay[idx * 4 + 3] = 0.75f;
}
''', 'fill_alpha')

overlay_1d = overlay.ravel()
total = h * w
threads = 256
blocks = (total + threads - 1) // threads
simple_kernel((blocks,), (threads,), (overlay_1d, w, h))
cp.cuda.Stream.null.synchronize()
print(f"After fill_alpha - max alpha: {cp.max(overlay[:,:,3])}")

# Reset overlay
overlay.fill(0.0)

# Test a minimal cross kernel
print(f"\n=== Testing minimal cross kernel ===")
minimal_cross = cp.RawKernel(r'''
extern "C" __global__
void minimal_cross(float* overlay, int w, int h, int cx, int cy) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total = w * h;
    if (idx >= total) return;
    
    int px = idx % w;
    int py = idx / w;
    
    // Just draw center pixel
    if (px == cx && py == cy) {
        int offset = idx * 4;
        overlay[offset + 0] = 1.0f;  // R
        overlay[offset + 1] = 0.0f;  // G
        overlay[offset + 2] = 0.0f;  // B
        overlay[offset + 3] = 1.0f;  // A
    }
}
''', 'minimal_cross')

cx, cy = 50, 50
minimal_cross((blocks,), (threads,), (overlay_1d, w, h, cx, cy))
cp.cuda.Stream.null.synchronize()
print(f"After minimal_cross - max alpha: {cp.max(overlay[:,:,3])}")
print(f"Center pixel [50,50]: {overlay[50, 50, :]}")

# Reset
overlay.fill(0.0)

# Try calling the cross kernel directly
print(f"\n=== Calling draw_cross kernel directly ===")
from core.gpu import kernel_draw_cross
overlay_1d = overlay.ravel()
print(f"Overlay 1D shape: {overlay_1d.shape}, is contiguous: {overlay_1d.flags.c_contiguous}")
print(f"Kernel grid: blocks={blocks}, threads={threads}")
kernel_draw_cross(
    (blocks,), (threads,),
    (overlay_1d, w, h, cx, cy, 20.0, 1.0, 0.0, 0.0, 1.0, 3, 0.0)
)
cp.cuda.Stream.null.synchronize()
print(f"After direct kernel call - max alpha: {cp.max(overlay[:,:,3])}")

# Try drawing a cross at center
print(f"\n=== Drawing cross at ({cx}, {cy}) via wrapper ===")
draw_cross(overlay, cx, cy, size=20, color=(1.0, 0.0, 0.0, 1.0), thickness=3, rotation=0.0)
cp.cuda.Stream.null.synchronize()

print(f"After draw_cross - max alpha: {cp.max(overlay[:,:,3])}")
print(f"Non-zero alpha pixels: {cp.count_nonzero(overlay[:,:,3] > 0.01)}")

# Check specific pixels near center
print(f"\nPixel values near center:")
for dy in range(-2, 3):
    for dx in range(-2, 3):
        alpha = overlay[cy+dy, cx+dx, 3]
        if alpha > 0:
            print(f"  [{cy+dy}, {cx+dx}]: alpha = {alpha}")

# Try manual kernel call
print(f"\n Testing if we can write to overlay at all...")
overlay[50, 50, :] = cp.array([1.0, 1.0, 0.0, 1.0])
print(f"After manual write - pixel [50,50]: {overlay[50, 50, :]}")

