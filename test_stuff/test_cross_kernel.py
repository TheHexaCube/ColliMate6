"""Debug the cross kernel step by step."""

import sys
sys.path.insert(0, 'D:\\EDA\\projects\\hexacube\\autocollimator\\ColliMate6')

import cupy as cp

# Force reload of the module to get latest kernel
import importlib
import processing.gpu_kernels
importlib.reload(processing.gpu_kernels)

# Create test buffer
h, w = 100, 100
overlay = cp.zeros((h, w, 4), dtype=cp.float32)
overlay_1d = overlay.ravel()

#Step 1: Test with minimal parameters
cross_kernel = cp.RawKernel(r'''
extern "C" __global__
void draw_cross(float* overlay, int width, int height)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total = width * height;
    if (idx >= total) return;
    
    // JUST FILL WITH 0.9 TO TEST
    int offset = idx * 4;
    overlay[offset + 3] = 0.9f;
}
''', 'draw_cross')
print(f"Kernel object: {cross_kernel}")

cx, cy = 50, 50
total = h * w
threads = 256
blocks = (total + threads - 1) // threads

print(f"Calling cross kernel:")
print(f"  Buffer: shape={overlay_1d.shape}, dtype={overlay_1d.dtype}")
print(f"  Grid: blocks={blocks}, threads={threads}")
print(f"  Params: w={w}, h={h}, cx={cx}, cy={cy}, size=20.0")
print(f"  Color: r=1.0, g=0.0, b=0.0, a=1.0")
print(f"  thickness=3, rotation=0.0")

cross_kernel(
    (blocks,), (threads,),
    (overlay_1d, w, h)
)
cp.cuda.Stream.null.synchronize()

print(f"\nResult: max alpha = {cp.max(overlay[:,:,3])}")
print(f"Non-zero pixels: {cp.count_nonzero(overlay[:,:,3] > 0.01)}")

if cp.max(overlay[:,:,3]) > 0:
    print("\n✓ SUCCESS! Kernel is working!")
    # Show some pixels
    for y in range(48, 53):
        for x in range(48, 53):
            alpha = overlay[y, x, 3]
            if alpha > 0:
                print(f"  Pixel [{y},{x}]: alpha={alpha:.2f}")
else:
    print("\n✗ FAILURE - kernel didn't write anything")
    print("This suggests a kernel compilation or parameter issue")

