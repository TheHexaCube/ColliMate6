"""Test explicit dtypes for CUDA parameters."""

import sys
sys.path.insert(0, 'D:\\EDA\\projects\\hexacube\\autocollimator\\ColliMate6')

import cupy as cp
import numpy as np

h, w = 100, 100
overlay = cp.zeros((h, w, 4), dtype=cp.float32)
overlay_1d = overlay.ravel()
total = h * w
threads = 256
blocks = (total + threads - 1) // threads

# Test with EXPLICIT numpy dtypes
print("Test with explicit numpy int32 and float32 dtypes")
k = cp.RawKernel(r'''
extern "C" __global__
void test(float* overlay, int width, int height, int cx, int cy, int thickness,
          float size, float r, float g, float b, float a, float rotation) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= width*height) return;
    overlay[idx*4+3] = a;
}
''', 'test')

# Cast everything explicitly
k((blocks,), (threads,), 
  (overlay_1d, 
   np.int32(w), np.int32(h), np.int32(50), np.int32(50), np.int32(3),
   np.float32(20.0), np.float32(1.0), np.float32(0.0), np.float32(0.0), np.float32(0.66), np.float32(0.0)))

cp.cuda.Stream.null.synchronize()
print(f"Result: {cp.max(overlay[:,:,3])} (expect 0.66)")

print("\nDone!")

