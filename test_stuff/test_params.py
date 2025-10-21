"""Test parameter passing to cross kernel."""

import sys
sys.path.insert(0, 'D:\\EDA\\projects\\hexacube\\autocollimator\\ColliMate6')

import cupy as cp

h, w = 100, 100
overlay = cp.zeros((h, w, 4), dtype=cp.float32)
overlay_1d = overlay.ravel()
total = h * w
threads = 256
blocks = (total + threads - 1) // threads

# Test 1: 3 params - WORKS
print("Test 1: 3 params")
k1 = cp.RawKernel(r'''
extern "C" __global__
void test(float* overlay, int width, int height) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= width*height) return;
    overlay[idx*4+3] = 0.1f;
}
''', 'test')
k1((blocks,), (threads,), (overlay_1d, w, h))
cp.cuda.Stream.null.synchronize()
print(f"  Result: {cp.max(overlay[:,:,3])}")
overlay.fill(0.0)

# Test 2: 5 params
print("Test 2: 5 params")
k2 = cp.RawKernel(r'''
extern "C" __global__
void test(float* overlay, int width, int height, int cx, int cy) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= width*height) return;
    overlay[idx*4+3] = 0.2f;
}
''', 'test')
k2((blocks,), (threads,), (overlay_1d, w, h, 50, 50))
cp.cuda.Stream.null.synchronize()
print(f"  Result: {cp.max(overlay[:,:,3])}")
overlay.fill(0.0)

# Test 3: 6 params (add float)
print("Test 3: 6 params (add float)")
k3 = cp.RawKernel(r'''
extern "C" __global__
void test(float* overlay, int width, int height, int cx, int cy, float size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= width*height) return;
    overlay[idx*4+3] = 0.3f;
}
''', 'test')
k3((blocks,), (threads,), (overlay_1d, w, h, 50, 50, 20.0))
cp.cuda.Stream.null.synchronize()
print(f"  Result: {cp.max(overlay[:,:,3])}")
overlay.fill(0.0)

# Test 4: 10 params (add floats for color) - ORIGINAL ORDER (BREAKS)
print("Test 4: 10 params (add RGBA floats) - OLD ORDER")
k4 = cp.RawKernel(r'''
extern "C" __global__
void test(float* overlay, int width, int height, int cx, int cy, float size,
          float r, float g, float b, float a) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= width*height) return;
    overlay[idx*4+3] = a;
}
''', 'test')
k4((blocks,), (threads,), (overlay_1d, w, h, 50, 50, 20.0, 1.0, 0.0, 0.0, 0.4))
cp.cuda.Stream.null.synchronize()
print(f"  Result: {cp.max(overlay[:,:,3])} (expect 0.4, but likely breaks)")
overlay.fill(0.0)

# Test 4b: INTS FIRST, THEN FLOATS
print("Test 4b: 10 params - INTS FIRST, FLOATS LAST")
k4b = cp.RawKernel(r'''
extern "C" __global__
void test(float* overlay, int width, int height, int cx, int cy,
          float size, float r, float g, float b, float a) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= width*height) return;
    overlay[idx*4+3] = a;
}
''', 'test')
k4b((blocks,), (threads,), (overlay_1d, w, h, 50, 50, 20.0, 1.0, 0.0, 0.0, 0.42))
cp.cuda.Stream.null.synchronize()
print(f"  Result: {cp.max(overlay[:,:,3])} (expect 0.42)")
overlay.fill(0.0)

# Test 5: 12 params - INTS FIRST
print("Test 5: 12 params - ALL INTS FIRST, THEN ALL FLOATS")
k5 = cp.RawKernel(r'''
extern "C" __global__
void test(float* overlay, int width, int height, int cx, int cy, int thickness,
          float size, float r, float g, float b, float a, float rotation) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= width*height) return;
    overlay[idx*4+3] = a;
}
''', 'test')
k5((blocks,), (threads,), (overlay_1d, w, h, 50, 50, 3, 20.0, 1.0, 0.0, 0.0, 0.55, 0.0))
cp.cuda.Stream.null.synchronize()
print(f"  Result: {cp.max(overlay[:,:,3])} (expect 0.55)")

print("\nDone!")

