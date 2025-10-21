import queue
import cupy as cp
import numpy as np
import time
from line_profiler import profile
from cupyx.profiler import benchmark
import threading
from queue import Full, Empty


# ---------------- OPTIMIZED DEMOSAIC KERNEL ----------------
kernel_demosaic = cp.RawKernel(r'''
extern "C" __global__
void demosaic_bayer_rg12(const unsigned short* __restrict__ src,
                         float* __restrict__ dst,
                         const int width, const int height)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int xe = x & 1;
    int ye = y & 1;
    int idx = y * width + x;

    // Inline clamp and pixel fetch
    #define CLAMP(v,l,h) ((v)<(l)?(l):((v)>(h)?(h):(v)))
    #define PIX(xx,yy) ((float)src[CLAMP((yy),0,height-1)*width + CLAMP((xx),0,width-1)])

    float rf,gf,bf;

    if (ye == 0 && xe == 0) {
        rf = PIX(x,y);
        gf = 0.25f * (PIX(x-1,y)+PIX(x+1,y)+PIX(x,y-1)+PIX(x,y+1));
        bf = 0.25f * (PIX(x-1,y-1)+PIX(x+1,y-1)+PIX(x-1,y+1)+PIX(x+1,y+1));
    } else if (ye == 0 && xe == 1) {
        rf = 0.5f * (PIX(x-1,y)+PIX(x+1,y));
        gf = PIX(x,y);
        bf = 0.5f * (PIX(x,y-1)+PIX(x,y+1));
    } else if (ye == 1 && xe == 0) {
        rf = 0.5f * (PIX(x,y-1)+PIX(x,y+1));
        gf = PIX(x,y);
        bf = 0.5f * (PIX(x-1,y)+PIX(x+1,y));
    } else {
        rf = 0.25f * (PIX(x-1,y-1)+PIX(x+1,y-1)+PIX(x-1,y+1)+PIX(x+1,y+1));
        gf = 0.25f * (PIX(x-1,y)+PIX(x+1,y)+PIX(x,y-1)+PIX(x,y+1));
        bf = PIX(x,y);
    }

    dst[idx*3 + 0] = rf * (1.0f / 4095.0f);
    dst[idx*3 + 1] = gf * (1.0f / 4095.0f);
    dst[idx*3 + 2] = bf * (1.0f / 4095.0f);
}
''', 'demosaic_bayer_rg12')


# ---------------- GPU QUEUE ----------------
class GPUQueue:
    def __init__(self, max_size: int, shape: tuple, dtype=cp.float32):
        self.capacity = max_size
        self.shape = shape
        self.dtype = dtype
        self.queue = cp.zeros((max_size, *shape), dtype=dtype)
        self.head = 0
        self.tail = 0
        self.size = 0

    def put(self, item: cp.ndarray):
        if self.is_full():
            raise Full
        # direct copy into queue slot
        self.queue[self.tail][:] = item.astype(self.dtype, copy=False)
        self.tail = (self.tail + 1) % self.capacity
        self.size += 1

    def get_view(self):
        if self.is_empty():
            raise Empty
        idx = self.head
        self.head = (self.head + 1) % self.capacity
        self.size -= 1
        return self.queue[idx]

    def is_empty(self): return self.size == 0
    def is_full(self): return self.size == self.capacity
    def clear(self): self.head = self.tail = self.size = 0


# ---------------- DEMOSAIC CALL ----------------
def run_demosaic(raw: cp.ndarray, dst: cp.ndarray):
    h, w = raw.shape
    threads = (16, 16)
    blocks = ((w + 15) // 16, (h + 15) // 16)
    kernel_demosaic(blocks, threads, (raw, dst, w, h))


# ---------------- SYNTHETIC TEST IMAGE ----------------
def generate_bayer_rg12_pattern(h, w):
    """Vectorized vertical CMYKRGBW stripes."""
    colors = cp.asarray([
        [0,1,1],[1,0,1],[1,1,0],[0,0,0],
        [1,0,0],[0,1,0],[0,0,1],[1,1,1]
    ], dtype=cp.float32)
    n = colors.shape[0]
    stripe_w = w // n
    X = cp.arange(w)
    Y = cp.arange(h)[:,None]
    idx = cp.minimum(X // stripe_w, n - 1)
    rgb = colors[idx]
    r,g,b = rgb[:,0], rgb[:,1], rgb[:,2]
    Rmask = ((Y & 1) == 0) & ((X & 1) == 0)
    Gmask = (((Y & 1) == 0) & ((X & 1) == 1)) | (((Y & 1) == 1) & ((X & 1) == 0))
    Bmask = ((Y & 1) == 1) & ((X & 1) == 1)
    out = cp.where(Rmask, r, 0) + cp.where(Gmask, g, 0) + cp.where(Bmask, b, 0)
    return (out * 4095).astype(cp.uint16)


# ---------------- PROCESS LOOP ----------------
@profile
def process_frame(src: GPUQueue, dst: GPUQueue):
    # Launch one kernel per frame, async
    h, w = src.shape if hasattr(src, 'shape') else src.queue[0].shape
    threads = (16,16)
    blocks = ((w + 15)//16, (h + 15)//16)

    for i in range(src.size):
        run_demosaic(src.queue[i], dst.queue[i])
    # Single sync for all kernels
    cp.cuda.Device().synchronize()


# ---------------- BENCHMARK ----------------
@profile
def benchmark_process_frame(src: GPUQueue, dst: GPUQueue, test_pattern: cp.ndarray):
    #start = time.perf_counter()
    #print("Clearing queues...")
    src.clear()
    dst.clear()
    #print("Filling input queue...")
    for i in range(src.capacity):
        src.put(test_pattern)
    #print("Running demosaic batch...")
    for i in range(src.size):
        run_demosaic(src.queue[i], dst.queue[i])
    cp.cuda.Device().synchronize()
    #end = time.perf_counter()
    #print(f"Done. Total time: {(end - start)*1000:.2f} ms")


# ---------------- MAIN ----------------
H, W = 2048, 1536

start_time = time.perf_counter()
stop_time = start_time

@profile
def run_fixed_rate(target_fps: float, func, *args, **kwargs):
    """
    Run `func(*args, **kwargs)` exactly `target_fps` times per second.
    Uses a high-precision loop with drift correction.
    """
    period = 1.0 / target_fps
    next_time = time.perf_counter()
    while True:
        start_time = time.perf_counter()
        if func(*args, **kwargs) is not None:
            stop_time = time.perf_counter()
            print(f"Time taken for {func.__name__}: {stop_time - start_time:.6f} seconds")
        next_time += period
        sleep_time = next_time - time.perf_counter()
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            # Missed deadline; skip correction to prevent cumulative drift
            next_time = time.perf_counter()


@profile
def put_frame(q: GPUQueue, frame: cp.ndarray):
    if not q.is_full():
        q.put(frame)
        return True
   


@profile
def get_frame(q: GPUQueue):
    if not q.is_empty():
        return q.get_view()
    else:
        return None


@profile
def process_frame2(src: GPUQueue, dst: GPUQueue):
    if not src.is_empty():
        frame_in = src.get_view()
        result = cp.zeros((H, W, 3), dtype=cp.float32)
        run_demosaic(frame_in, result)
        dst.put(result)
        return True
        


def display_frame(src: GPUQueue):
    if not src.is_empty():
        print("Displaying frame\n")
        frame = src.get_view()
        return True
    else:
        return None

if __name__ == "__main__":
    
    test_pattern = generate_bayer_rg12_pattern(H, W)

    # print("Benchmark for demosaicing: ")
    queue_size = 16
    # for queue_size in [32]:
    q_in = GPUQueue(queue_size, (H, W), dtype=cp.uint16)
    q_out = GPUQueue(queue_size, (H, W, 3), dtype=cp.float32)

    #     test_pattern = generate_bayer_rg12_pattern(H, W)

    #     print("--------------------------------")
    #     print(f"Queue size: {queue_size}, Image size: {H}x{W}")
    #     print(f"Expected memory usage: {((queue_size * H * W * 2) + (queue_size * H * W * 3*4)) / 1024 / 1024 :.2f} MB")

    #     result = benchmark(benchmark_process_frame, (q_in, q_out, test_pattern), n_repeat=100, n_warmup=1)
    #     print(f"Average time per frame [CPU]: {np.mean(result.cpu_times)*1000*1000 /queue_size:.2f} µs")
    #     print(f"Average time per frame [GPU]: {np.mean(result.gpu_times)*1000*1000 /queue_size:.2f} µs")
    #     print(result)
    

    img_gen_thread = threading.Thread(target=run_fixed_rate, args=(55, put_frame, q_in, test_pattern), daemon=True)
    img_gen_thread.start()

    img_grab_thread = threading.Thread(target=run_fixed_rate, args=(55, process_frame2, q_in, q_out), daemon=True)
    img_grab_thread.start()

    img_display_thread = threading.Thread(target=run_fixed_rate, args=(55, display_frame, q_out), daemon=True)
    img_display_thread.start()


    while True:
        time.sleep(1)
    

    