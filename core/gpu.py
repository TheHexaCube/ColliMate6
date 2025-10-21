import cupy as cp
from queue import Full, Empty

class GPUQueue:
    def __init__(self, max_size: int, shape: tuple, dtype=cp.float32):
        self.max_size = max_size
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
        self.tail = (self.tail + 1) % self.max_size
        self.size += 1

    def get_view(self):
        if self.is_empty():
            raise Empty
        idx = self.head
        self.head = (self.head + 1) % self.max_size
        self.size -= 1
        return self.queue[idx]

    def get_queue_view(self):
        return self.queue

    def discard_frame(self):        
        if self.is_empty():
            raise Empty
        self.head = (self.head + 1) % self.max_size
        self.size -= 1

    def is_empty(self): return self.size == 0
    def is_full(self): return self.size == self.max_size
    def clear(self): self.head = self.tail = self.size = 0


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


def run_demosaic(src: cp.ndarray, dst: cp.ndarray):
    h, w = src.shape
    threads = (16, 16)
    blocks = ((w + 15) // 16, (h + 15) // 16)
    kernel_demosaic(blocks, threads, (src, dst, w, h))


def average_gpuqueue(q: "GPUQueue", out: cp.ndarray):
    """
    Average all valid frames in GPUQueue into `out` (on-GPU, in place).
    Expects q.queue shape = (max_size, h, w[, c]) and dtype float32.
    """
    if q.is_empty():
        raise ValueError("Queue is empty")

    n = q.size
    h, w = q.shape[:2]
    c = q.shape[2] if len(q.shape) == 3 else 1
    size = h * w * c
    threads = 256
    blocks = (size + threads - 1) // threads

    print(f"Average GPUQueue: {n} frames")

    kernel_avg_q = cp.RawKernel(r'''
    extern "C" __global__
    void avg_gpuqueue(const float* frames, float* out,
                      const int max_size, const int n,
                      const int head, const int h, const int w, const int c)
    {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        int size = h * w * c;
        if (idx >= size) return;

        float s = 0.0f;
        for (int i = 0; i < n; ++i) {
            int q_idx = (head + i) % max_size;
            s += frames[q_idx * size + idx];
        }
        out[idx] = s / n;
    }
    ''', "avg_gpuqueue")

    kernel_avg_q(
        (blocks,), (threads,),
        (q.queue, out, q.max_size, n, q.head, h, w, c)
    )
    cp.cuda.Device().synchronize()