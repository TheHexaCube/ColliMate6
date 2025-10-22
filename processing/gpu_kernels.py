import cupy as cp
import numpy as np

# ============================================================================
# Demosaic Kernel
# ============================================================================

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

# ============================================================================
# Averaging Kernel
# ============================================================================

kernel_avg_q_windowed = cp.RawKernel(r'''
extern "C" __global__
void avg_gpuqueue_windowed(const float* frames, float* out,
                           const int max_size, const int count,
                           const int tail, const int h, const int w, const int c)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int size = h * w * c;
    if (idx >= size) return;

    float s = 0.0f;
    // accumulate last `count` frames from tail-1 backwards
    for (int i = 0; i < count; ++i) {
        int q_idx = ( (tail - 1 - i) % max_size + max_size ) % max_size; // safe mod
        s += frames[q_idx * size + idx];
    }
    out[idx] = s / (float)count;
}
''', 'avg_gpuqueue_windowed')


def average_gpuqueue_windowed(q, out: cp.ndarray, window: int) -> int:
    """
    Average last `min(window, size)` frames in GPUQueue into `out` (on-GPU).
    Returns the number of frames used for averaging.
    """
    head, tail, size = q.snapshot_state()
    if size == 0:
        raise ValueError("Queue is empty")

    h, w = q.shape[:2]
    c = q.shape[2] if len(q.shape) == 3 else 1
    count = int(min(window, size))

    total_elems = h * w * c
    threads = 256
    blocks = (total_elems + threads - 1) // threads

    kernel_avg_q_windowed(
        (blocks,), (threads,),
        (q.queue, out, q.max_size, count, tail, h, w, c)
    )

    return count

# ============================================================================
# Overlay Drawing Kernels
# ============================================================================

kernel_draw_line = cp.RawKernel(r'''
extern "C" __global__
void draw_line(float* overlay, const int width, const int height,
               const int x0, const int y0, const int x1, const int y1,
               const float r, const float g, const float b, const float a,
               const int thickness)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total = width * height;
    if (idx >= total) return;
    
    int px = idx % width;
    int py = idx / width;
    
    // Bresenham distance to line segment
    int dx = x1 - x0;
    int dy = y1 - y0;
    
    // Compute perpendicular distance from point to line
    float len_sq = (float)(dx*dx + dy*dy);
    if (len_sq < 1.0f) {
        // Degenerate line (point)
        int dist = abs(px - x0) + abs(py - y0);
        if (dist <= thickness) {
            int offset = idx * 4;
            overlay[offset + 0] = r;
            overlay[offset + 1] = g;
            overlay[offset + 2] = b;
            overlay[offset + 3] = a;
        }
        return;
    }
    
    // Project point onto line segment
    float t = ((px - x0) * dx + (py - y0) * dy) / len_sq;
    t = fmaxf(0.0f, fminf(1.0f, t));
    
    float proj_x = x0 + t * dx;
    float proj_y = y0 + t * dy;
    
    float dist = sqrtf((px - proj_x)*(px - proj_x) + (py - proj_y)*(py - proj_y));
    
    if (dist <= (float)thickness * 0.5f) {
        int offset = idx * 4;
        overlay[offset + 0] = r;
        overlay[offset + 1] = g;
        overlay[offset + 2] = b;
        overlay[offset + 3] = a;
    }
}
''', 'draw_line')


kernel_draw_circle = cp.RawKernel(r'''
extern "C" __global__
void draw_circle(float* overlay, const int width, const int height,
                 const int cx, const int cy, const float radius,
                 const float r, const float g, const float b, const float a,
                 const int thickness, const int filled)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total = width * height;
    if (idx >= total) return;
    
    int px = idx % width;
    int py = idx / width;
    
    float dx = (float)(px - cx);
    float dy = (float)(py - cy);
    float dist = sqrtf(dx*dx + dy*dy);
    
    int draw = 0;
    if (filled) {
        draw = (dist <= radius);
    } else {
        float inner = radius - (float)thickness * 0.5f;
        float outer = radius + (float)thickness * 0.5f;
        draw = (dist >= inner && dist <= outer);
    }
    
    if (draw) {
        int offset = idx * 4;
        overlay[offset + 0] = r;
        overlay[offset + 1] = g;
        overlay[offset + 2] = b;
        overlay[offset + 3] = a;
    }
}
''', 'draw_circle')


kernel_draw_cross = cp.RawKernel(r'''
extern "C" __global__
void draw_cross(float* overlay, 
                int width, int height, int cx, int cy, int thickness,
                float size, float r, float g, float b, float a, float rotation)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total = width * height;
    if (idx >= total) return;
    
    int px = idx % width;
    int py = idx / width;
    
    // Distance from center
    int dx = px - cx;
    int dy = py - cy;
    
    // Rotate point
    float cos_r = cosf(-rotation);
    float sin_r = sinf(-rotation);
    float rx = (float)dx * cos_r - (float)dy * sin_r;
    float ry = (float)dx * sin_r + (float)dy * cos_r;
    
    int abs_rx = (int)((rx < 0.0f) ? -rx : rx);
    int abs_ry = (int)((ry < 0.0f) ? -ry : ry);
    
    int isize = (int)size;
    int half_thick = thickness / 2;
    
    // Horizontal arm (along rotated x-axis)
    int on_horizontal = (abs_ry <= half_thick) && (abs_rx <= isize);
    
    // Vertical arm (along rotated y-axis)
    int on_vertical = (abs_rx <= half_thick) && (abs_ry <= isize);
    
    if (on_horizontal || on_vertical) {
        int offset = idx * 4;
        overlay[offset + 0] = r;
        overlay[offset + 1] = g;
        overlay[offset + 2] = b;
        overlay[offset + 3] = a;
    }
}
''', 'draw_cross')


kernel_alpha_composite = cp.RawKernel(r'''
extern "C" __global__
void alpha_composite(const float* rgb, const float* overlay, float* out,
                     const int width, const int height)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total = width * height;
    if (idx >= total) return;
    
    int rgb_offset = idx * 3;
    int rgba_offset = idx * 4;
    
    float alpha = overlay[rgba_offset + 3];
    
    if (alpha > 0.0f) {
        float inv_alpha = 1.0f - alpha;
        out[rgb_offset + 0] = rgb[rgb_offset + 0] * inv_alpha + overlay[rgba_offset + 0] * alpha;
        out[rgb_offset + 1] = rgb[rgb_offset + 1] * inv_alpha + overlay[rgba_offset + 1] * alpha;
        out[rgb_offset + 2] = rgb[rgb_offset + 2] * inv_alpha + overlay[rgba_offset + 2] * alpha;
    } else {
        // No overlay, just copy RGB
        out[rgb_offset + 0] = rgb[rgb_offset + 0];
        out[rgb_offset + 1] = rgb[rgb_offset + 1];
        out[rgb_offset + 2] = rgb[rgb_offset + 2];
    }
}
''', 'alpha_composite')

# ============================================================================
# Python wrapper functions for drawing primitives
# ============================================================================

def draw_line(overlay: cp.ndarray, x0: int, y0: int, x1: int, y1: int,
              color: tuple, thickness: int):
    """Draw a line into RGBA overlay buffer."""
    h, w = overlay.shape[:2]
    total = h * w
    threads = 256
    blocks = (total + threads - 1) // threads
    
    r, g, b, a = color
    # Use ravel to get a contiguous 1D view
    overlay_1d = overlay.ravel()
    kernel_draw_line(
        (blocks,), (threads,),
        (overlay_1d, np.int32(w), np.int32(h), np.int32(x0), np.int32(y0), np.int32(x1), np.int32(y1), 
         np.float32(r), np.float32(g), np.float32(b), np.float32(a), np.int32(thickness))
    )


def draw_circle(overlay: cp.ndarray, cx: int, cy: int, radius: float,
                color: tuple, thickness: int, filled: bool):
    """Draw a circle into RGBA overlay buffer."""
    h, w = overlay.shape[:2]
    total = h * w
    threads = 256
    blocks = (total + threads - 1) // threads
    
    r, g, b, a = color
    # Use ravel to get a contiguous 1D view
    overlay_1d = overlay.ravel()
    kernel_draw_circle(
        (blocks,), (threads,),
        (overlay_1d, np.int32(w), np.int32(h), np.int32(cx), np.int32(cy), np.float32(radius), 
         np.float32(r), np.float32(g), np.float32(b), np.float32(a), np.int32(thickness), np.int32(int(filled)))
    )


def draw_cross(overlay: cp.ndarray, cx: int, cy: int, size: float,
               color: tuple, thickness: int, rotation: float):
    """Draw a rotated cross into RGBA overlay buffer."""
    h, w = overlay.shape[:2]
    total = h * w
    threads = 256
    blocks = (total + threads - 1) // threads
    
    r, g, b, a = color
    # Use ravel to get a contiguous 1D view
    overlay_1d = overlay.ravel()
    # Parameter order: ints first, then floats (for proper CUDA alignment)
    # CRITICAL: Must use np.int32() and np.float32() for proper CUDA parameter passing!
    kernel_draw_cross(
        (blocks,), (threads,),
        (overlay_1d, np.int32(w), np.int32(h), np.int32(cx), np.int32(cy), np.int32(thickness),
         np.float32(size), np.float32(r), np.float32(g), np.float32(b), np.float32(a), np.float32(rotation))
    )


def composite_overlay(rgb_frame: cp.ndarray, overlay: cp.ndarray, out: cp.ndarray):
    """Alpha-composite RGBA overlay onto RGB frame."""
    h, w = rgb_frame.shape[:2]
    total = h * w
    threads = 256
    blocks = (total + threads - 1) // threads
    
    kernel_alpha_composite(
        (blocks,), (threads,),
        (rgb_frame, overlay, out, w, h)
    )
