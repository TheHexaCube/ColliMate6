
from logging import INFO
import weakref
import numpy as np
import time
import os
SCALE = 1.0 / 4096.0

os.environ["NUMBA_NUM_THREADS"] = "8"

import numba



@numba.njit(parallel=True, nogil=True, fastmath=False)
def raw_to_mono(src):
    """
    Converts a raw BayerRG12 image to a single-channel (monochrome) float32 image using a simple demosaicing algorithm.
    
    Args:
        src (np.ndarray): 2D numpy array of shape (H, W) containing raw BayerRG12 image data.
        
    Returns:
        np.ndarray: 3D numpy array of shape (H, W) in float32, normalized to [0, 1],
        where each pixel represents the computed green value (using simple interpolation for RG and GB sites).
    
    Notes:
        - Edges are replicated to handle border conditions.
        - The output image is normalized using SCALE = 1.0 / 4096.0.
        - Implemented with Numba for performance.
    """
    h, w = src.shape
    out = np.empty((h, w), np.float32)
    for y in numba.prange(h):
        ye = (y & 1) == 0
        y0 = max(y - 1, 0)
        y1 = min(y + 1, h - 1)
        for x in range(w):
            xe = (x & 1) == 0
            x0 = max(x - 1, 0)
            x1 = min(x + 1, w - 1)
            if ye == xe:
                g = 0.25 * (src[y0, x] + src[y1, x] + src[y, x0] + src[y, x1])
            else:
                g = src[y, x]
            out[y, x] = g * SCALE
    return out


@numba.njit(parallel=True, nogil=True, fastmath=False)
def average_frames(frames, count):
    """
    Fast average of first `count` frames from 3D array (n,h,w), float32 contiguous.
    """
    n = len(frames)
    h, w = frames[0].shape
    if count > n:
        count = n

    out = np.empty((h, w), np.float32)
    inv_count = 1.0 / count

    for y in numba.prange(h):
        for x in range(w):
            s = 0.0
            for i in range(count):
                s += frames[i][y, x]
            out[y, x] = s * inv_count

    return out
    


@numba.njit(parallel=True, nogil=True, fastmath=False)
def mono_to_rgb(src):
    """
    Converts a single-channel (monochrome) float32 image to a three-channel (RGB) float32 image.
    
    Args:
        src (np.ndarray): 2D numpy array of shape (H, W) containing monochrome float32 image data.
        
    Returns:
        np.ndarray: 1D numpy array of shape (H*W*3) in float32
    """
    h, w = src.shape
    out = np.empty((h, w, 3), np.float32)
    for y in numba.prange(h):
        for x in range(w):
            v = src[y, x]
            out[y, x, 0] = v
            out[y, x, 1] = v
            out[y, x, 2] = v
    return out
    

def precompile_functions():
    print("Precompiling raw_to_mono function...")
    dummy_frame = np.zeros((1080, 1920), np.uint16)
    start_time = time.time()
    raw_to_mono(dummy_frame)
    end_time = time.time()
    print(f"Raw to mono function precompiled in { (end_time - start_time)*1000:.2f} ms")

    print("Precompiling average_frames function...")
    start_time = time.time()
    dummy_frames = np.zeros((10, 1080, 1920), np.float32)
    average_frames(dummy_frames, 10)
    end_time = time.time()
    print(f"Average frames function precompiled in { (end_time - start_time)*1000:.2f} ms")

    print("Precompiling mono_to_rgb function...")
    start_time = time.time()
    dummy_frame = np.zeros((1080, 1920), np.float32)
    mono_to_rgb(dummy_frame)
    end_time = time.time()
    print(f"Mono to rgb function precompiled in { (end_time - start_time)*1000:.2f} ms")
