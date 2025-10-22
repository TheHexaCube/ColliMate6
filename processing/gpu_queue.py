import cupy as cp
from queue import Full, Empty
import threading

class GPUQueue:
    def __init__(self, max_size: int, shape: tuple, dtype=cp.float32):
        self.max_size = max_size
        self.shape = shape
        self.dtype = dtype
        self.queue = cp.zeros((max_size, *shape), dtype=dtype)
        self.head = 0
        self.tail = 0
        self.size = 0
        self._lock = threading.Lock()

    def put(self, item: cp.ndarray):
        with self._lock:
            if self.size == self.max_size:
                raise Full
            # direct copy into queue slot
            self.queue[self.tail][:] = item.astype(self.dtype, copy=False)
            self.tail = (self.tail + 1) % self.max_size
            self.size += 1

    def put_overwrite(self, item: cp.ndarray) -> bool:
        """
        Put item into the queue; if full, overwrite the oldest frame.
        Returns True if an old frame was overwritten.
        """
        with self._lock:
            overwritten = self.size == self.max_size
            # copy into current tail position
            self.queue[self.tail][:] = item.astype(self.dtype, copy=False)
            if overwritten:
                # advance head to drop the oldest
                self.head = (self.head + 1) % self.max_size
            else:
                self.size += 1
            self.tail = (self.tail + 1) % self.max_size
            return overwritten

    def get_view(self):
        with self._lock:
            if self.size == 0:
                raise Empty
            idx = self.head
            self.head = (self.head + 1) % self.max_size
            self.size -= 1
            return self.queue[idx]

    def get_queue_view(self):
        return self.queue

    def discard_frame(self):        
        with self._lock:
            if self.size == 0:
                raise Empty
            self.head = (self.head + 1) % self.max_size
            self.size -= 1

    def is_empty(self):
        with self._lock:
            return self.size == 0

    def is_full(self):
        with self._lock:
            return self.size == self.max_size

    def clear(self):
        with self._lock:
            self.head = 0
            self.tail = 0
            self.size = 0

    def snapshot_state(self):
        """Atomically snapshot (head, tail, size)."""
        with self._lock:
            return self.head, self.tail, self.size
