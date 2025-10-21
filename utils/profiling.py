import time
import threading
from collections import deque
from typing import Dict, Tuple


class PerformanceProfiler:
    """Lightweight profiler for tracking operation timings and frequencies."""
    
    def __init__(self, window_size: int = 100):
        """
        Args:
            window_size: Number of samples to keep for rolling averages
        """
        self.window_size = window_size
        self._timings: Dict[str, deque] = {}
        self._counts: Dict[str, int] = {}
        self._last_reset: Dict[str, float] = {}
        self._lock = threading.Lock()
        
    def record_timing(self, operation: str, duration_ms: float):
        """Record a timing measurement for an operation."""
        with self._lock:
            if operation not in self._timings:
                self._timings[operation] = deque(maxlen=self.window_size)
                self._counts[operation] = 0
                self._last_reset[operation] = time.time()
            
            self._timings[operation].append(duration_ms)
            self._counts[operation] += 1
    
    def get_stats(self, operation: str) -> Tuple[float, float, float, float, int, float, float]:
        """
        Get statistics for an operation.
        
        Returns:
            (avg_ms, min_ms, max_ms, ops_per_sec, sample_count, p50_ms, p95_ms)
        """
        with self._lock:
            if operation not in self._timings or not self._timings[operation]:
                return (0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0)
            
            timings = sorted(list(self._timings[operation]))
            n = len(timings)
            avg_ms = sum(timings) / n
            min_ms = timings[0]
            max_ms = timings[-1]
            
            # Calculate percentiles
            p50_idx = n // 2
            p95_idx = int(n * 0.95)
            p50_ms = timings[p50_idx] if n > 0 else 0.0
            p95_ms = timings[p95_idx] if n > 0 else 0.0
            
            # Calculate operations per second based on count since last reset
            elapsed = time.time() - self._last_reset[operation]
            if elapsed > 0:
                ops_per_sec = self._counts[operation] / elapsed
            else:
                ops_per_sec = 0.0
                
            return (avg_ms, min_ms, max_ms, ops_per_sec, n, p50_ms, p95_ms)
    
    def reset_counters(self, operation: str = None):
        """Reset counters for frequency calculation (keeps timing history)."""
        with self._lock:
            if operation:
                if operation in self._counts:
                    self._counts[operation] = 0
                    self._last_reset[operation] = time.time()
            else:
                # Reset all
                for op in self._counts:
                    self._counts[op] = 0
                    self._last_reset[op] = time.time()
    
    def reset_all(self, operation: str = None):
        """Reset all profiling data including timing samples (min/max/avg) and counters."""
        with self._lock:
            if operation:
                if operation in self._timings:
                    self._timings[operation].clear()
                if operation in self._counts:
                    self._counts[operation] = 0
                    self._last_reset[operation] = time.time()
            else:
                # Reset all operations
                for op in list(self._timings.keys()):
                    self._timings[op].clear()
                for op in self._counts:
                    self._counts[op] = 0
                    self._last_reset[op] = time.time()
    
    def clear(self):
        """Clear all profiling data completely (removes all operations)."""
        with self._lock:
            self._timings.clear()
            self._counts.clear()
            self._last_reset.clear()


class TimingContext:
    """Context manager for timing operations."""
    
    def __init__(self, profiler: PerformanceProfiler, operation: str, use_cuda_sync: bool = False):
        self.profiler = profiler
        self.operation = operation
        self.use_cuda_sync = use_cuda_sync
        self.start_time = None
        
    def __enter__(self):
        if self.use_cuda_sync:
            import cupy as cp
            cp.cuda.Stream.null.synchronize()
        self.start_time = time.perf_counter()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.use_cuda_sync:
            import cupy as cp
            cp.cuda.Stream.null.synchronize()
        duration_ms = (time.perf_counter() - self.start_time) * 1000.0
        self.profiler.record_timing(self.operation, duration_ms)
        return False

