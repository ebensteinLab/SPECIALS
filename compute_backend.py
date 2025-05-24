# gpu_cpu_compat.py
"""
Compatibility layer for GPU/CPU operations.
Automatically detects GPU availability and provides unified interface.
"""

import numpy as np
from scipy.ndimage import zoom, convolve as scipy_convolve
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.signal import fftconvolve as scipy_fftconvolve

# Try to import GPU libraries
try:
    import cupy as cp
    import cupyx
    from cupyx.scipy.fft import fft2 as gpu_fft2, ifft2 as gpu_ifft2
    from cupyx.scipy.fft import fftshift as gpu_fftshift, ifftshift as gpu_ifftshift
    from cupyx.scipy.ndimage import convolve as gpu_convolve
    from cupyx.scipy.signal import fftconvolve as gpu_fftconvolve

    # Test GPU availability
    try:
        cp.cuda.Device(0).compute_capability
        GPU_AVAILABLE = True
        print("GPU acceleration available")
    except:
        GPU_AVAILABLE = False
        print("GPU hardware not available, falling back to CPU")

except ImportError:
    GPU_AVAILABLE = False
    print("CuPy not installed, using CPU operations")


class ComputeBackend:
    """Unified interface for GPU/CPU operations"""

    def __init__(self):
        self.use_gpu = GPU_AVAILABLE

    def asarray(self, array, dtype=None):
        """Convert array to appropriate backend format"""
        if self.use_gpu:
            return cp.asarray(array, dtype=dtype)
        else:
            return np.asarray(array, dtype=dtype)

    def asnumpy(self, array):
        """Convert array back to numpy"""
        if self.use_gpu and hasattr(array, 'get'):
            return array.get()
        else:
            return np.asarray(array)

    def zeros(self, shape, dtype=None):
        """Create zeros array"""
        if self.use_gpu:
            return cp.zeros(shape, dtype=dtype)
        else:
            return np.zeros(shape, dtype=dtype)

    def zeros_like(self, array):
        """Create zeros array with same shape and type"""
        if self.use_gpu:
            return cp.zeros_like(array)
        else:
            return np.zeros_like(array)

    def full(self, shape, fill_value, dtype=None):
        """Create filled array"""
        if self.use_gpu:
            return cp.full(shape, fill_value, dtype=dtype)
        else:
            return np.full(shape, fill_value, dtype=dtype)

    def concatenate(self, arrays, axis=0):
        """Concatenate arrays"""
        if self.use_gpu:
            return cp.concatenate(arrays, axis=axis)
        else:
            return np.concatenate(arrays, axis=axis)

    # Mathematical operations
    def conj(self, array):
        if self.use_gpu:
            return cp.conj(array)
        else:
            return np.conj(array)

    def abs(self, array):
        if self.use_gpu:
            return cp.abs(array)
        else:
            return np.abs(array)

    def max(self, array):
        if self.use_gpu:
            return cp.max(array)
        else:
            return np.max(array)

    def maximum(self, a, b):
        if self.use_gpu:
            return cp.maximum(a, b)
        else:
            return np.maximum(a, b)

    def sign(self, array):
        if self.use_gpu:
            return cp.sign(array)
        else:
            return np.sign(array)

    def flip(self, array, axis=None):
        if self.use_gpu:
            return cp.flip(array, axis=axis)
        else:
            return np.flip(array, axis=axis)

    def clip(self, array, a_min, a_max, out=None):
        if self.use_gpu:
            return cp.clip(array, a_min, a_max, out=out)
        else:
            return np.clip(array, a_min, a_max, out=out)

    def sum(self, array):
        if self.use_gpu:
            return cp.sum(array)
        else:
            return np.sum(array)

    def where(self, condition, x, y):
        if self.use_gpu:
            return cp.where(condition, x, y)
        else:
            return np.where(condition, x, y)

    def finfo(self, dtype):
        if self.use_gpu:
            return cp.finfo(dtype)
        else:
            return np.finfo(dtype)

    # FFT operations
    def fft2(self, array):
        if self.use_gpu:
            return gpu_fft2(array)
        else:
            return fft2(array)

    def ifft2(self, array):
        if self.use_gpu:
            return gpu_ifft2(array)
        else:
            return ifft2(array)

    def fftshift(self, array):
        if self.use_gpu:
            return gpu_fftshift(array)
        else:
            return fftshift(array)

    def ifftshift(self, array):
        if self.use_gpu:
            return gpu_ifftshift(array)
        else:
            return ifftshift(array)

    def fftconvolve(self, in1, in2, mode='full', axes=None):
        if self.use_gpu:
            return gpu_fftconvolve(in1, in2, mode=mode, axes=axes)
        else:
            return scipy_fftconvolve(in1, in2, mode=mode, axes=axes)

    def convolve(self, input_array, weights):
        if self.use_gpu:
            return gpu_convolve(input_array, weights)
        else:
            return scipy_convolve(input_array, weights)

    def scatter_add(self, target, indices, source):
        """Scatter add operation with CPU fallback"""
        if self.use_gpu:
            cupyx.scatter_add(target, indices, source)
        else:
            # CPU fallback for scatter_add
            rows, cols = indices
            for i in range(len(source)):
                target[rows[i], cols[i]] += source[i]

    def mean(self, array, axis=None):
        """Compute mean along specified axis"""
        if self.use_gpu:
            return cp.mean(array, axis=axis)
        else:
            return np.mean(array, axis=axis)

    def var(self, array, axis=None):
        """Compute variance along specified axis"""
        if self.use_gpu:
            return cp.var(array, axis=axis)
        else:
            return np.var(array, axis=axis)

    def ones_like(self, array, dtype=None):
        """Create ones array with same shape"""
        if self.use_gpu:
            return cp.ones_like(array, dtype=dtype)
        else:
            return np.ones_like(array, dtype=dtype)

    def gaussian_filter1d(self, input_array, sigma, axis=-1):
        """1D Gaussian filter"""
        if self.use_gpu:
            from cupyx.scipy.ndimage import gaussian_filter1d as gpu_gaussian_filter1d
            return gpu_gaussian_filter1d(input_array, sigma=sigma, axis=axis)
        else:
            from scipy.ndimage import gaussian_filter1d
            return gaussian_filter1d(input_array, sigma=sigma, axis=axis)

    def round(self, array):
        """Round array elements"""
        if self.use_gpu:
            return cp.round(array)
        else:
            return np.round(array)

    def arange(self, *args, **kwargs):
        """Create range array"""
        if self.use_gpu:
            return cp.arange(*args, **kwargs)
        else:
            return np.arange(*args, **kwargs)

    def zoom(self, input_array, zoom_factor, order=1):
        """Zoom/resize array"""
        if self.use_gpu:
            from cupyx.scipy.ndimage import zoom as gpu_zoom
            return gpu_zoom(input_array, zoom_factor, order=order)
        else:
            from scipy.ndimage import zoom
            return zoom(input_array, zoom_factor, order=order)

    def get_memory_info(self):
        """Get available memory information"""
        if self.use_gpu:
            try:
                free_mem, total_mem = cp.cuda.runtime.memGetInfo()
                return free_mem, total_mem
            except:
                # Fallback if CUDA not available
                return None, None
        else:
            # For CPU, estimate available RAM (simplified)
            import psutil
            try:
                virtual_memory = psutil.virtual_memory()
                return virtual_memory.available, virtual_memory.total
            except:
                return None, None

    def free_memory_pool(self):
        """Free memory pool (GPU only operation)"""
        if self.use_gpu:
            try:
                mempool = cp._default_memory_pool
                mempool.free_all_blocks()
            except:
                pass  # Silently fail if memory pool not available
        # No-op for CPU

    def handle_out_of_memory_error(self, exception):
        """Check if exception is out of memory error"""
        if self.use_gpu:
            return isinstance(exception, cp.cuda.memory.OutOfMemoryError)
        else:
            # For CPU, check for MemoryError
            return isinstance(exception, MemoryError)


# Create global backend instance
backend = ComputeBackend()

# Export commonly used functions for convenience
asarray = backend.asarray
asnumpy = backend.asnumpy
zeros = backend.zeros
zeros_like = backend.zeros_like
full = backend.full
concatenate = backend.concatenate