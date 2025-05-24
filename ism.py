from compute_backend import backend
from utils import CROP_SIZE
import numpy as np


def run_ism(all_convolved_crops, M, peaks, red_x_location):
    """
    Generate final image using ISM from the convolved crops.
    Automatically uses GPU if available, falls back to CPU otherwise.
    """
    example_image = np.asarray(all_convolved_crops[0])
    single_image_memory = example_image.nbytes

    num_images = all_convolved_crops.shape[0]

    im_size_M = int(512 * M * 2)
    num_channels = all_convolved_crops.shape[3]
    ism_rounded_peaks = backend.round(backend.asarray(peaks) * 2 * M)
    window_y = slice(CROP_SIZE[0])
    window_x = slice(CROP_SIZE[1])

    rows = (
        ism_rounded_peaks[:, 1, None, None, None]
        + backend.arange(-int(CROP_SIZE[0] * M // 2), int(CROP_SIZE[0] * M // 2))[
            None, :, None, None
        ]
    )
    cols = (
        ism_rounded_peaks[:, 0, None, None, None]
        + backend.arange(
            -int(red_x_location * M), int((CROP_SIZE[1] - red_x_location) * M)
        )[None, None, :, None]
    )
    channels = (
        backend.zeros((ism_rounded_peaks.shape[0], 1, 1, 1))
        + backend.arange(num_channels)[None, None, None, :]
    )
    rows, cols, channels = rows.astype(int), cols.astype(int), channels.astype(int)

    # Memory management - different approaches for GPU vs CPU
    free_mem, total_mem = backend.get_memory_info()

    if backend.use_gpu and free_mem is not None:
        usable_memory = free_mem * 0.8
        # Estimate optimal batch size for GPU
        memory_per_pair = 2 * single_image_memory * (M**2) * num_channels
        batch_size = max(1, int(usable_memory // memory_per_pair))
        print(
            f"Available GPU memory: {free_mem / 1e6:.2f} MB, Using batch size: {batch_size}"
        )
    else:
        # For CPU, use a more conservative batch size based on data size
        if free_mem is not None:
            usable_memory = free_mem * 0.3  # More conservative for CPU
            memory_per_pair = 2 * single_image_memory * (M**2) * num_channels
            batch_size = max(1, int(usable_memory // memory_per_pair))
            print(
                f"Available CPU memory: {free_mem / 1e6:.2f} MB, Using batch size: {batch_size}"
            )
        else:
            # Fallback batch size if memory info not available
            batch_size = max(1, min(100, num_images // 10))
            print(
                f"Memory info not available, using conservative batch size: {batch_size}"
            )

    accum = backend.zeros((im_size_M, im_size_M, num_channels))

    batch_start = 0
    while batch_start < num_images:
        batch_end = min(batch_start + batch_size, num_images)
        backend.free_memory_pool()  # Free memory pool (GPU only)

        try:
            batch_idx = slice(batch_start, batch_end)
            resized_crops = backend.zoom(
                backend.asarray(
                    all_convolved_crops[batch_idx, window_y, window_x, :],
                    dtype=np.float32,
                ),
                (1, M, M, 1),
                order=1,
            )
            resized_crops = backend.clip(resized_crops, 0, None)

            backend.scatter_add(
                accum,
                (rows[batch_idx], cols[batch_idx], channels[batch_idx]),
                resized_crops,
            )

            del resized_crops
            backend.free_memory_pool()  # Free memory pool (GPU only)

            batch_start = batch_end

        except Exception as e:
            if backend.handle_out_of_memory_error(e):
                # If out of memory, reduce batch size
                batch_size = max(1, batch_size // 2)
                print(f"Reduced batch size to {batch_size} due to memory constraints.")
                if batch_size == 1 and batch_start == batch_end - 1:
                    # If we can't even process a single image, skip it
                    print(f"Skipping image {batch_start} due to memory constraints.")
                    batch_start += 1
            else:
                # Re-raise other exceptions
                raise e

    final_ism = backend.zoom(accum, (1 / M, 1 / M, 1), order=1)
    return backend.asnumpy(final_ism)
