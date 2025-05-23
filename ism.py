from utils import CROP_SIZE, RED_X_LOCATION
import numpy as np
import cupy as cp
import cupyx


def run_ism(all_convolved_crops, M, peaks, scatter_add_batch_size=100000):
    """
    Generate final image using ISM from the convolved crops.
    """
    example_image = np.asarray(all_convolved_crops[0])
    single_image_memory = example_image.nbytes

    num_images = all_convolved_crops.shape[0]

    im_size_M = int(512 * M * 2)
    num_channels = all_convolved_crops.shape[3]
    ism_rounded_peaks = cp.round(cp.asarray(peaks) * 2 * M)
    window_y = slice(CROP_SIZE[0])
    window_x = slice(CROP_SIZE[1])

    rows = (
        ism_rounded_peaks[:, 1, None, None, None]
        + cp.arange(-int(CROP_SIZE[0] * M // 2), int(CROP_SIZE[0] * M // 2))[
            None, :, None, None
        ]
    )
    cols = (
        ism_rounded_peaks[:, 0, None, None, None]
        + cp.arange(-int(RED_X_LOCATION * M), int((CROP_SIZE[1] - RED_X_LOCATION) * M))[
            None, None, :, None
        ]
    )
    channels = (
        cp.zeros((ism_rounded_peaks.shape[0], 1, 1, 1))
        + cp.arange(num_channels)[None, None, None, :]
    )
    rows, cols, channels = rows.astype(int), cols.astype(int), channels.astype(int)

    mempool = cp._default_memory_pool
    free_mem, _ = cp.cuda.runtime.memGetInfo()
    usable_memory = free_mem * 0.8

    # Estimate optimal batch size
    memory_per_pair = 2 * single_image_memory * (M**2) * num_channels
    batch_size = max(1, int(usable_memory // memory_per_pair))
    print(
        f"Available GPU memory: {free_mem / 1e6:.2f} MB, Using batch size: {batch_size}"
    )
    accum = cp.zeros((im_size_M, im_size_M, num_channels))

    batch_start = 0
    while batch_start < num_images:
        batch_end = min(batch_start + batch_size, num_images)
        mempool.free_all_blocks()
        try:
            batch_idx = slice(batch_start, batch_end)
            resized_crops = cupyx.scipy.ndimage.zoom(
                cp.asarray(
                    all_convolved_crops[batch_idx, window_y, window_x, :],
                    dtype=cp.float32,
                ),
                (1, M, M, 1),
                order=1,
            )
            resized_crops = cp.clip(resized_crops, 0, None)
            # for i in range(0, len(test_crops),scatter_add_batch_size):
            #     batch_idx = slice(i, min(i + scatter_add_batch_size, len(test_crops)))
            cupyx.scatter_add(
                accum,
                (rows[batch_idx], cols[batch_idx], channels[batch_idx]),
                resized_crops,
            )

            del resized_crops
            mempool.free_all_blocks()

            batch_start = batch_end

        except cp.cuda.memory.OutOfMemoryError:
            # If out of memory, reduce batch size
            batch_size = max(1, batch_size // 2)
            print(f"Reduced batch size to {batch_size} due to memory constraints.")

    final_ism = cupyx.scipy.ndimage.zoom(accum, (1 / M, 1 / M, 1), order=1)
    return cp.asnumpy(final_ism)
