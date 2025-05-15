import numpy as np

CROP_SIZE = (9, 19)
RED_X_LOCATION = int(CROP_SIZE[1] - 4)
RED_Y_LOCATION = np.ceil(CROP_SIZE[0] / 2).astype(int)


def rescale_float32_to_uint16(arr):
    """
    Rescales a NumPy or CuPy array of type float32 to uint16.
    Args:
        arr (ndarray): Input array of type float32.
    Returns:
        ndarray: Rescaled array of type uint16.
    """
    min_val = arr.min()
    max_val = arr.max()

    if max_val == min_val:
        # Handle case where all values are the same
        return np.zeros_like(arr, dtype=np.uint16)

    # Normalize to [0, 1] and scale to [0, 65535]
    rescaled = (arr - min_val) / (max_val - min_val) * 65535
    rescaled = np.clip(rescaled, 0, 65535)  # Ensure values are within range
    return rescaled.astype(np.uint16)


CONTRAST_FACTORS = [0.9, 0.2]


def myimrescale(im, factors=None):
    # Calculate the minimum and maximum values
    if factors is None:
        factors = CONTRAST_FACTORS
    im_min = np.median(im, axis=(0, 1)) * factors[0]
    im_max = np.max(im, axis=(0, 1)) * factors[1]

    # Rescale the image
    im_rescaled = (im - im_min) / (im_max - im_min) * 255
    im_rescaled = np.clip(im_rescaled, 0, 255)
    return im_rescaled.astype(np.uint8)


def flatten(xss):
    return [x for xs in xss for x in xs]


def centers_to_rows_and_columns(centers):
    centers = np.array(centers).astype(int)

    # Precompute all crops at once
    rows = (
        centers[:, 1, None, None]
        + np.arange(RED_Y_LOCATION - CROP_SIZE[0], RED_Y_LOCATION)[None, :, None]
    )
    cols = (
        centers[:, 0, None, None]
        + np.arange(-RED_X_LOCATION, CROP_SIZE[1] - RED_X_LOCATION)[None, None, :]
    )
    return rows, cols
