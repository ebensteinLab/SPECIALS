import numpy as np
import cupy as cp


def get_crop_basis(psfs):
    psfs_t = np.transpose(psfs, (2, 0, 1))
    psfs_t /= np.max(psfs_t, axis=(1, 2), keepdims=True)
    psf_dy, psf_dx = np.gradient(psfs_t, axis=(1, 2))
    psf_dy /= np.max(psf_dy, axis=(1, 2), keepdims=True)
    psf_dx /= np.max(psf_dx, axis=(1, 2), keepdims=True)
    basis_images = np.concatenate((psfs_t, psf_dx, psf_dy), axis=0)
    return basis_images


def get_crop_basis_no_grad(psfs):
    psfs_t = np.transpose(psfs, (2, 0, 1))
    psfs_t /= np.max(psfs_t, axis=(1, 2), keepdims=True)
    basis_images = psfs_t
    return basis_images


def sparse_deconvolution_ista(ism_image, psf, lambda_reg=0.1, num_iterations=50):
    """
    Perform sparse deconvolution using ISTA with sparsity constraint.

    Args:
        ism_image (numpy.ndarray): Observed ISM image of shape (height, width, num_channels).
        psf (numpy.ndarray): Point spread function (PSF), shape (9, 19).
        lambda_reg (float): Regularization parameter controlling sparsity.
        num_iterations (int): Number of iterations for ISTA.

    Returns:
        numpy.ndarray: Reconstructed sparse image, shape (height, width, num_channels).
    """
    ism_image = cp.asarray(ism_image)
    psf = cp.asarray(psf)

    # Dimensions
    height, width, num_channels = ism_image.shape

    # Pad the PSF to match the ISM image dimensions
    psf_padded = cp.zeros((height, width), dtype=psf.dtype)
    psf_padded[
        height // 2 - psf.shape[0] // 2 : height // 2 + psf.shape[0] // 2 + 1,
        width // 2 - psf.shape[1] // 2 : width // 2 + psf.shape[1] // 2 + 1,
    ] = psf

    # Fourier transform of the PSF
    psf_ft = cp.fft.fft2(psf_padded)
    psf_ft_conj = cp.conj(psf_ft)

    # Compute the step size for ISTA
    lipschitz_constant = cp.max(cp.abs(psf_ft) ** 2)
    step_size = 1.0 / lipschitz_constant

    # Initialize the reconstructed image
    reconstructed_image = cp.zeros_like(ism_image)

    # ISTA iterations
    for iteration in range(num_iterations):
        for channel in range(num_channels):
            # Compute the gradient of the data fidelity term
            current_estimate_ft = cp.fft.fft2(reconstructed_image[:, :, channel])
            fidelity_gradient_ft = psf_ft_conj * (
                current_estimate_ft * psf_ft - cp.fft.fft2(ism_image[:, :, channel])
            )
            fidelity_gradient = cp.fft.ifft2(fidelity_gradient_ft).real

            # Update the reconstructed image with gradient descent
            updated_image = (
                reconstructed_image[:, :, channel] - step_size * fidelity_gradient
            )

            # Apply soft thresholding for sparsity
            reconstructed_image[:, :, channel] = cp.sign(updated_image) * cp.maximum(
                cp.abs(updated_image) - lambda_reg * step_size, 0
            )

        # Optionally monitor convergence (loss, PSNR, etc.)

    return cp.asnumpy(reconstructed_image)


def gpu_nnls(A, B, num_iter=50, tol=1e-4):
    """
    Solve the non-negative least squares problem using a gradient projection method on the GPU.
    Args:
        A (cp.ndarray): Matrix of size (m, n), where m is the number of equations, n is the number of variables.
        B (cp.ndarray): Target matrix of size (batch_size, m).
        num_iter (int): Number of iterations.
        tol (float): Convergence tolerance for the residual norm.
    Returns:
        cp.ndarray: Solution matrix of size (batch_size, n).
    """
    import cupy as cp

    batch_size, m = B.shape
    n = A.shape[1]

    # Initialize solutions (batch_size, n)
    X = cp.zeros((batch_size, n), dtype=A.dtype)

    # Precompute A^T A and A^T B for efficiency
    AtA = A.T @ A  # (n, n)
    AtB = A.T @ B.T  # (n, batch_size)

    for _ in range(num_iter):
        # Compute the gradient: grad = A^T(Ax - b)
        grad = AtA @ X.T - AtB  # (n, batch_size)

        # Update the solution using gradient descent
        X -= grad.T  # Update (batch_size, n)

        # Project onto non-negative orthant
        X = cp.maximum(X, 0)

        # Check convergence (optional)
        residuals = cp.linalg.norm(B.T - A @ X.T, axis=0)  # (batch_size,)
        if cp.all(residuals < tol):
            break

    return X
