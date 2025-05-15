# SPECIALS - Spectral Extraction and Classification for ISM Analysis using Least Squares

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)]()
[![Paper DOI](https://img.shields.io/badge/Paper%20DOI-10.XXXX%2FXXXXXX-brightgreen)]()


<img src="https://github.com/user-attachments/assets/f8ed6def-6690-48f4-a949-73bde453d7c0" width="500">

## I. Overview
SPECIALS (Spectral Extraction and Classification for ISM Analysis using Least Squares) is a GPU-compatible, Python-based image processing pipeline for data from spectral Confocal Spinning Disk Image Scanning Microscopy (CSD-ISM) systems. It implements the image reconstruction algorithms from "A Spectral Image Scanning Microscope for Multi-Color High-Resolution Imaging" by Bram L., et al. The pipeline decomposes spectral signatures into multi-color images while enhancing optical resolution.

**Problem Solved:** Addresses limitations of conventional multi-color fluorescence microscopy, such as time-intensive sequential acquisition and spatio-temporal chromatic artifacts. CoCos processes data from a system designed for concurrent high-resolution and simultaneous multi-color acquisition.

**Core Contribution:** The system combines a hardware modification (a custom linear Amici prism in the CSD-ISM setup) with the CoCos software. The software performs:
1.  **Spectral Decomposition:** Translates dispersed spectral signatures into distinct color channels.
2.  **ISM Pixel Reassignment:** Computationally enhances spatial resolution beyond the diffraction limit.

This approach enables multi-color, super-resolution imaging significantly faster (reportedly 3x) than standard CSD-ISM, with flexible color palette selection.

## II. System Requirements

| Category        | Recommendation                                         | Notes                                                        |
| :-------------- | :----------------------------------------------------- | :----------------------------------------------------------- |
| Operating System| Linux (e.g., Ubuntu 20.04+), Windows 10/11              | macOS compatibility was not tested vary.                                |
| Python Version  | 3.8 - 3.10                                             | Check `requirements.txt`.                                    |
| Key Libraries   | NumPy, SciPy, CuPy, scikit-image, Matplotlib, tifffile | See `requirements.txt` for full list.                        |
| GPU             | NVIDIA GPU with CUDA support (CUDA Toolkit 11.x+)      | Min. 6-8 GB VRAM (e.g., NVIDIA RTX 2070+). Essential for performance. |

## III. Installation Guide

It is strongly recommended to use a virtual environment.

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/ebensteinLab/SPECIALS.git
    cd SPECIALS
    ```

2.  **Set Up a Virtual Environment:**
    * E.g., Using `venv`:
        ```bash
        python -m venv specials-env
        # On Linux/macOS:
        source specials/bin/activate
        # On Windows:
        # cocos-env\Scripts\activate
        ```
        
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note on CuPy:* Ensure your NVIDIA drivers and CUDA Toolkit are correctly installed. You might need to install a specific CuPy version matching your CUDA Toolkit (e.g., `pip install cupy-cuda11x`). Refer to the [official CuPy installation guide](https://docs.cupy.dev/en/stable/install.html).


## IV. Citing CoCos

If you use SPECIALS in your work, please cite our paper:
    Bram L., Tal-Friedman O., Fibeesh N., Bar-Sinai Y., Flaxer E., Roichman Y., Ashery U., Ebenstein Y.*, Jeffet J.* (Year). A Spectral Image Scanning Microscope for Multi-Color High-Resolution Imaging. *Journal Name, Volume(Issue), Pages*.

## V. Contributing

Contributions are welcome! Please use GitHub Issues for bug reports and feature suggestions. Submit pull requests for code contributions.

## VI. License

SPECIALS is distributed under the terms of the **MIT License**.
See the `LICENSE` file in the repository for the full license text.
