Improved Deep Learning Architecture for Person Re-Identification
================================================================

This repo attempts to recreate the person re-identification architecture described in [An Improved Deep Learning Architecture for Person Re-Identification by Ahmed et al.](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Ahmed_An_Improved_Deep_2015_CVPR_paper.pdf). The main deep learning library used to do this is the [*dlib* machine learning library](http://dlib.net/).

Requirements
------------

### dlib

- **Minimum Required Version:** 19.0
- **Dependencies**
  - `C++11`-compatible compiler
  - `CUDA 7.5` or greater
  - `cuDNN v5` or greater

### CMake

- **Minimum Required Version:** 2.6

### HDF5

Used for loading the `CUHK03` dataset from a MATLAB `mat` file.
