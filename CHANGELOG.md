# Changelog

All notable changes to this project will be documented in this file.

## [1.0.0] - 2026-03-22

### Added
- **Adam Optimizer**: Switched to Adam (0.001 LR) for superior convergence at Batch 32.
- **WSL2 Compatibility**: Implemented parameter scaling (80k steps @ Batch 32) to support memory-constrained virtual environments.
- **Integrated Evaluation**: Main training loop now automatically generates CMC curves upon completion.

### Fixed
- **Architectural Fidelity**: Restored missing ReLU layers, recovering Rank-1 accuracy to **56.70%**.
- **Critical Memory Bug**: Fixed 6-channel pointer arithmetic overflow in the input layer.
- **Modern Toolchain**: Updated for **CUDA 12.6**, cuDNN 9, and dlib 20.0 compatibility.

### Performance
- **Rank-1 Accuracy**: 56.70% (CUHK03 Labeled, 50k steps).
- **Rank-5 Accuracy**: 86.31%
- **Rank-10 Accuracy**: 95.03%
