# Real-time Depth Estimation GUI

A PyQt6-based application for real-time depth estimation using your webcam. Built with PyTorch and optimized for Apple Silicon (M1/M2) GPUs.

## Features

- Real-time depth map visualization
- GPU acceleration support for Apple Silicon
- Multiple colormap options
- Adjustable depth threshold
- Video recording capability
- FPS monitoring

## Installation

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Unix/macOS
# or
.venv\Scripts\activate  # Windows

# Install dependencies
pip install torch torchvision transformers opencv-python pillow PyQt6


