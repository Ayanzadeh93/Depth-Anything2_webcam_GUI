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





Usage
bashCopypython depth_estimation.py
Requirements

Python 3.9+
PyTorch 2.0+
Apple Silicon Mac (for GPU support) or any computer with a webcam
Required Python packages:

PyQt6
torch
torchvision
transformers
opencv-python
pillow



Controls

Start/Stop: Begin/end depth estimation
Record: Save video output
Depth Threshold: Adjust depth sensitivity
Colormap: Change visualization style

License
MIT License
Acknowledgments
Uses the Depth-Anything model for depth estimation.
