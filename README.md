# 3D Gaussian Splatting Portfolio

Interactive 3D Gaussian Splatting reconstruction and browser-based WebGPU viewer.

## Live Demo

**Interactive Portfolio:**
https://atal-neupane.github.io/3dgs-portfolio/

**GitHub Repository:**
https://github.com/atal-neupane/3dgs-portfolio

## Overview

This project demonstrates an end-to-end 3D Gaussian Splatting pipeline, from multi-view images and camera reconstruction to GPU-based Gaussian optimization and interactive browser rendering.

The reconstruction was trained using the GraphDeco implementation of 3D Gaussian Splatting on an NVIDIA RTX 3050 laptop GPU with 6 GB of VRAM.

The trained model contains **238,078 Gaussian primitives** reconstructed from **84 input images**, with **79 registered images** after COLMAP processing.

## Interactive Viewer

The reconstructed scene can be explored directly in the browser using WebGPU.

**[Open the Interactive 3D Viewer](https://atal-neupane.github.io/3dgs-portfolio/viewer.html)**

The viewer supports:

- Orbit
- Pan
- Zoom
- Interactive 3D scene inspection
- Browser-based WebGPU rendering

## Pipeline

```
Multi-view Images
        ↓
      COLMAP
        ↓
Camera & Sparse Reconstruction
        ↓
GraphDeco 3D Gaussian Splatting
        ↓
238K Gaussian Model
        ↓
SOG Web Representation
        ↓
WebGPU / SuperSplat Viewer
```

### Current Reconstruction

| Property | Value |
| --- | --- |
| Input images | 84 |
| Registered images | 79 |
| Trained Gaussians | 238,078 |
| Training iterations | 3,000 |
| GPU | NVIDIA RTX 3050 |
| GPU memory | 6 GB VRAM |
| Browser rendering | WebGPU |
| Viewer | SuperSplat |

## Reconstruction

COLMAP was used to estimate camera parameters and poses from the multi-view image sequence and generate the sparse reconstruction used to initialize the Gaussian Splatting pipeline.

The trained model was then optimized using the GraphDeco implementation of 3D Gaussian Splatting.

The resulting reconstruction contains 238,078 trained Gaussian primitives.

## Web Deployment

The original trained GraphDeco PLY is preserved separately as the reconstruction artifact.

For browser deployment, a separate SOG representation is generated from the trained model and hosted remotely. The SOG representation is used only for web delivery and does not replace the original training artifact.

The public viewer therefore follows this architecture:

```
Browser
   ↓
GitHub Pages
   ↓
SuperSplat Viewer
   ↓
SOG Model
   ↓
Cloudflare R2
   ↓
WebGPU
```

## Technologies

- Python
- PyTorch
- CUDA
- COLMAP
- 3D Gaussian Splatting
- WebGPU
- SuperSplat
- FastAPI
- Docker
- GitHub Pages
- Cloudflare R2

## Repository Structure

```
3dgs-portfolio/
│
├── docs/
│   ├── index.html        # Portfolio landing page
│   ├── viewer.html       # WebGPU Gaussian Splatting viewer
│   └── settings.json     # Viewer configuration
│
├── app.py                # FastAPI rendering backend
├── Dockerfile            # GPU-enabled backend container
├── gaussian_renderer/    # Gaussian rendering implementation
├── scene/                # Scene and camera handling
├── utils/                # Supporting utilities
│
├── .gitignore
└── README.md
```

## Backend

The repository also contains a Dockerized FastAPI backend based on the GraphDeco rendering pipeline.

The backend loads the trained Gaussian model using PyTorch and CUDA and provides endpoints for health checking and server-side rendering.

Example architecture:

```
Client
  ↓
FastAPI
  ↓
PyTorch
  ↓
GraphDeco 3DGS
  ↓
CUDA
  ↓
GPU Rendering
```

The backend and browser viewer are separate components. The public interactive portfolio currently uses the WebGPU viewer for real-time browser visualization.

## Research Direction

This implementation serves as a baseline for ongoing MSc research into memory-efficient 3D Gaussian Splatting for memory-constrained devices.

The research investigates methods for reducing representation size and memory requirements while studying the trade-offs between:

- Gaussian count
- Model size
- GPU memory usage
- Rendering quality
- Rendering speed
- Computational cost

The current public implementation represents the baseline reconstruction and deployment pipeline. Proposed memory-optimization methods and experimental results are part of ongoing research and are not presented here as completed results.

## Current Status

The project currently provides:

- [x] Multi-view image reconstruction with COLMAP
- [x] Camera pose estimation
- [x] GraphDeco 3D Gaussian Splatting training
- [x] GPU-based rendering with CUDA
- [x] 238K-Gaussian trained reconstruction
- [x] Interactive WebGPU visualization
- [x] SuperSplat browser viewer
- [x] SOG web-delivery representation
- [x] Public GitHub Pages deployment
- [x] Dockerized FastAPI rendering backend
- [ ] Memory-efficient Gaussian compression and pruning research
- [ ] Quantitative evaluation of memory/quality trade-offs

## Author

**Atal Neupane**

MSc in Informatics and Intelligent Systems Engineering
Computer Vision · Artificial Intelligence · 3D Reconstruction

Portfolio: https://atal-neupane.github.io/3dgs-portfolio/
GitHub: https://github.com/atal-neupane
