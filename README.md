# High-Fidelity Automated Paleontology: Swin Transformer architecture via Formula-Driven Supervised Pre-training for Enhanced Microfossil Classification

---

## 🔬 Project Overview
This repository contains the code and pre-trained weights for classifying microfossils (radiolarians) using the Swin Transformer architecture. We investigate the performance enhancement achieved through Formula-Driven Supervised Learning (FDSL) pre-training compared to traditional ImageNet pre-training for geological classification tasks.

## 🚀 Getting Started

### Prerequisites
* Python 3.x
* Cuda 11.x (or newer)
* Access to the dataset (Download instructions/link will go here).

### Setup and Installation
1.  Clone the repository:
    ```bash
    git clone YOUR_GITHUB_URL
    # Ensure you have Git LFS installed to download the models
    git lfs pull
    ```
2.  Create and activate the environment:
    ```bash
    conda create -n microfossil-env python=3.9
    conda activate microfossil-env
    ```
3.  Install required libraries:
    ```bash
    pip install -r requirements.txt
    ```

## 📊 Results and Models
The best-performing model (e.g., the FDSL-trained Swin Transformer) is located in the `swin_final_results_advanced/Replicate_1/exfractal/best_model.pth` directory. Mean test accuracy achieved: [Insert Best Metric Here].

## 📚 References
* [we will Link to our paper]
* Swin Transformer: odels
* Formula-Driven Supervised Learning (FDSL): we will link the paper soon