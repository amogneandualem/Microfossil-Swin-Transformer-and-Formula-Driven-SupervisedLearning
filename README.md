### High-Fidelity Automated Paleontology: Swin Transformer architecture via Formula-Driven Supervised Pre-training for Enhanced Microfossil Classification


## Project Overview
This repository contains the code and pre-trained weights for classifying microfossils (radiolarians) using the Swin Transformer architecture. We investigate the performance enhancement achieved through Formula-Driven Supervised Learning (FDSL) pre-training compared to traditional ImageNet pre-training for geological classification tasks. 
This repository presents a comprehensive comparison of three pre-training strategies for Swin Transformer models on a 32-class image classification task. The study evaluates the performance of ImageNet, ExFractal, and RCDB pre-training methods across three independent replicates to ensure statistical reliability.

## Getting Started experiment

### Prerequisites
* Python 3.x
* CUDA 11.x and above for GPU version
* Access to the dataset  will be linked 

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
#### Architecture & Setup
 ## Model Configuration
Model: swin_base_patch4_window7_224
- Parameters: 87,079,364
Input Size: 224×224 pixels
Batch Size: 16
Epochs per Experiment: 50
Replicates: 3 (for statistical significance)
#### Dataset
Total Images: 7,287
Classes: 32
## Split:
Training: 5,829 images
Validation: 728 images
Testing: 730 images
##  Results Summary
Average Test Accuracy (3 Replicates)
Pre-training Method	Mean Test Accuracy	Standard Deviation	Best Replicate
ExFractal	90.41%	±0.89%	91.64%
ImageNet	90.27%	±0.30%	90.55%
RCDB	85.71%	±1.12%	87.26%
Validation Performance
ExFractal: 91.85% mean validation accuracy

ImageNet: 90.89% mean validation accuracy

RCDB: 86.49% mean validation accuracy

Swin Transformer Pre-training Comparison Study. 
Experimental results comparing ImageNet, ExFractal, and RCDB pre-training.
Available: [GitHub Repository URL]
## 📊 Results and Models
The best-performing model (e.g., the FDSL-trained Swin Transformer) is located in the `swin_final_results_advanced/Replicate_1/exfractal/best_model.pth` directory. Mean test accuracy achieved: [Insert Best Metric Here].

## 📚 References
* [we will Link to our paper]
* Swin Transformer: odels
* Formula-Driven Supervised Learning (FDSL): we will link the paper soon
