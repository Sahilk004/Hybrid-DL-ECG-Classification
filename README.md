# Hybrid Deep Learning Model for Scalogram-Based ECG Classification

This repository contains the implementation of a major project based on the paper "Hybrid Deep Learning Model for Scalogram-Based ECG Classification of Cardiovascular Diseases" by Kathayat and Renold (2025).

## Overview
The project processes 1D ECG signals from PhysioNet into 2D scalograms using Continuous Wavelet Transform (CWT). It utilizes a hybrid deep learning architecture to classify the signals into three categories:
* Arrhythmia (ARR)
* Congestive Heart Failure (CHF/CSR)
* Normal Sinus Rhythm (NSR)

## Architecture
1. **Spatial Feature Extraction:** ResNet-50 (pre-trained on ImageNet) enhanced with Squeeze-and-Excitation (SE) blocks for channel attention.
2. **Temporal Modeling:** A two-layer Long Short-Term Memory (LSTM) network to capture heartbeat sequential dependencies.

## Setup Instructions
1. Create a virtual environment and install dependencies: `pip install numpy scipy matplotlib wfdb pywt opencv-python tensorflow`
2. Download the MIT-BIH Arrhythmia, MIT-BIH NSR, and BIDMC CHF databases from PhysioNet.
3. Place the `.dat` and `.hea` files into `data/ARR`, `data/CSR`, and `data/NSR`.
4. Run `python preprocess_ecg.py` to denoise, segment, augment, and generate scalogram images.