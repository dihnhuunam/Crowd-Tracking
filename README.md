
# Crowd Tracking using MCNN and CRSNet

## Description
This repository provides code for the paper "Crowd Tracking using MCNN and CRSNet." It builds on the approaches in "Crowd Tracking using Multi-Column Convolutional Neural Networks" and "Crowd Tracking via Correlation Structure of Regions," both by Y. Li et al.

## Table of Contents
- [Introduction](#introduction)
- [Requirements](#requirements)
- [Usage](#usage)
- [Results](#results)
- [Citation](#citation)

## Introduction
This project leverages Multi-Column Convolutional Neural Networks (MCNN) and CRSNet to tackle the problem of crowd tracking. By combining these architectures, the model aims to capture crowd density accurately and track movement across frames, making it suitable for use in surveillance, public event monitoring, and safety assessment systems.

## Requirements
- Python 3.9
- Conda (for environment setup)

### Dependencies
Run the `setup.sh` script to install all required packages and dependencies.

## Usage
1. **Clone the repository**:
   ```bash
   git clone https://github.com/dihnhuunam/Crowd-Tracking.git
   cd Crowd-Tracking
   ```

2. **Download the dataset**:
   - Download the ShanghaiTech dataset from [Kaggle](https://www.kaggle.com/datasets/tthien/shanghaitech).
   - Place the dataset as the `input` folder in the repository directory. Make sure it follows the required structure as specified in the code or preprocessing script.

3. **Install dependencies**:
   - Run `setup.sh`:
     ```bash
     ./setup.sh
     ```

4. **Run the training script**:
   - Start training with the following command:
     ```bash
     python train.py
     ```

5. **Run the testing script**:
   - Start testing with the following command:
     ```bash
     python test.py
     ```


## Results
After training, the model should provide metrics like MAE and MSE, and visualizations of crowd density estimates can be generated. Examples will be available in the `results` folder.

## Citation
If you find this work helpful, please consider citing the following papers:
- Y. Li et al., "Crowd Tracking using Multi-Column Convolutional Neural Networks."
- Y. Li et al., "Crowd Tracking via Correlation Structure of Regions."
