# Kolmogorov flow prediction
### This repository contains the implementation and comparation of U-Net and Fourier Neural Operators for predicting the dynamics of Kolmogorov flows.

## 1. Instalation and dependencies

* 1. Clone the repository:
```bash
git clone https://github.com/JulenMR/kolmogorov_flow_JMR.git
cd kolmogorov_flow_JMR
``` 
* 2. Create virtual environment

    python -m venv venv

  source venv/bin/activate

* 3. Install dependencies:
```bash
  pip install numpy matplotlib torch wandb json pandas h5py huggingface_hub
```

## 2. Downloading the dataset

  The dataset is not included in this repository due to its size. To download Kolmogorov flow dataset, run the preprocessing script:
```bash
python scr/preprocessing.py
```

## 3. Reproduce results

 Once the dataset is downloaded, you can execute the `reproduce_results.ipynb` notebook. This notebook performs a comprehensive evaluation through the following stages:

*   Model Selection: Compares the performance across all trained architectures to identify the optimal hyperparameter configurations for both U-Net and FNO.
*   Training Dynamics: Visualizes and benchmarks the learning curves and validation NRMSE metrics.
*   One-Step Error Analysis: Evaluates local predictive accuracy by calculating the error between the ground truth and model predictions for a single future timestep.
*   Autoregressive Rollout: Simulates and visualizes the accumulated error over time, demonstrating the stability and physical consistency of the models during long-term temporal predictions.
