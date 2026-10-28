# Project 9: Neural Network Optimizer Comparison (SGD vs. Adam)

## Overview
An experimental investigation into optimizer convergence behaviors by training a multi-class feedforward neural network on the Iris dataset under identical train-test splits and layer architectures using:
1. **Stochastic Gradient Descent (SGD)**: Static learning rate $\eta = 0.01$.
2. **Adaptive Moment Estimation (Adam)**: Adaptive learning rate $\eta = 0.001$ with exponential moving averages of gradients.

## Dataset & Training Setup
- **Dataset**: Fisher's Iris Dataset (150 samples, 4 features, 3 target classes: *Setosa*, *Versicolor*, *Virginica*)
- **Split**: 120 training samples (80%), 30 test samples (20%)
- **Epochs**: 50 epochs with validation tracking

## Recorded Experimental Results
| Optimizer | Learning Rate | Test Loss | Test Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SGD** | 0.010 | 0.4278 | **80.00%** | 0.88 | 0.80 | 0.78 |
| **Adam** | 0.001 | 0.3270 | **86.67%** | 0.88 | 0.87 | 0.87 |

## Key Insights
- Adam achieved lower test loss (0.3270 vs. 0.4278) and higher classification accuracy (86.67% vs. 80.00%) due to its adaptive learning rate per parameter.
- SGD struggled on *Versicolor* class recall (0.40), whereas Adam achieved 0.70 recall, indicating superior escape from local plateaus.

## Artifacts
- Jupyter Notebook: `dl_assgn_09.ipynb`
