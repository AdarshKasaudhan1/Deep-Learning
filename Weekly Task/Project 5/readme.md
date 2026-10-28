# Project 5: Activation Functions & Multi-Layer Neural Networks

## Overview
An empirical study implementing standard non-linear activation functions in Python and training a multi-layer neural network on the Pima Indians Diabetes Database.

## Activation Functions Analyzed
- **Sigmoid**: $\sigma(z) = \frac{1}{1 + e^{-z}}$ (mapping outputs to $(0, 1)$, ideal for binary classification output layers)
- **Tanh**: $\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$ (zero-centered mapping to $(-1, 1)$)
- **ReLU (Rectified Linear Unit)**: $f(z) = \max(0, z)$ (mitigating vanishing gradient issues in deep networks)
- **Leaky ReLU**: $f(z) = \max(\alpha z, z)$ (addressing the dying ReLU problem)

## Dataset & Experiments
- **Dataset**: `diabetes.csv` (768 patient records, 8 clinical features including Glucose, Blood Pressure, Insulin, BMI, Age)
- **Target**: Outcome (0: Non-diabetic, 1: Diabetic)
- **Evaluation**: Convergence behavior, loss reduction, and classification accuracy across different activation function combinations.

## Artifacts
- Jupyter Notebook: `Dl_assignment_5.ipynb`
- Dataset File: `diabetes.csv`
