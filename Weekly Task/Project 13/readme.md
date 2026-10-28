# Project 13: CNN Architecture Comparison (Custom CNN vs. ResNet50)

## Overview
This project presents an empirical comparative evaluation between two Convolutional Neural Network architectures trained for multi-class image classification on the **CIFAR-10** benchmark dataset:
1. **Custom CNN**: A convolutional neural network built and trained from scratch.
2. **ResNet50 (Transfer Learning)**: A deep residual network pretrained on ImageNet, adapted with custom classification head layers.

## Dataset
- **Dataset**: CIFAR-10 (10 object categories: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- **Image Dimensions**: 32x32 RGB (upscaled to 96x96 for ResNet50 compatibility)
- **Split**: Train, validation, and test subsets

## Model Architectures & Parameter Comparison
| Metric | Custom CNN | ResNet50 (Transfer Learning) |
| :--- | :--- | :--- |
| **Trainable Parameters** | 356,810 | 263,562 |
| **Total Parameters** | 356,810 | 23,851,274 |
| **Training Duration** | ~2.74 minutes | ~29.74 minutes |
| **Test Accuracy** | **39.00%** | **47.60%** |

## Key Insights
- Pretrained weights from ImageNet enable ResNet50 to achieve higher classification accuracy (+8.60% absolute gain) despite training fewer parameters.
- Custom CNN trains significantly faster (~11x speedup) with low memory overhead, serving as a lightweight baseline.
- Performance evaluation includes confusion matrices, per-class precision/recall/F1-score classification reports, and visual inspection of predicted test samples.
