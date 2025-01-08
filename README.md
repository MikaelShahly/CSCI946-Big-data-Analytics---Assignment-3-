# Evaluation of Generalisation Accuracy on Deep Features from ImageNet
This repository contains the implementation and results of our project for CSCI946 Big Data Analytics (University of Wollongong). The project investigates the generalisation performance of models trained on the ImageNet dataset across its original validation set (Test Set 1) and a new validation set (Test Set 2, ImageNetV2).

## Project Overview
Deep learning models often exhibit performance variability across datasets. This project focuses on:
- Transfer Learning: Utilised pre-trained models on ImageNet to extract deep features, significantly reducing computational complexity while retaining meaningful representations for downstream classification tasks.
- Examines the hypothesis that dataset complexity, rather than overfitting, drives performance drops between ImageNet and ImageNetV2.
- Evaluates the performance of two model types: CatBoost Classifier and Feed-Forward Neural Networks (FFNs).
- Provides recommendations to improve model generalisation.

## Key Features
- **Transfer Learning Approach**:
Deep features extracted from pre-trained ImageNet models were used as inputs, avoiding the need to train models from scratch.
This approach enabled faster experimentation and efficient utilisation of resources while leveraging the robustness of pre-trained representations.
Data Preparation: Dimensionality reduction (PCA) and stratified sampling for computational efficiency.
- **Models Used**:
CatBoost: Gradient boosting with optimised parameters.
FFNs: Multiple architectures tested with hyperparameter tuning using Optuna and Random Search.
- **Advanced Analysis**: Investigated performance gaps for specific labels and their causes using visual inspection and clustering.

## Tools and Libraries
Python: Primary programming language.
TensorFlow/Keras: FFN implementation.
CatBoost Library: For gradient boosting.
Optuna and Random Search: For hyperparameter optimisation.
Google Colab Pro: Computational environment.
Results

**Model Generalisation**:
Both models showed strong performance on Test Set 1 but struggled with Test Set 2.
FFN models achieved up to 92% accuracy on validation data, with custom architectures offering robust generalisation.
Findings:
Test Set 2 images are more complex and noisy, aligning with the hypothesis of dataset variability influencing performance.
Domain shift and dataset diversity significantly impact model accuracy.

**Recommendations**
Enhance training data diversity with more complex and varied images.
Incorporate data augmentation techniques to simulate real-world scenarios.
Experiment with deeper FFN architectures and domain adaptation techniques.

**Lessons Learned**
The importance of leveraging transfer learning for computational efficiency and model robustness.
The necessity of diverse datasets for better model generalisation.
Efficient resource management when handling large datasets.
The role of exploratory data analysis and clustering in understanding performance gaps.
