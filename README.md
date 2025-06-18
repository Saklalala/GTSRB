# GTSRB - German Traffic Sign Classification

A machine learning project implementing classification techniques for German traffic sign recognition using the GTSRB dataset, achieving 91.99% test accuracy.

## Overview

This project addresses traffic sign classification by comparing traditional machine learning approaches with modern transfer learning methods. The work demonstrates the superiority of learned representations over hand-crafted feature engineering.

**Key Results:**
- 91.99% test accuracy in Kaggle competition
- 35% performance improvement using deep learning over traditional feature engineering
- Comprehensive analysis with 5-fold cross-validation

## Technical Approach

### Feature Engineering
- Domain-specific features: HOG on digit ROI, projection profiles, Local Binary Patterns
- Transfer learning: ResNet50 pre-trained on ImageNet
- Comparison across 4 distinct feature sets

### Machine Learning Models
- Support Vector Machine (SVM)
- Random Forest 
- CNN Transfer Learning
- Ensemble methods with hard voting

### Validation
- Stratified 5-fold cross-validation
- GridSearchCV hyperparameter optimization
- Statistical significance testing

## Results

| Model | Feature Set | CV Accuracy | Test Accuracy |
|-------|-------------|-------------|---------------|
| SVM | Deep Features | 91.38 ± 0.51% | 91.99% |
| RF | Deep Features | 81.23 ± 0.58% | 80.68% |
| CNN | Transfer Learning | 80.24% | 79.88% |
| SVM | Provided Features | 85.29 ± 0.89% | 46.35% |
| Ensemble | Hard Voting | - | 90.39% |


## Key Findings

- Transfer learning with deep features significantly outperformed traditional feature engineering
- SVM with ResNet50 features achieved the best performance
- Speed limit classification remains challenging due to fine-grained digit recognition
- Ensemble methods showed limited improvement when base learners used similar high-quality features

## Dataset

The project uses a subset of the German Traffic Sign Recognition Benchmark (GTSRB):
- 5,488 training images
- 2,353 test images  
- 43 traffic sign classes
- Real-world conditions with varying lighting, angles, and occlusions


This project is licensed under the MIT License

## Acknowledgments:
University of Melbourne - COMP30027 Machine Learning Course
GTSRB Dataset - German Traffic Sign Recognition Benchmark
Kaggle - Competition platform and evaluation framework
Open Source Community - scikit-learn, TensorFlow, and related libraries
