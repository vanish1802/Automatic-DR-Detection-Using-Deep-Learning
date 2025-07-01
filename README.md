# Diabetic Retinopathy Detection (Binary Classification) using DenseNet169

This repository contains the implementation of a binary classification model for early-stage detection of **Diabetic Retinopathy (DR)** using fundus images. The model is based on **DenseNet169** and has been trained and evaluated on a preprocessed subset of the **APTOS 2019 Blindness Detection** dataset.

## Project Objective

The primary goal is to **classify fundus images as either showing signs of DR or not**, simplifying the original 5-class DR grading task into a binary format. This decision follows discussions with Dr. Himanshu and Prof. Surya Prakash, where we agreed that binary classification is a more practical initial step due to severe class imbalance in multi-class settings.

## Dataset

- **Source**: [APTOS 2019 Blindness Detection](https://www.kaggle.com/competitions/aptos2019-blindness-detection/)
- **Preprocessing**:
  - Applied CLAHE (Contrast Limited Adaptive Histogram Equalization)
  - Gaussian blur for denoising
  - Custom data generator for efficient loading and on-the-fly augmentation

## Model Architecture

- **Base Model**: DenseNet169 (ImageNet weights, without top layers)
- **Custom Head**:
  - Global Average Pooling
  - Dense layer(s) + Dropout
  - Final output layer with sigmoid activation

## Training Details

- **Epochs**: 10 (initial) + 5 (fine-tuning)
- **Loss**: Binary Crossentropy
- **Optimizer**: Adam
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1 Score, Confusion Matrix

## Results

**Evaluation Metrics (Test Set):**
- Accuracy: **97.16%**
- Precision: **97.47%**
- Recall (Sensitivity): **96.25%**
- Specificity: **97.92%**
- F1 Score: **96.86%**
- AUC: **99.27%**

**Confusion Matrix:**
- True Positives (TP): 154  
- True Negatives (TN): 188  
- False Positives (FP): 4  
- False Negatives (FN): 6

The model achieved high accuracy and robustness in detecting DR vs. No DR cases, showing significant improvement over previous multi-class attempts.


## Future Work

- Explore more balanced training using GAN-based data augmentation
- Extend model to multi-class DR classification once a stable binary model is achieved
- Test ensemble methods (e.g., EfficientNetB0, InceptionV3) with meta-classifiers

## Author

**Vanish Jain**  
MS (Research), Computer Science  
Specialization: Deep Learning, Medical Image Classification  
IIT Indore
