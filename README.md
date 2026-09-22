# 🍎 Hyperspectral Image Based Food Quality Inspection for Pesticide Detection in Apples

An AI-based food quality inspection system that uses **Hyperspectral Imaging (HSI)**, **Deep Learning**, and **Machine Learning** to classify apples according to their pesticide concentration.

The system classifies apples into three categories:

- 🟢 Fresh
- 🟡 Low Pesticide Concentration
- 🔴 High Pesticide Concentration

The project combines hyperspectral image preprocessing, data augmentation, pretrained deep learning feature extractors, and machine learning classifiers to build an automated and non-destructive pesticide detection pipeline.

---

## 👥 Project Team

### Md Dilnawaz Hussain
- Faculty No.: 22AIB470

### Hamza Ehsan
- Faculty No.: 23AIBEA627

**Supervisor:** Dr. Junaid Ali Reshi

**Project:** Minor Project - 2  
**Course Code:** AIP-3952

**Institution:**  
Zakir Husain College of Engineering & Technology  
Aligarh Muslim University, Aligarh, India

---

# 📌 Project Overview

Pesticide residues on agricultural products can create serious food safety concerns. Traditional pesticide detection techniques such as laboratory-based chemical analysis can be accurate but are generally time-consuming, costly, and destructive.

This project explores **Hyperspectral Imaging (HSI)** as a non-destructive alternative.

Hyperspectral imaging captures information across hundreds of spectral bands, providing both spatial and spectral information about the target object.

In this project, hyperspectral images of apples are processed using PCA and pretrained deep learning architectures. The extracted features are then classified using different machine learning models.

The complete pipeline is:

```text
Hyperspectral Image
        ↓
Data Augmentation
        ↓
Data Split
        ↓
PCA Dimensionality Reduction
        ↓
RGB Conversion
        ↓
Resize to 224 × 224
        ↓
ImageNet Normalization
        ↓
Deep Learning Feature Extraction
        ↓
VGG16 / ResNet50 / ViT-B/16
        ↓
Feature Vector
        ↓
Machine Learning Classifier
        ↓
Fresh / Low / High Pesticide
````

---

# 🎯 Objectives

The main objectives of this project are:

1. Develop an automated food quality inspection system for apples.
2. Detect pesticide concentration using hyperspectral images.
3. Classify apples into:

   * Fresh
   * Low pesticide concentration
   * High pesticide concentration
4. Apply data augmentation to increase the available training samples.
5. Reduce the dimensionality of hyperspectral data using PCA.
6. Extract features using pretrained:

   * VGG16
   * ResNet50
   * Vision Transformer (ViT-B/16)
7. Evaluate multiple machine learning classifiers.
8. Compare different feature extractor and classifier combinations.
9. Develop a prediction dashboard for hyperspectral apple quality inspection.

---

# 🧩 Problem Statement

The core problem is to accurately detect pesticide and fertilizer residues in apples using a **non-destructive hyperspectral imaging approach**.

Traditional chemical testing is slow, costly, and not suitable for real-time large-scale inspection.

The project therefore aims to develop an automated machine learning pipeline capable of distinguishing between:

```text
Fresh Apple
     ↓
Low Pesticide Concentration
     ↓
High Pesticide Concentration
```

The major technical challenges include:

* High-dimensional hyperspectral data
* 281 spectral channels
* Limited original dataset size
* Class imbalance
* Selection of suitable feature extraction architecture
* Selection of an effective machine learning classifier

---

# 📊 Dataset

The project uses the hyperspectral apple dataset introduced by **Roomi et al.**

The dataset was captured using a **Resonon Pika L hyperspectral camera**.

### Camera Specifications

```text
Spectral Range : 400–1000 nm
Spectral Bands : 281
Spatial Pixels : 900 pixels per line
```

The original dataset contains **617 hyperspectral images**.

The apples belong to three categories:

| Class              | Description                         |
| ------------------ | ----------------------------------- |
| Fresh              | No chemical treatment               |
| Low Concentration  | 1 ml/g chemical in 1 litre of water |
| High Concentration | 3 ml/g chemical in 1 litre of water |

After data augmentation:

```text
Total Images = 2,634
```

Class distribution:

| Class              | Samples | Approx. Percentage |
| ------------------ | ------: | -----------------: |
| Fresh              |     684 |                26% |
| High Concentration |     954 |                36% |
| Low Concentration  |     996 |                38% |

---

# 🔄 Data Augmentation

Because the original dataset contains only 617 images and has class imbalance, multiple augmentation techniques were applied.

The following transformations were used:

### 1. Horizontal Flip

Mirrors the image from left to right.

### 2. Vertical Flip

Flips the image from top to bottom.

### 3. Rotation 90°

Rotates the image by 90° counter-clockwise.

### 4. Rotation 180°

Rotates the image by 180°.

### 5. Rotation 270°

Rotates the image by 270° counter-clockwise.

After augmentation:

```text
617 Images
     ↓
Data Augmentation
     ↓
2,634 Images
```

---

# 🔬 Exploratory Data Analysis

The augmented dataset contains three classes:

```text
Fresh       → 684 samples
High        → 954 samples
Low         → 996 samples
```

The project performs EDA to examine the class distribution and identify class imbalance.

---

# ⚙️ Data Preprocessing

The original hyperspectral image contains 281 spectral channels.

The raw hyperspectral cube has approximately:

```text
300 × 900 × 281
```

Since pretrained RGB models such as VGG16, ResNet50 and ViT require three-channel input, PCA is used to reduce the hyperspectral representation.

### PCA Dimensionality Reduction

```text
Original:
300 × 900 × 281

        ↓ PCA

Reduced:
300 × 900 × 3
```

The three principal components are used as an RGB-like representation.

### Resizing

The image is resized to:

```text
224 × 224 × 3
```

### ImageNet Normalization

The following ImageNet statistics are used:

```text
Mean:
(0.485, 0.456, 0.406)

Standard Deviation:
(0.229, 0.224, 0.225)
```

---

# 🏗️ System Architecture

The complete system consists of six major layers.

```text
┌─────────────────────────────┐
│ 1. Data Ingestion Layer     │
│ Raw Hyperspectral Images    │
└──────────────┬──────────────┘
               ↓
┌─────────────────────────────┐
│ 2. Data Augmentation        │
│ Flip + Rotation             │
└──────────────┬──────────────┘
               ↓
┌─────────────────────────────┐
│ 3. Data Split &             │
│    Preprocessing             │
│ PCA → RGB → Resize → Norm.  │
└──────────────┬──────────────┘
               ↓
┌─────────────────────────────┐
│ 4. Feature Extraction       │
│ VGG16 / ResNet50 / ViT      │
└──────────────┬──────────────┘
               ↓
┌─────────────────────────────┐
│ 5. ML Classification        │
│ ANN / XGBoost / RF / NB     │
│ C4.5 / Ensemble             │
└──────────────┬──────────────┘
               ↓
┌─────────────────────────────┐
│ 6. Output & UI              │
│ Fresh / Low / High          │
└─────────────────────────────┘
```

---

# 🧠 Feature Extraction Models

Three pretrained deep learning architectures are used as feature extractors.

## 1. VGG16

VGG16 is used as a pretrained feature extraction network.

Input:

```text
224 × 224 × 3
```

Output feature vector:

```text
25,088 dimensions
```

The extracted features are passed to the machine learning classifiers.

---

## 2. ResNet50

ResNet50 is used to extract deep visual features using residual connections.

Input:

```text
224 × 224 × 3
```

Output feature vector:

```text
2,048 dimensions
```

---

## 3. Vision Transformer (ViT-B/16)

ViT-B/16 is used as a transformer-based feature extractor.

Input:

```text
224 × 224 × 3
```

Output feature vector:

```text
768 dimensions
```

The ViT model captures long-range relationships using self-attention.

---

# 🤖 Machine Learning Classifiers

The extracted feature vectors are passed to six different machine learning classifiers.

### Classifiers Used

1. Artificial Neural Network (ANN)
2. XGBoost
3. Random Forest
4. Naive Bayes
5. C4.5 Decision Tree
6. Ensemble

### Ensemble Model

The ensemble combines:

```text
Logistic Regression
        +
Naive Bayes
        +
C4.5
```

---

# 🔀 Dataset Splits

Two different train-validation-test configurations were used.

## 70:15:15 Split

```text
Training   : 1,842
Validation : 395
Testing    : 395
Total      : 2,634
```

## 80:10:10 Split

```text
Training   : 2,106
Validation : 263
Testing    : 263
Total      : 2,634
```

---

# 📏 Evaluation Metrics

The models are evaluated using:

* Accuracy
* Macro Precision
* Macro Recall
* Macro F1-Score

These metrics are used to compare different feature extractor and classifier combinations.

---

# 📈 Results

## VGG16 + Classifiers — 70:15:15

| Model         | Accuracy | Recall |    F1 | Precision |
| ------------- | -------: | -----: | ----: | --------: |
| ANN           |    0.810 |  0.816 | 0.819 |     0.822 |
| XGBoost       |    0.734 |  0.741 | 0.744 |     0.751 |
| Ensemble      |    0.714 |  0.754 | 0.752 |     0.750 |
| Naive Bayes   |    0.582 |  0.608 | 0.584 |     0.582 |
| Random Forest |    0.729 |  0.742 | 0.739 |     0.741 |
| C4.5          |    0.650 |  0.660 | 0.656 |     0.656 |

---

## ResNet50 + Classifiers — 70:15:15

| Model         | Accuracy | Recall |    F1 | Precision |
| ------------- | -------: | -----: | ----: | --------: |
| ANN           |    0.853 |  0.860 | 0.857 |     0.863 |
| XGBoost       |    0.787 |  0.788 | 0.789 |     0.792 |
| Ensemble      |    0.713 |  0.733 | 0.713 |     0.722 |
| Naive Bayes   |    0.453 |  0.489 | 0.425 |     0.488 |
| Random Forest |    0.812 |  0.819 | 0.815 |     0.814 |
| C4.5          |    0.617 |  0.633 | 0.627 |     0.623 |

---

# 🏆 Best Performance

The best-performing configuration reported in the project is:

```text
Feature Extractor : ViT-B/16
Classifier         : ANN
Data Split         : 70:15:15
```

### Performance

```text
Accuracy  : 0.951
Recall    : 0.950
F1 Score  : 0.953
Precision : 0.957
```

The project identifies **ViT + ANN with the 70:15:15 split** as the best-performing configuration.

The ANN uses two hidden layers, with Log Loss as the loss function and Adam as the optimizer.

---

# 📊 ViT + ANN Results

The project also includes:

* Training loss curve
* Validation loss curve
* Test-set confusion matrix

The confusion matrix provides class-wise prediction information for:

```text
Fresh
High Concentration
Low Concentration
```

---

# 🖥️ Inference Dashboard

A user-facing inference dashboard is included in the project.

The dashboard allows users to:

1. Upload a hyperspectral image.
2. Select the feature extraction/model configuration.
3. Run inference.
4. View the predicted apple quality class.
5. View prediction confidence/results.

Output classes:

```text
Fresh
High Concentration
Low Concentration
```

The project presentation includes screenshots of the inference dashboard and model-by-model prediction results.

---

# 🔬 Methodology

The complete methodology can be summarized as:

```text
Step 1
Collect Hyperspectral Apple Images
        ↓
Step 2
Apply Data Augmentation
        ↓
Step 3
Split Dataset
70:15:15 / 80:10:10
        ↓
Step 4
Apply PCA
281 Spectral Bands → 3 Components
        ↓
Step 5
Resize
224 × 224
        ↓
Step 6
ImageNet Normalization
        ↓
Step 7
Feature Extraction
VGG16 / ResNet50 / ViT
        ↓
Step 8
Extract Feature Vectors
        ↓
Step 9
Machine Learning Classification
ANN / XGBoost / RF / NB / C4.5 / Ensemble
        ↓
Step 10
Evaluate Performance
        ↓
Step 11
Predict Apple Quality
Fresh / Low / High
```

---

# 🛠️ Technologies Used

### Programming

* Python

### Deep Learning

* VGG16
* ResNet50
* Vision Transformer (ViT-B/16)

### Machine Learning

* Artificial Neural Network
* XGBoost
* Random Forest
* Naive Bayes
* C4.5 Decision Tree
* Ensemble Learning

### Image Processing

* Hyperspectral Imaging
* PCA
* Image Resizing
* ImageNet Normalization
* Data Augmentation

### Evaluation

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix
* Training/Validation Loss

---

# 📂 Suggested Repository Structure

```text
Hyperspectral-Apple-Pesticide-Detection/
│
├── README.md
│
├── data/
│   └── README.md
│
├── preprocessing/
│   ├── pca.py
│   ├── augmentation.py
│   └── preprocessing.py
│
├── feature_extraction/
│   ├── vgg16.py
│   ├── resnet50.py
│   └── vit.py
│
├── models/
│   ├── ann.py
│   ├── xgboost_model.py
│   ├── random_forest.py
│   ├── naive_bayes.py
│   ├── c45.py
│   └── ensemble.py
│
├── evaluation/
│   ├── metrics.py
│   ├── confusion_matrix.py
│   └── results.py
│
├── dashboard/
│   └── app.py
│
├── results/
│   ├── confusion_matrix.png
│   ├── loss_curve.png
│   └── model_results.csv
│
├── notebooks/
│   └── experiments.ipynb
│
└── requirements.txt
```

> The above structure is a suggested organization. Rename files/folders according to the actual files in your repository.

---

# 🚀 Applications

This project can be used as a foundation for:

* 🍎 Apple quality inspection
* 🧪 Pesticide residue screening
* 🌾 Agricultural quality assessment
* 🏭 Automated food inspection
* 🔬 Non-destructive food analysis
* 📊 Hyperspectral image classification
* 🤖 AI-based agricultural monitoring

---

# 🔍 Key Features

* Non-destructive hyperspectral inspection
* Three-class apple quality classification
* PCA-based hyperspectral dimensionality reduction
* Data augmentation
* Pretrained deep learning feature extraction
* Multiple ML classifier comparison
* ViT-based feature extraction
* ANN-based classification
* Confusion matrix analysis
* Inference dashboard
* Model performance comparison

---

# 📌 Key Results

The project demonstrates that combining hyperspectral preprocessing with pretrained deep learning feature extraction and traditional machine learning classification can be used for automated apple quality classification.

The reported best configuration is:

```text
ViT-B/16 + ANN
70:15:15 Split
```

with:

```text
Accuracy  = 95.1%
Recall    = 95.0%
F1 Score  = 95.3%
Precision = 95.7%
```

---

# ⚠️ Limitations

The project is based on a specific hyperspectral apple dataset and three pesticide concentration categories.

The system's performance may depend on:

* Dataset characteristics
* Image acquisition conditions
* Hyperspectral camera properties
* Chemical treatment conditions
* Data distribution
* Feature extractor
* Classification model

Further validation on larger and more diverse real-world datasets would be required before practical deployment.

---

# 🔮 Future Scope

Possible future improvements include:

* Testing on larger hyperspectral datasets.
* Using additional hyperspectral feature extraction techniques.
* Exploring end-to-end deep learning architectures.
* Evaluating additional transformer architectures.
* Improving class imbalance handling.
* Expanding the system to additional fruits and agricultural products.
* Integrating real-time hyperspectral cameras.
* Improving the inference dashboard.
* Deploying the system for large-scale food quality inspection.
* Exploring more advanced spectral-spatial learning techniques.

---

# 📚 References

1. S. Md. Mansoor Roomi, B. Sathya Bama, V. Puvi Lakshmi, and M. Vaishnavi,
   **"Hyperspectral dataset of pure and pesticide-coated apples for measuring the level of fertilizers used,"** Data in Brief, Vol. 49, Article 109321, 2023.

2. A. Shafique, M. Siraj, B. Cheng, S. A. Alsaif, and T. Sadad,
   **"Hyperspectral imaging and advanced vision transformers for identifying pure and pesticide-coated apples,"** IEEE Access, Vol. 13, pp. 66405–66419, 2025.

---

# 👨‍💻 Authors

### Md Dilnawaz Hussain

Faculty No.: 22AIB470

### Hamza Ehsan

Faculty No.: 23AIBEA627

### Supervisor

**Dr. Junaid Ali Reshi**

Interdisciplinary Centre for Artificial Intelligence
Zakir Husain College of Engineering & Technology
Aligarh Muslim University, Aligarh

---

# ⭐ Project Summary

**Hyperspectral Image Based Food Quality Inspection for Pesticide Detection in Apples** combines hyperspectral imaging, PCA-based preprocessing, pretrained deep learning models, and machine learning classifiers to build a non-destructive apple quality inspection system.

The pipeline processes hyperspectral images, reduces their spectral dimensionality, extracts deep visual features using VGG16, ResNet50, or ViT-B/16, and performs three-class classification using multiple machine learning algorithms.

The reported best configuration, **ViT-B/16 + ANN with a 70:15:15 data split**, achieved an accuracy of **95.1%** on the evaluated dataset.

Aur final **ViT + ANN (70:15:15)** result `0.951 accuracy` report/PPT mein explicitly diya gaya hai. :contentReference[oaicite:3]{index=3}
```
