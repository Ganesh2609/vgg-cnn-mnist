# VGG-CNN Feature Extraction with Custom SVM Implementation

A comprehensive machine learning project that combines VGG16 convolutional neural network feature extraction with custom Support Vector Machine implementation for multi-class classification of handwritten digits and signal data.

## Overview

This project implements a hybrid approach to classification by leveraging the feature extraction capabilities of pre-trained VGG16 CNN and combining it with custom-built SVM classifiers. The system supports both image classification (handwritten digits) and signal classification tasks using two different SVM strategies: One-vs-One and One-vs-Rest.

## Theoretical Background

### VGG16 Feature Extraction
The project utilizes VGG16, a deep convolutional neural network pre-trained on ImageNet, to extract high-level features from input images. By removing the final classification layer and using the penultimate layer's output, we obtain 4096-dimensional feature vectors that capture rich semantic information about the input images.

### Custom SVM Implementation
The SVM implementation uses convex optimization (CVXPY) to solve the quadratic programming problem inherent in SVM training. Two multi-class strategies are implemented:

- **One-vs-Rest (OvR)**: Trains one binary classifier per class against all other classes
- **One-vs-One (OvO)**: Trains binary classifiers for each pair of classes, then uses voting for final prediction

The SVM formulation includes:
- Soft margin with regularization parameter C
- Slack variables for handling non-linearly separable data
- Distance-based decision making for classification

## Project Structure

```
├── Image Application/
│   ├── Extracting_Features.py      # VGG16 feature extraction for images
│   ├── Trainer_One_Digit.py        # Train One-vs-One SVM for digits
│   ├── Trainer_Rest_Digit.py       # Train One-vs-Rest SVM for digits
│   ├── Tester_One_Digit.py         # Test One-vs-One classifier
│   ├── Tester_Rest_Digit.py        # Test One-vs-Rest classifier
│   └── import_external.py          # Path configuration
├── Signal Application/
│   ├── Extract_signal_features.py  # Feature extraction for signal data
│   ├── Train_one_one.py            # Train One-vs-One SVM for signals
│   ├── Train_one_rest.py           # Train One-vs-Rest SVM for signals
│   ├── test_one_one.py             # Test One-vs-One signal classifier
│   ├── test_one_rest.py            # Test One-vs-Rest signal classifier
│   └── import_external.py          # Path configuration
└── Libraries/
    ├── Feature.py                   # Feature extraction utilities
    ├── SVM.py                       # Custom SVM implementation
    └── RKS.py                       # Random Kitchen Sinks (experimental)
```

## Key Features

### Dual Application Domains
- **Image Classification**: Handwritten digit recognition using VGG16 features
- **Signal Classification**: Time-series signal classification with statistical features

### Custom SVM Implementation
- Built from scratch using CVXPY for convex optimization
- Implements both One-vs-One and One-vs-Rest strategies
- Includes soft margin formulation with error handling
- Distance-based classification with hyperplane separation

### Feature Engineering
- VGG16-based deep feature extraction for images (4096-dimensional vectors)
- Statistical feature extraction for signal data (32-dimensional vectors)
- Automated data preprocessing and normalization

### Modular Architecture
- Separate training and testing pipelines
- JSON-based model persistence
- Configurable hyperparameters (C parameter for SVM)
- Extensible library structure

## Technical Implementation

### Image Processing Pipeline
1. Load and resize images to 224x224 pixels
2. Apply VGG16 preprocessing (normalization, mean subtraction)
3. Extract features from the second-to-last layer (4096 dimensions)
4. Group features by class labels
5. Train SVM classifiers using extracted features

### Signal Processing Pipeline
1. Load CSV signal data with multiple features
2. Clean data and handle missing values
3. Group signals by type (excluding 'Q' type)
4. Extract 32-dimensional feature vectors
5. Apply SVM classification with optimized parameters

### SVM Optimization
The custom SVM solver formulates the problem as:
- Minimize: 0.5 * ||w||² + C * Σ(error_i)
- Subject to: margin constraints and non-negativity of slack variables
- Uses CVXPY's efficient quadratic programming solver

## Usage

### Image Classification
```bash
# Extract features from training and test datasets
python Image Application/Extracting_Features.py

# Train classifiers
python Image Application/Trainer_One_Digit.py
python Image Application/Trainer_Rest_Digit.py

# Test performance
python Image Application/Tester_One_Digit.py
python Image Application/Tester_Rest_Digit.py
```

### Signal Classification
```bash
# Extract signal features
python Signal Application/Extract_signal_features.py

# Train classifiers
python Signal Application/Train_one_one.py
python Signal Application/Train_one_rest.py

# Evaluate models
python Signal Application/test_one_one.py
python Signal Application/test_one_rest.py
```

## Dependencies

- TensorFlow/Keras (VGG16 model)
- OpenCV (image processing)
- CVXPY (convex optimization)
- NumPy (numerical computations)
- Pandas (data manipulation)
- JSON (model serialization)

## Performance Characteristics

The system provides accuracy metrics for each class individually and overall average accuracy. The One-vs-One approach typically shows better performance on multi-class problems due to its localized decision boundaries, while One-vs-Rest is computationally more efficient for large numbers of classes.

## Applications

This hybrid approach is particularly suitable for:
- Handwritten digit recognition systems
- Signal classification in telecommunications
- Pattern recognition in time-series data
- Educational demonstrations of ML concepts
- Research in combining deep learning with traditional ML methods