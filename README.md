# Accident Detection and Alerting System

## Overview
The **Accident Detection and Alerting System** is a machine learning-based project designed to automatically detect road accidents from video frames and alert relevant authorities. By leveraging deep learning models and ensemble techniques, the system ensures high accuracy in classifying accident scenarios.

## Features
- Uses **EfficientNetB0** for feature extraction from images.
- Employs multiple machine learning classifiers (**Random Forest, Bagging, and Gradient Boosting**) for accurate accident detection.
- Implements **majority voting ensemble** to improve prediction accuracy.
- Provides **real-time visualization** of predictions for better interpretability.
- Optimized dataset handling using TensorFlow's `image_dataset_from_directory()` for efficient training and validation.

## Dataset
The dataset used for training and testing is available at the following link:  
[Download Dataset](https://drive.google.com/file/d/1fG8YNwqYJ3Ad2l4Mom0eiNf5ZkRoP5jG/view?usp=sharing)  

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/accident-detection-alert.git
   cd accident-detection-alert
   ```
2. Install the required dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Ensure TensorFlow and necessary machine learning libraries are installed.

## Usage
1. Place the dataset in the appropriate directory as specified in `ML_Project.py`.
2. Run the model training script:
   ```sh
   python ML_Project.py
   ```
3. The system will train models and provide evaluation metrics for accuracy, precision, recall, and F1-score.

## Model Architecture
- **Feature Extraction**: EfficientNetB0 extracts high-level features from video frames.
- **Classification Models**:
  - **Random Forest**
  - **Bagging Classifier**
  - **Gradient Boosting Classifier**
- **Ensemble Learning**: Uses majority voting to combine model predictions for better accuracy.

## Evaluation
The trained models are evaluated based on:
- **Validation Accuracy**
- **Test Accuracy**
- **F1 Score (Macro-Average)**
- **Ensemble Model Performance**

## Results
A comparative analysis of different models is presented through visualizations, including:
- Accuracy comparisons across classifiers
- F1-score comparisons
- Model performance graphs for validation vs. test accuracy

## Future Enhancements
- Implement real-time accident detection using a video feed.
- Integrate IoT-based alerting mechanisms.
- Deploy the model as a web service for live monitoring.

## License
This project is open-source and available under the **MIT License**.

Let me know if you need any modifications! 🚀
