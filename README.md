# 🩺 Multi-Disease Prediction System using Machine Learning
An AI-powered web application that predicts the risk of multiple diseases using trained machine learning models.

## 🚀 Project Overview

This project is a web-based healthcare prediction platform built using Machine Learning and Streamlit.

The system allows users to:
- Predict Diabetes Risk
- Predict Heart Disease
- Predict Liver Disease
- Predict Kidney Disease
- Predict Breast Cancer

Each prediction is powered by trained machine learning models with proper preprocessing and feature selection.


## 🧠 Machine Learning Models

| Disease        | Model Used             | Accuracy |
|---------------|-------------------------|----------|
| Diabetes      | Support Vector Machine  | ~77%     |
| Heart Disease | Random Forest           | ~85%     |
| Liver Disease | Gradient Boosting       | ~87%     |
| Kidney Disease| Random Forest           | 100%     |
| Breast Cancer | Random Forest           | ~95%     |


## 🏗 System Architecture

The application follows a modular machine learning pipeline architecture.

### Architecture Layers

1. **User Interface Layer**
   - Built using Streamlit
   - Collects user input
   - Displays prediction results

2. **Prediction Engine**
   - Input preprocessing
   - Feature alignment
   - Scaling using StandardScaler
   - Model prediction

3. **Machine Learning Models**
   - Disease-specific trained models
   - Saved using pickle (.pkl files)

4. **Model Storage Layer**
   - Model files
   - Scaler files
   - Feature column files

### System Flow Diagram

User
  │
  ▼
Streamlit Web App
  │
  ▼
Prediction Engine
  │
  ▼
Trained ML Models (.pkl)
  │
  ▼
Prediction Output

