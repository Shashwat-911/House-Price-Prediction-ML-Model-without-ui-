# House-Price-Prediction-ML-Model-without-ui-

# House Price Prediction Model 🏠

A robust, end-to-end Machine Learning pipeline that predicts house prices based on structural and locational features. This project utilizes a `RandomForestRegressor` and features an interactive terminal-based UI for real-time predictions.

## Features
*   **Production-Ready Pipeline:** Utilizes scikit-learn's `Pipeline` and `ColumnTransformer` to prevent data leakage and ensure scalable preprocessing (One-Hot Encoding for categorical data, Standard Scaling for numerical data).
*   **High-Accuracy Model:** Powered by a Random Forest algorithm optimized for tabular data.
*   **Feature Importance Mapping:** Automatically extracts and displays the top features driving property prices (e.g., Location, Area).
*   **Interactive Terminal UI:** Includes a dynamic prediction loop that pulls valid categories and ranges directly from the training data, ensuring robust user input validation and preventing application crashes.

## Prerequisites
Ensure you have Python 3.8+ installed on your system. 

## Installation

1. Clone or download this repository to your local machine.
2. Navigate to the project directory in your terminal:
   ```bash
   cd path/to/your/project/folder
