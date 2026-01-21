# Heart Disease Prediction
## 📌 Overview
This project is a comprehensive Machine Learning pipeline designed to predict the likelihood of heart disease in patients based on clinical parameters. It includes a complete end-to-end workflow: data preprocessing, feature engineering, model selection, and a web-based interface for real-time predictions.

## 🚀 Features
- Automated Pipeline: Handles missing values, outlier detection, and data balancing (SMOTE).
- Feature Selection: Uses Variance Thresholding and Pearson Correlation to identify the most impactful features.
- Multi-Model Evaluation: Trains and compares 8+ algorithms, including Random Forest, XGBoost, and Naive Bayes.
- Web Interface: A Flask-based web application for easy user interaction.
- Robust Logging: Detailed logs are generated for every step of the pipeline to assist in debugging and monitoring.

## ⚙️ Project Structure
File,Description
- main.py  ---> The entry point that orchestrates the entire ML pipeline.
- app.py   ---> Flask application script to serve the model via a web UI.
- all_model.py ---> Contains the logic for training and evaluating multiple classifiers.
- missing_value.py ---> Implements Random Sample Imputation for handling missing data.
- feature_selection.py---> Logic for removing constant, quasi-constant, and weak features.
- balance_data.py ---> Uses SMOTE to handle class imbalance in the training set.
- log_code.py ---> Custom logging configuration for the project.
- Heart_Disease.pkl ---> "The serialized ""Best Model"" (Naive Bayes) ready for production."

## 📊 Machine Learning Pipeline
- Data Loading: Loads clinical data from heart (1).csv.
- Imputation: Addresses missing values using random sampling to preserve distribution.
- Feature Selection: * Removes features with zero or low variance.Drops features with high p-   values ($> 0.05$) via Pearson correlation.
- Scaling & Balancing: Standardizes features and applies SMOTE to ensure the model isn't biased toward a specific outcome.
- Model Selection: Compares ROC-AUC scores across various models and exports the best performer.

## 💻 How to Run
Prerequisites
Ensure you have Python installed, then install the required dependencies:
Bash
pip install flask numpy pandas scikit-learn xgboost imbalanced-learn
1. Train the Model
Run the main pipeline to process the data and generate the .pkl files:
`python main.py`
2. Start the Web App
Launch the Flask server to interact with the model:
Bash
`python app.py`
Open your browser and navigate to http://127.0.0.1:5000.






