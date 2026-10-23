# 🛒 Online Shoppers Intention Prediction

## Project Overview

This project uses Machine Learning to predict whether an online shopping session will result in **Revenue** or **No Revenue**.

The project includes data preprocessing, model training, hyperparameter tuning, model evaluation, and an interactive Streamlit web application.

## Objectives

- Analyse online shopper behaviour.
- Prepare numerical and categorical features for Machine Learning.
- Train and compare multiple classification models.
- Evaluate model performance using appropriate metrics.
- Predict whether an online shopping session will generate revenue.
- Display model performance and feature importance in a web application.

## Dataset

The project uses the Online Shoppers Purchasing Intention dataset.

The target variable is `Revenue`:
- `True`: The shopping session resulted in revenue.
- `False`: The shopping session did not result in revenue.

## Machine Learning Workflow

1. Load and explore the dataset.
2. Clean and preprocess the data.
3. Handle numerical and categorical features.
4. Split the data into training and testing sets.
5. Train and tune classification models using GridSearchCV.
6. Compare models using evaluation metrics.
7. Save the selected model and evaluation results.
8. Build an interactive Streamlit application.

## Models Evaluated

The project evaluates several classification algorithms, including:

- Gaussian Naive Bayes
- K-Nearest Neighbors (KNN)
- Logistic Regression
- Decision Tree
- Random Forest
- AdaBoost
- Gradient Boosting

Include other models only if they are present in your final training code.

## Model Evaluation

The project uses:
- Accuracy
- Confusion Matrix
- ROC-AUC
- ROC Curve
- Model Comparison
- Permutation Feature Importance

The Streamlit application displays the evaluation results and the most influential input features.

## Streamlit Application

The application includes five pages:

- **Home:** Project introduction.
- **Prediction:** Predict Revenue or No Revenue for a shopping session.
- **Model Performance:** View evaluation metrics, confusion matrix, ROC curve, and model comparison.
- **Feature Importance:** Explore permutation importance results.
- **About:** Project and dataset information.

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- Joblib
- Streamlit

## How to Run the Project

1. Clone or download the project.
2. Create and activate a Python virtual environment.
3. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

4. Ensure the saved model and evaluation files are available in the project directory.
5. Start the Streamlit application:

   ```bash
   streamlit run app.py
   ```

## Project Files

- `app.py` — Streamlit application.
- `end_to_end.py` — Model training and evaluation workflow.
- `online_shoppers_best_model.pkl` — Saved trained model.
- `online_shoppers_evaluation.pkl` — Saved evaluation results.
- `requirements.txt` — Python dependencies.

## Disclaimer

This project is intended for educational purposes and demonstrates how Machine Learning can be used to analyse online shopping behaviour. Predictions are estimates and are not guaranteed to represent actual customer behaviour.