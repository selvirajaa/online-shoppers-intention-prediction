# Import Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.inspection import permutation_importance
from sklearn.ensemble import (RandomForestClassifier,
                              AdaBoostClassifier,
                              GradientBoostingClassifier
                              )
from sklearn.metrics import(
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve
    )

# Load Dataset
df = pd.read_csv("online_shoppers_intention.csv")
print("Dataset Loaded Successfully!")

# First 5 Rows
print("\n----First 5 Rows----")
print(df.head())

# Dataset Information
print("\n----Dataset Information----")
print(df.info())

# Dataset Shape
print("\n----Dataset Shape----")
print(df.shape)

# Duplicate Values
print("\n----Duplicate Values----")
print(df.duplicated())

# Statistical Summary
print("\n----Statistical Summary----")
print(df.describe())

# Missing Values
print("\n----Missing Values----")
print(df.isnull().sum())

# Columns Names
print("\n----Columns Names----")
print(df.columns.tolist())

# Data Visualization
df["Revenue"].value_counts().plot(kind="bar")
plt.title("Revenue Distribution")
plt.xlabel("Revenue")
plt.ylabel("Count")
plt.show()

# Remove unnecessary Columns
X = df.drop("Revenue", axis=1)
y = df["Revenue"]

# Identify Numerical Features and Categorical Features

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

# Numerical Features
print("\nNumerical Features")
print(numeric_features)

# Categorical_Features
print("\nCategorical Features")
print(categorical_features)

# Preprocessing
# Numerical Features
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

# Categorical Features
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

# Pipeline
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# X Shape
print("\nX Shape")
print(X.shape)

# Y Shape
print("\ny Shape")
print(y.shape)

# Train test Split
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.20,
                                                    random_state=42,
                                                    stratify=y
                                                    )
# Training Samples
print("\nTraining Samples")
print(X_train.shape[0])

# Testing samples
print("\nTesting samples")
print(X_test.shape[0])

models = {
    "Logistic Regression" : LogisticRegression(),
    "Naive Bayes" : GaussianNB(),
    "KNeighborsClassifier" : KNeighborsClassifier(),
    "Decision Tree" : DecisionTreeClassifier(),
    "Random Forest" : RandomForestClassifier(),
    "Gradient Boost" : GradientBoostingClassifier(),
    "Ada Boost" : AdaBoostClassifier(estimator=DecisionTreeClassifier(random_state=42),
    random_state=42),
    "XG Boost" : XGBClassifier()
}

param_grids = {
    "Logistic Regression" : {
        "model__C" : [0.1, 1, 10],
        "model__solver" : ["liblinear", "lbfgs"]
    },
    "Naive Bayes" : {
        "model__var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
    },

    "KNeighborsClassifier" : {
        "model__n_neighbors" : [3,5,7,9,11,13,15,17],
        "model__weights" : ["uniform", "distance"],
        "model__p" : [1,2]
    },
    "Decision Tree" : {
        "model__criterion" : ["gini", "entropy"],
        "model__max_depth" : [3,5,7,10,None],
        "model__min_samples_split" : [2,5,10],
        "model__min_samples_leaf" : [1,2,5]
    },
    "Random Forest" : {
        "model__n_estimators" : [100,150,200],
        "model__max_depth" : [5,10,None],
        "model__min_samples_split" : [2,5],
        "model__min_samples_leaf" : [1,2]
    },
    "Gradient Boost" : {
        "model__n_estimators" : [100,200],
        "model__max_depth" : [3,5],
        "model__min_samples_split" : [2,5],
        "model__learning_rate" : [0.01,0.1]
    },
    "Ada Boost" : {
        "model__n_estimators": [50, 100, 200],
        "model__learning_rate": [0.01, 0.1, 1.0],
        "model__estimator__max_depth": [1, 2, 3],
        "model__estimator__min_samples_split": [2, 5],
        "model__estimator__min_samples_leaf": [1, 2]

    },
    "XG Boost": {
        "model__n_estimators": [100, 200],
        "model__max_depth": [3, 5],
        "model__min_child_weight": [1, 3],
        "model__learning_rate": [0.01, 0.1],
        "model__subsample": [0.8, 1.0],
        "model__colsample_bytree": [0.8, 1.0],
        "model__reg_alpha": [0, 0.1],
        "model__reg_lambda": [1, 2]

    }
}
# Train and Evaluation
results = {}
trained_models = {}
roc_data = {}
pr_data = {}

for name, model in models.items():
    print(f"\n====={name}=====")
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    grid_search =GridSearchCV(
        estimator=pipeline,
        param_grid=param_grids[name],
        cv = 5,
        scoring= "accuracy",
        n_jobs=-1
    )
    grid_search.fit(X_train,y_train)

    print("\nBest Parameters:")
    print(grid_search.best_params_)

    print("\nBest CV Accuracy")
    print(grid_search.best_score_)

    # Best tuned model
    best_model = grid_search.best_estimator_

    # Class Prediction
    y_pred = best_model.predict(X_test)

    # Probability Prediction
    y_proba = best_model.predict_proba(X_test)[:,1]

    # Accuracy
    accuracy = accuracy_score(y_test,y_pred)
    print("\nAccuracy:",accuracy)

    # Confusion Matrix
    cm = confusion_matrix(y_test,y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # Confusion Matrix Visualization
    plt.figure(figsize=(5, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Revenue", "Revenue"],
        yticklabels=["No Revenue", "Revenue"]
    )

    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    # Classification
    classification = classification_report(y_test,y_pred)
    print("\nClassification Report")
    print(classification)

    # ROC-AUC
    roc_auc = roc_auc_score(y_test,y_proba)
    print("\nROC_AUC:",roc_auc)

    # ROC Curve Data
    fpr, tpr, thresholds = roc_curve (y_test, y_proba)
    roc_data[name] = {
        "fpr" : fpr,
        "tpr" : tpr,
        "roc_auc" : roc_auc
    }

    # Precision_recall Curve
    precision, recall, thresholds = precision_recall_curve(
        y_test,
        y_proba
    )

    pr_data[name] = {
        "precision": precision,
        "recall": recall
    }

    # Store Results
    results[name] = {
        "CV Accuracy" : grid_search.best_score_,
        "Test Accuracy" : accuracy,
        "ROC_AUC" : roc_auc
    }

    # Saved Best Tuned Model
    trained_models[name] = best_model

plt.figure(figsize=(10, 7))

for name, data in roc_data.items():
    plt.plot(
        data["fpr"],
        data["tpr"],
        label=f"{name} (AUC = {data['roc_auc']:.3f})"
    )

# Random classifier line
plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.title("ROC Curve - Model Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# Precision Recall Curve
plt.figure(figsize=(10, 7))

for name, data in pr_data.items():
    plt.plot(
        data["recall"],
        data["precision"],
        label=name
    )

plt.title("Precision-Recall Curve - Model Comparison")
plt.xlabel("Recall")
plt.ylabel("Precision")

plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
# Model Comparison
comparison_df = pd.DataFrame(results).T
print("\n=====Model Comparison=====")
print(comparison_df)

# ROC AUC Comparison
roc_comparison = comparison_df[["ROC_AUC"]]

print("\n=====ROC Comparison=====")
print(roc_comparison)

# ROC Comparison Plot
roc_comparison.plot(
    kind = "bar",
    figsize = (8,5),
    legend = False
)
plt.title("ROC-AUC Comparison")
plt.ylabel("ROC-AUC")
plt.xlabel("Model")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Find the best model Based on ROC-AUC
best_model_name = comparison_df["ROC_AUC"].idxmax()
best_roc_auc = comparison_df["ROC_AUC"].max()

# Get the actual best trained model
best_model = trained_models[best_model_name]

print("\n=====Best Model=====")
print("\nBest Model:",best_model_name)
print("\nBest ROC_AUC:",best_roc_auc)

# PERMUTATION IMPORTANCE
perm_result = permutation_importance(
    best_model,
    X_test,
    y_test,
    n_repeats=10,
    random_state=42,
    scoring="roc_auc",
    n_jobs=-1
)

# Original feature names
feature_names = X_test.columns.tolist()

# Check lengths
print("\nNumber of feature names:", len(feature_names))
print("Number of importance values:", len(perm_result.importances_mean))

# Create DataFrame
permutation_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": perm_result.importances_mean,
    "Std": perm_result.importances_std
})

# Sort by importance
permutation_df = permutation_df.sort_values(
    by="Importance",
    ascending=False
)

print("\n===== Permutation Importance =====")
print(permutation_df.head(15))

# Permutation Importance Visualization
top_perm = permutation_df.head(15)

plt.figure(figsize=(10, 6))

plt.barh(
    top_perm["Feature"],
    top_perm["Importance"]
)

plt.title(f"Permutation Importance - {best_model_name}")
plt.xlabel("Decrease in ROC-AUC")
plt.ylabel("Feature")

plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()

# Get Best Trained Model
best_model = trained_models[best_model_name]

if hasattr(best_model.named_steps["model"], "feature_importances_"):
    print("\nFeature Importance is available.")
else:
    print("\nFeature Importance is not available for this model.")

feature_names = best_model.named_steps["preprocessor"].get_feature_names_out()

model = best_model.named_steps["model"]

if hasattr(model, "feature_importances_"):

    importances = model.feature_importances_

    feature_importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    })

    feature_importance_df = feature_importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    print("\n===== Feature Importance =====")
    print(feature_importance_df.head(15))

    top_features = feature_importance_df.head(15)

    plt.figure(figsize=(10, 6))

    plt.barh(
        top_features["Feature"],
        top_features["Importance"]
    )

    plt.title(f"Top 15 Feature Importance - {best_model_name}")
    plt.xlabel("Importance")
    plt.ylabel("Feature")

    plt.gca().invert_yaxis()

    plt.tight_layout()
    plt.show()


# ===========================================
# Save Model Evaluation Results
# ===========================================

# Predictions from the best model
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

# ROC curve for the best model
fpr, tpr, thresholds = roc_curve(y_test, y_proba)

# Save all evaluation results
evaluation_results = {
    "best_model_name": best_model_name,
    "comparison_df": comparison_df,
    "accuracy": comparison_df.loc[
        best_model_name, "Test Accuracy"
    ],
    "roc_auc": comparison_df.loc[
        best_model_name, "ROC_AUC"
    ],
    "confusion_matrix": cm,
    "fpr": fpr,
    "tpr": tpr,
    "roc_data": roc_data,
    "importance_df": permutation_df
}

joblib.dump(
    evaluation_results,
    "online_shoppers_evaluation.pkl"
)
joblib.dump(
    best_model,
    "online_shoppers_best_model.pkl"
)

print("Best model saved successfully!")
print("Model evaluation results saved successfully!")
print("Best Model:", best_model_name)
print("Accuracy:", evaluation_results["accuracy"])
print("ROC-AUC:", evaluation_results["roc_auc"])








