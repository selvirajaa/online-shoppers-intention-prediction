# ===========================================
# Import Libraries
# ===========================================
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ===========================================
# Page Configuration
# ===========================================
st.set_page_config(
    page_title = "Online Shoppers Intention Prediction",
    page_icon = "🛒",
    layout = "wide"
)

# ===========================================
# Load Model
# ===========================================
@st.cache_resource
def load_model():
    try:
        best_model = joblib.load("online_shoppers_best_model.pkl")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

    return best_model


best_model = load_model()


# ===========================================
# Load Evaluation Results
# ===========================================
@st.cache_resource
def load_evaluation_results():
    try:
        evaluation_data = joblib.load(
            "online_shoppers_evaluation.pkl"
        )
    except Exception as e:
        st.error(f"Error loading evaluation results: {e}")
        st.stop()

    return evaluation_data


evaluation_results = load_evaluation_results()
# =============================================
# Title
# =============================================
st.title("🛒 Online Shoppers Intention Prediction")
st.write(
    """
    This application predicts whether an online shopping session
    will result in **Revenue** or **No Revenue** using Machine Learning.
    """
)

# ===========================================
# Navigation Buttons
# ===========================================

if "page" not in st.session_state:
    st.session_state.page = "🏠 Home"

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    if st.button("🏠 Home", use_container_width=True):
        st.session_state.page = "🏠 Home"

with col2:
    if st.button("🔮 Prediction", use_container_width=True):
        st.session_state.page = "🔮 Prediction"

with col3:
    if st.button("📊 Model Performance", use_container_width=True):
        st.session_state.page = "📊 Model Performance"

with col4:
    if st.button("📈 Feature Importance", use_container_width=True):
        st.session_state.page = "📈 Feature Importance"

with col5:
    if st.button("ℹ️ About", use_container_width=True):
        st.session_state.page = "ℹ️ About"

page = st.session_state.page

st.divider()

# =========================================================
# SIDEBAR NAVIGATION
# =========================================================
with st.sidebar:

    st.divider()

    st.subheader("📖 About the App")

    st.caption(
        "This application uses a Machine Learning Classification models "
        "to online shopping session samples will result in **Revenue** or **No Revenue**"
    )

    st.caption(
        "Enter the 17 feature values to generate a model prediction."
    )

    st.caption(" 🛒 Online Shoppers Intention Prediction")
    st.divider()

    st.subheader("👤 Developer")
    st.write("**Name:** SelviRaja")
    st.write("**Role:** Machine Learning Learner")
    st.write("**Project:** 🛒 Online Shoppers Intention Prediction")

    st.caption("Built with Python, Scikit-learn & Streamlit")

# ============================================
# Home
# ============================================
if page == "🏠 Home":
    st.header("Online Shoppers Intention")
    st.write(
        """
        Online shopping behavior can be analyzed using machine learning
        to predict whether a visitor is likely to generate revenue.

        This project compares multiple classification algorithms and
        selects the best-performing model based on ROC-AUC.
        """
    )
    st.subheader("Models Used")

    models = [
        "Logistic Regression",
        "Naive Bayes",
        "K-Nearest Neighbors",
        "Decision Tree",
        "Random Forest",
        "Gradient Boosting",
        "AdaBoost",
        "XGBoost"
    ]

    for model in models:
        st.write("•", model)

# ==============================================
# PREDICTION
# ==============================================


elif page == "🔮 Prediction":

    st.header("🔮 Predict Customer Revenue")

    st.write(
        "Enter the customer's online shopping session information."
    )

    with st.form("prediction_form"):

        st.subheader("📄 Page Visit Information")

        col1, col2, col3 = st.columns(3)

        with col1:
            administrative = st.number_input(
                "Administrative",
                min_value=0,
                value=0
            )

            informational = st.number_input(
                "Informational",
                min_value=0,
                value=0
            )

            product_related = st.number_input(
                "ProductRelated",
                min_value=0,
                value=10
            )

        with col2:
            administrative_duration = st.number_input(
                "Administrative_Duration",
                min_value=0.0,
                value=0.0
            )

            informational_duration = st.number_input(
                "Informational_Duration",
                min_value=0.0,
                value=0.0
            )

            product_related_duration = st.number_input(
                "ProductRelated_Duration",
                min_value=0.0,
                value=100.0
            )

        with col3:
            bounce_rates = st.number_input(
                "BounceRates",
                min_value=0.0,
                max_value=1.0,
                value=0.02
            )

            exit_rates = st.number_input(
                "ExitRates",
                min_value=0.0,
                max_value=1.0,
                value=0.05
            )

            page_values = st.number_input(
                "PageValues",
                min_value=0.0,
                value=0.0
            )

        st.subheader("🛍️ Shopping Session Details")

        col4, col5, col6 = st.columns(3)

        with col4:
            special_day = st.number_input(
                "SpecialDay",
                min_value=0.0,
                max_value=1.0,
                value=0.0
            )

            month = st.selectbox(
                "Month",
                [
                    "Feb", "Mar", "May", "June",
                    "Jul", "Aug", "Sep", "Oct",
                    "Nov", "Dec"
                ]
            )

        with col5:
            operating_systems = st.number_input(
                "OperatingSystems",
                min_value=1,
                value=2
            )

            browser = st.number_input(
                "Browser",
                min_value=1,
                value=2
            )

        with col6:
            region = st.number_input(
                "Region",
                min_value=1,
                value=1
            )

            traffic_type = st.number_input(
                "TrafficType",
                min_value=1,
                value=2
            )

        visitor_type = st.selectbox(
            "VisitorType",
            [
                "Returning_Visitor",
                "New_Visitor",
                "Other"
            ]
        )

        weekend = st.selectbox(
            "Weekend",
            [False, True],
            format_func=lambda x: "Yes" if x else "No"
        )

        submitted = st.form_submit_button(
            "🔮 Predict Revenue",
            use_container_width=True
        )

    if submitted:

        input_data = pd.DataFrame([{
            "Administrative": administrative,
            "Administrative_Duration": administrative_duration,
            "Informational": informational,
            "Informational_Duration": informational_duration,
            "ProductRelated": product_related,
            "ProductRelated_Duration": product_related_duration,
            "BounceRates": bounce_rates,
            "ExitRates": exit_rates,
            "PageValues": page_values,
            "SpecialDay": special_day,
            "Month": month,
            "OperatingSystems": operating_systems,
            "Browser": browser,
            "Region": region,
            "TrafficType": traffic_type,
            "VisitorType": visitor_type,
            "Weekend": weekend
        }])

        try:
            prediction = best_model.predict(input_data)[0]

            st.subheader("Prediction Result")

            if prediction in [True, 1, "True"]:
                st.success("🟢 Revenue: The session is predicted to generate revenue.")
            else:
                st.warning("🔴 No Revenue: The session is predicted not to generate revenue.")

        except Exception as e:
            st.error(f"Prediction error: {e}")



# ==============================================
# MODEL PERFORMANCE
# ==============================================

elif page == "📊 Model Performance":

    st.header("📊 Model Performance")

    # Load saved evaluation values
    best_model_name = evaluation_results["best_model_name"]
    accuracy = evaluation_results["accuracy"]
    roc_auc = evaluation_results["roc_auc"]

    confusion = np.array(
        evaluation_results["confusion_matrix"]
    )
    fpr = evaluation_results["fpr"]
    tpr = evaluation_results["tpr"]

    # 1. Display model name and metrics
    st.subheader("Best Model: " + best_model_name)

    col1, col2 = st.columns(2)

    col1.metric(
        "Accuracy",
        f"{accuracy * 100:.2f}%"
    )

    col2.metric(
        "ROC-AUC",
        f"{roc_auc:.4f}"
    )

    st.divider()

    # 2. Confusion Matrix
    st.subheader("Confusion Matrix")

    fig, ax = plt.subplots(figsize=(6, 4))

    sns.heatmap(
        confusion,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Revenue", "Revenue"],
        yticklabels=["No Revenue", "Revenue"],
        ax=ax
    )

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    ax.set_title("Confusion Matrix")

    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "Rows represent actual outcomes; columns represent "
        "the model's predictions."
    )

    st.divider()

    # 3. ROC Curve
    st.subheader("ROC Curve")

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(
        fpr,
        tpr,
        label=f"{best_model_name} (AUC = {roc_auc:.4f})"
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random Classifier"
    )

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    ax.grid(alpha=0.3)

    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    # 4. Model Comparison
    st.subheader("Model Comparison")

    comparison_df = evaluation_results["comparison_df"]

    # Display comparison table
    st.write("Performance of all trained models")

    st.dataframe(
        comparison_df.style.format("{:.3f}"),
        use_container_width=True

    )

    # Plot comparison chart
    st.write("Compare Accuracy and ROC-AUC")

    metrics = [
        "CV Accuracy",
        "Test Accuracy",
        "ROC_AUC"
    ]

    fig, ax = plt.subplots(figsize=(12, 6))

    comparison_df[metrics].plot(
        kind="bar",
        ax=ax
    )

    ax.set_title("Machine Learning Model Comparison")
    ax.set_xlabel("Machine Learning Models")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.legend(title="Metrics")
    ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================
elif page == "📈 Feature Importance":

    st.title("📈 Feature Importance")

    st.write(
        "This page shows which features most affect "
        "the model's ROC-AUC score."
    )

    # Load saved feature importance results
    importance_df = evaluation_results["importance_df"]

    # Sort features from highest to lowest importance
    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    # Display the complete table
    st.subheader("Feature Importance Table")

    st.dataframe(
        importance_df,
        width="stretch"
    )

    # Display the top 15 features
    st.subheader("Top 15 Important Features")

    top_features = importance_df.head(15).sort_values(
        by="Importance",
        ascending=True
    )

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.barh(
        top_features["Feature"],
        top_features["Importance"],
        xerr=top_features["Std"]
    )

    ax.set_xlabel("Decrease in ROC-AUC")
    ax.set_ylabel("Feature")
    ax.set_title("Top 15 Features by Permutation Importance")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "Higher positive importance means shuffling that feature "
        "reduced the model's ROC-AUC more. Values near zero suggest "
        "little measured impact; negative values can occur."
    )


# ============================================================
# ABOUT
# ============================================================

elif page == "ℹ️ About":

    st.header("ℹ️ About This Project")

    st.write(
        """
        **Project:** Online Shoppers Intention Prediction

        **Task:** Binary Classification

        **Target:** Revenue

        **Goal:** Predict whether an online shopping session
        results in revenue.

        The project includes preprocessing, hyperparameter tuning,
        model comparison, ROC-AUC evaluation, permutation importance,
        and model deployment using Streamlit.
        """
    )





















