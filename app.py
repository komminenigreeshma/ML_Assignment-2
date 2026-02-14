import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, matthews_corrcoef
)

# Page config
st.set_page_config(page_title="ML Assignment 2", page_icon="📊", layout="wide")

# Dark theme styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #1e1e2f; /* deep navy background */
        color: #f0f0f0; /* light text */
    }
    h1, h2, h3, h4, h5, h6 {
        color: #00c9a7; /* teal accent for headers */
    }
    .css-1d391kg, .css-1v3fvcr {
        background-color: #2a2a3d !important; /* darker cards/containers */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Gradient header banner (dark-friendly colors)
st.markdown(
    """
    <div style="background: linear-gradient(to right, #141e30, #243b55);
                padding: 20px; border-radius: 8px; text-align: center;">
        <h1 style="color: #00c9a7; font-size: 36px;">ML Assignment 2 - Letter Recognition</h1>
        <p style="color: #f0f0f0; font-size: 18px;">Interactive evaluation of ML models on the UCI dataset</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Load models and preprocessors
scaler = joblib.load("models/scaler.pkl")
le = joblib.load("models/label_encoder.pkl")
models = {
    "Logistic Regression": joblib.load("models/log_reg.pkl"),
    "Decision Tree": joblib.load("models/decision_tree.pkl"),
    "KNN": joblib.load("models/knn.pkl"),
    "Naive Bayes": joblib.load("models/naive_bayes.pkl"),
    "Random Forest": joblib.load("models/random_forest.pkl"),
    "XGBoost": joblib.load("models/xgboost.pkl"),
}

# Sidebar

# Quick Guide (smaller size, collapsed by default)
st.sidebar.markdown(
    "<h4 style='color:#00c9a7; margin-bottom:0;'>📖 Quick Guide</h4>",
    unsafe_allow_html=True
)
with st.sidebar.expander("💡 How to Use", expanded=False):  # collapsed initially
    st.write(
        "1. Upload a test CSV file.\n"
        "2. Select a model.\n"
        "3. View metrics, confusion matrix, and classification report.\n"
        "4. Download predictions if needed."
    )

# Sample file download
try:
    with open("dataset/test.csv", "rb") as f:
        st.sidebar.download_button(
            label="⬇️ Download Test File",
            data=f,
            file_name="test.csv",
            mime="text/csv"
        )
except FileNotFoundError:
    st.sidebar.warning("No test.csv found. Run Training_model.py to generate it.")

# Model & Data Settings
st.sidebar.markdown(
    "<h4 style='color:#00c9a7; margin-bottom:0;'>🎛️ Model & Data Settings</h4>",
    unsafe_allow_html=True
)
with st.sidebar.expander("🧩 Choose Model & Upload Data", expanded=True):
    model_choice = st.selectbox("Select Model", ["-- Select Model --"] + list(models.keys()))
    uploaded_file = st.file_uploader("Upload Test CSV", type=["csv"])

# Main logic
if uploaded_file is not None and model_choice != "-- Select Model --":
    test_df = pd.read_csv(uploaded_file)

    st.markdown("### 📂 Uploaded Dataset Preview")
    st.dataframe(test_df.head())

    if "letter" not in test_df.columns:
        st.error("Uploaded file must contain a 'letter' column for labels.")
        st.stop()

    expected_features = ["x-box","y-box","width","height","onpix","x-bar","y-bar",
                         "x2bar","y2bar","xybar","x2ybr","xy2br","x-ege","xegvy","y-ege","yegvx"]
    if list(test_df.drop("letter", axis=1).columns) != expected_features:
        st.error("Feature mismatch. Expected features are:")
        st.write(expected_features)
        st.stop()

    X_test = test_df.drop("letter", axis=1)
    y_test = le.transform(test_df["letter"])
    X_test_scaled = scaler.transform(X_test)

    model = models[model_choice]
    y_pred = model.predict(X_test_scaled)

    tab1, tab2, tab3, tab4 = st.tabs(["📈 Metrics", "📊 Confusion Matrix", "📑 Classification Report", "📊 Per-Class Performance"])

    with tab1:
        st.subheader(f"Results for {model_choice}")
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
        col2.metric("Precision", f"{precision_score(y_test, y_pred, average='macro'):.4f}")
        col3.metric("Recall", f"{recall_score(y_test, y_pred, average='macro'):.4f}")
        col4.metric("F1 Score", f"{f1_score(y_test, y_pred, average='macro'):.4f}")
        col5.metric("AUC", f"{roc_auc_score(pd.get_dummies(y_test), pd.get_dummies(y_pred), average='macro'):.4f}")
        col6.metric("MCC", f"{matthews_corrcoef(y_test, y_pred):.4f}")

    with tab2:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm", ax=ax,
                    xticklabels=le.classes_, yticklabels=le.classes_,
                    cbar_kws={'label': 'Number of Samples'})
        ax.set_xlabel("Predicted", fontsize=12, fontweight="bold", color="#f0f0f0")
        ax.set_ylabel("True", fontsize=12, fontweight="bold", color="#f0f0f0")
        plt.xticks(rotation=45, color="#f0f0f0")
        plt.yticks(rotation=0, color="#f0f0f0")
        st.pyplot(fig)

    with tab3:
        st.subheader("Detailed Classification Report")
        report_dict = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()
        st.dataframe(report_df.style.highlight_max(axis=0))
        st.download_button("📥 Download Report", report_df.to_csv().encode("utf-8"), "classification_report.csv", "text/csv")

    with tab4:
        st.subheader("Per-Class F1 Scores")
        metrics_df = pd.DataFrame(report_dict).transpose().iloc[:-3]  # exclude avg rows

        fig, ax = plt.subplots(figsize=(14, 6))
        # Use single accent color for all bars
        sns.barplot(x=metrics_df.index, y=metrics_df["f1-score"], color="#4682B4", ax=ax)

        # Force x-axis labels (A–Z) to be visible and styled
        plt.xticks(rotation=45, color="#4682B4", fontsize=10)
        plt.ylabel("F1 Score", color="#4682B4", fontsize=12)
        plt.title("Per-Class F1 Scores (A–Z)", color="#4682B4", fontsize=14)

        # Ensure tick labels (letters) are not cut off
        plt.tight_layout()
        st.pyplot(fig)

    output_df = test_df.copy()
    output_df["Predicted"] = le.inverse_transform(y_pred)
    st.download_button(
        label="📥 Download Predictions",
        data=output_df.to_csv(index=False),
        file_name="predictions.csv",
        mime="text/csv"
    )

elif uploaded_file is not None and model_choice == "-- Select Model --":
    st.info("ℹ️ Please select a model from the dropdown to run predictions.")