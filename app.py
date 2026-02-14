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

# Title
st.title("📊 ML Assignment 2 - Letter Recognition")
st.markdown("An interactive app to evaluate multiple ML models on the UCI Letter Recognition dataset.")

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

# Sidebar controls
st.sidebar.header("⚙️ Controls")
model_choice = st.sidebar.selectbox("Select Model", ["-- Select Model --"] + list(models.keys()))
uploaded_file = st.sidebar.file_uploader("Upload Test CSV", type=["csv"])

# Option to download sample test file
try:
    with open("dataset/test.csv", "rb") as f:
        st.sidebar.download_button(
            label="⬇️ Download Sample Test File",
            data=f,
            file_name="test.csv",
            mime="text/csv"
        )
except FileNotFoundError:
    st.sidebar.warning("No test.csv found. Run Training_model.py to generate it.")

# Run only if a file is uploaded and a model is selected
if uploaded_file is not None and model_choice != "-- Select Model --":
    test_df = pd.read_csv(uploaded_file)

    # Preview dataset
    st.subheader("📂 Uploaded Dataset Preview")
    st.dataframe(test_df.head())

    # Check for 'letter' column
    if "letter" not in test_df.columns:
        st.error("Uploaded file must contain a 'letter' column for labels.")
        st.write("Columns found:", list(test_df.columns))
        st.stop()

    # Feature alignment
    expected_features = ["x-box","y-box","width","height","onpix","x-bar","y-bar",
                         "x2bar","y2bar","xybar","x2ybr","xy2br","x-ege","xegvy","y-ege","yegvx"]
    if list(test_df.drop("letter", axis=1).columns) != expected_features:
        st.error("Feature mismatch. Expected features are:")
        st.write(expected_features)
        st.stop()

    # Preprocess
    X_test = test_df.drop("letter", axis=1)
    y_test = le.transform(test_df["letter"])
    X_test_scaled = scaler.transform(X_test)

    # Predict
    model = models[model_choice]
    y_pred = model.predict(X_test_scaled)

    # Tabs for results
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Metrics", "📊 Confusion Matrix", "📑 Classification Report", "📊 Per-Class Metrics"])

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
        view_choice = st.radio("Choose View", ["Heatmap", "Raw Matrix", "Normalized"])
        cm = confusion_matrix(y_test, y_pred)
        if view_choice == "Heatmap":
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(cm, annot=False, cmap="Blues", ax=ax,
                        xticklabels=le.classes_, yticklabels=le.classes_)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            st.pyplot(fig)
        elif view_choice == "Raw Matrix":
            cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
            st.dataframe(cm_df)
        elif view_choice == "Normalized":
            cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
            cm_df = pd.DataFrame(cm_norm, index=le.classes_, columns=le.classes_)
            st.dataframe(cm_df.style.format("{:.2f}"))

    with tab3:
        st.subheader("Detailed Classification Report")
        report_dict = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()
        st.dataframe(report_df.style.highlight_max(axis=0))
        st.download_button("📥 Download Report", report_df.to_csv().encode("utf-8"), "classification_report.csv", "text/csv")

    with tab4:
        st.subheader("Per-Class Metrics (Precision, Recall, F1)")
        report_dict = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
        metrics_df = pd.DataFrame(report_dict).transpose().iloc[:-3]  # exclude avg rows
        fig, ax = plt.subplots(figsize=(14, 6))
        metrics_df[["precision", "recall", "f1-score"]].plot(kind="bar", ax=ax)
        plt.xticks(rotation=45)
        plt.ylabel("Score")
        plt.title("Per-Class Metrics")
        st.pyplot(fig)

    # Option to download predictions
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