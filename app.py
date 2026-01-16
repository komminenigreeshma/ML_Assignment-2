import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Title
st.title("📊 ML Assignment 2 - Letter Recognition")

# Load models
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
model_choice = st.sidebar.selectbox("Select Model", list(models.keys()))
uploaded_file = st.sidebar.file_uploader("Upload Test CSV", type=["csv"])

# Option to download the test file (generated in training)
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

if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)
    X_test = test_df.drop("letter", axis=1)
    y_test = le.transform(test_df["letter"])
    X_test_scaled = scaler.transform(X_test)

    # Predict
    model = models[model_choice]
    y_pred = model.predict(X_test_scaled)

    # Metrics
    st.subheader(f"📈 Results for {model_choice}")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
    col2.metric("Precision", f"{precision_score(y_test, y_pred, average='macro'):.4f}")
    col3.metric("Recall", f"{recall_score(y_test, y_pred, average='macro'):.4f}")
    col4.metric("F1 Score", f"{f1_score(y_test, y_pred, average='macro'):.4f}")

    # Confusion Matrix Heatmap
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=False, cmap="Blues", ax=ax)
    st.pyplot(fig)