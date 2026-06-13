import streamlit as st
import pandas as pd
import joblib

# Load saved model
model = joblib.load("fraud_detection_model.pkl")

st.set_page_config(page_title="Credit Card Fraud Detection")

st.title("💳 Credit Card Fraud Detection")
st.write("Upload a CSV file containing transaction data.")

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data")
    st.dataframe(df.head())

    # Remove target column if present
    if "Class" in df.columns:
        df = df.drop("Class", axis=1)

    # Required features
    required_columns = [
        'Time','V1','V2','V3','V4','V5','V6','V7','V8','V9',
        'V10','V11','V12','V13','V14','V15','V16','V17','V18',
        'V19','V20','V21','V22','V23','V24','V25','V26','V27',
        'V28','Amount'
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        st.error(
            f"Missing columns: {missing_columns}"
        )
        st.stop()

    # Arrange columns in exact order
    df = df[required_columns]

    if st.button("Predict Fraud"):

        predictions = model.predict(df)

        result_df = df.copy()

        result_df["Prediction"] = predictions

        result_df["Prediction"] = result_df["Prediction"].map({
            0: "Normal Transaction",
            1: "Fraud Transaction"
        })

        st.subheader("Prediction Results")
        st.dataframe(result_df)

        fraud_count = (
            result_df["Prediction"]
            == "Fraud Transaction"
        ).sum()

        normal_count = (
            result_df["Prediction"]
            == "Normal Transaction"
        ).sum()

        st.success(
            f"Normal Transactions: {normal_count}"
        )

        st.error(
            f"Fraud Transactions: {fraud_count}"
        )

        csv = result_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Results",
            data=csv,
            file_name="fraud_predictions.csv",
            mime="text/csv"
        )