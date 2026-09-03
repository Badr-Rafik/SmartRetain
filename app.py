from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from data_preprocessing import add_features

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "smartretain_model.joblib"
PREPROCESSOR_PATH = PROJECT_ROOT / "smartretain_preprocessor.joblib"


@st.cache_resource
def load_model_assets():
    
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    return model, preprocessor


def build_customer_dataframe() -> pd.DataFrame:

    customer_data = {
        "CreditScore": st.number_input("Credit Score", min_value=300, max_value=900, value=650),
        "Geography": st.selectbox("Geography", ["France", "Spain", "Germany"]),
        "Gender": st.selectbox("Gender", ["Female", "Male"]),
        "Age": st.number_input("Age", min_value=18, max_value=100, value=40),
        "Tenure": st.number_input("Tenure (Years)", min_value=0, max_value=20, value=5),
        "Balance": st.number_input(
            "Balance",
            min_value=0.0,
            max_value=200000.0,
            value=50000.0,
            format="%.2f",
        ),
        "NumOfProducts": st.number_input("Number of Products", min_value=1, max_value=4, value=2),
        "HasCrCard": st.selectbox("Has Credit Card?", [0, 1]),
        "IsActiveMember": st.selectbox("Is Active Member?", [0, 1]),
        "EstimatedSalary": st.number_input(
            "Estimated Salary",
            min_value=0.0,
            max_value=200000.0,
            value=70000.0,
            format="%.2f",
        ),
    }

    return pd.DataFrame([customer_data])


def main():
    """
    Build the local Streamlit interface and run the churn prediction.
    """
    st.set_page_config(page_title="SmartRetain", page_icon="", layout="centered")
    st.title("SmartRetain: Bank Churn Prediction")
    st.write("This tool predicts whether a bank customer is likely to churn based on their profile.")

    if not MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists():
        st.error("The trained model files are missing. Please run: python train.py")
        return

    model, preprocessor = load_model_assets()
    customer_frame = build_customer_dataframe()

    customer_frame = add_features(customer_frame)

    customer_frame = customer_frame.drop(columns=["CustomerId", "Surname"], errors="ignore")

    if st.button("Predict Churn Risk"):
        processed_features = preprocessor.transform(customer_frame)
        churn_probability = model.predict_proba(processed_features)[0, 1]
        churn_prediction = model.predict(processed_features)[0]

        probability_percent = churn_probability * 100

        if churn_prediction == 1:
            st.error("Prediction: High churn risk")
            st.metric("Churn Probability", f"{probability_percent:.2f}%")
        else:
            st.success("Prediction: Low churn risk")
            st.metric("Churn Probability", f"{probability_percent:.2f}%")

        st.write(
            "Interpretation: customers with a probability close to or above 50% are considered "
            "at higher churn risk and may need retention attention."
        )


if __name__ == "__main__":
    main()

