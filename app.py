import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings

warnings.filterwarnings("ignore")

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Credit Card Fraud Detector",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# LOAD FILES
# =========================================================

@st.cache_resource
def load_files():

    model = joblib.load("fraud_model.pkl")

    scaler = joblib.load("scaler.pkl")

    features = joblib.load("features.pkl")

    return model, scaler, features

model, scaler, features = load_files()

# =========================================================
# HEADER
# =========================================================

st.title("💳 AI Credit Card Fraud Detection System")

st.markdown(
    "Real-time fraud detection using Machine Learning."
)

st.markdown("---")

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("Transaction Details")

amount = st.sidebar.number_input(
    "Transaction Amount",
    1.0,
    50000.0,
    1200.0
)

v1 = st.sidebar.slider(
    "V1",
    -30.0,
    30.0,
    -1.23
)

v2 = st.sidebar.slider(
    "V2",
    -30.0,
    30.0,
    2.56
)

# =========================================================
# INPUT DATA
# =========================================================

default_values = {}

for feature in features:

    default_values[feature] = 0

# Assign actual values
if "Amount" in default_values:
    default_values["Amount"] = amount

if "V1" in default_values:
    default_values["V1"] = v1

if "V2" in default_values:
    default_values["V2"] = v2

input_data = pd.DataFrame([default_values])

input_data = input_data[features]

# =========================================================
# PREDICTION
# =========================================================

if st.button("Detect Fraud"):

    try:

        scaled_data = scaler.transform(input_data)

        prediction = model.predict(scaled_data)[0]

        probability = model.predict_proba(
            scaled_data
        )[0][1] * 100

        # =================================================
        # RISK LEVEL
        # =================================================

        if probability >= 80:

            risk = "🔴 CRITICAL"

        elif probability >= 50:

            risk = "🟠 HIGH"

        else:

            risk = "🟢 LOW"

        # =================================================
        # RESULTS
        # =================================================

        st.subheader("Fraud Detection Result")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Fraud Probability",
                f"{probability:.2f}%"
            )

            st.progress(int(probability))

        with col2:

            st.success(f"Risk Level: {risk}")

            if prediction == 1:

                st.error(
                    "Fraudulent transaction detected."
                )

            else:

                st.info(
                    "Legitimate transaction."
                )

    except Exception as e:

        st.error(f"Prediction Error: {e}")

st.markdown("---")

st.markdown(
    "Built with Streamlit • Random Forest • Isolation Forest"
)

