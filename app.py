```python
import streamlit as st
import pandas as pd
import joblib
import warnings

warnings.filterwarnings("ignore")

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Fraud Detector",
    page_icon="💳",
    layout="wide"
)

# =========================================================
# LOAD MODEL ONLY
# =========================================================

@st.cache_resource
def load_model():

    model = joblib.load("fraud_model.pkl")

    return model

model = load_model()

# =========================================================
# TITLE
# =========================================================

st.title("💳 AI Credit Card Fraud Detection System")

st.markdown(
    "Real-time fraud prediction using Machine Learning"
)

st.markdown("---")

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("Transaction Details")

amount = st.sidebar.number_input(
    "Transaction Amount",
    1.0,
    100000.0,
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

input_data = pd.DataFrame([{

    'Time': 0,
    'V1': v1,
    'V2': v2,
    'V3': 0,
    'V4': 0,
    'V5': 0,
    'V6': 0,
    'V7': 0,
    'V8': 0,
    'V9': 0,
    'V10': 0,
    'V11': 0,
    'V12': 0,
    'V13': 0,
    'V14': 0,
    'V15': 0,
    'V16': 0,
    'V17': 0,
    'V18': 0,
    'V19': 0,
    'V20': 0,
    'V21': 0,
    'V22': 0,
    'V23': 0,
    'V24': 0,
    'V25': 0,
    'V26': 0,
    'V27': 0,
    'V28': 0,
    'Amount': amount

}])

# =========================================================
# PREDICTION
# =========================================================

if st.button("Detect Fraud"):

    try:

        prediction = model.predict(input_data)[0]

        probability = model.predict_proba(
            input_data
        )[0][1] * 100

        # ==============================================
        # RISK LEVEL
        # ==============================================

        if probability >= 80:

            risk = "🔴 CRITICAL"

        elif probability >= 50:

            risk = "🟠 HIGH"

        else:

            risk = "🟢 LOW"

        # ==============================================
        # RESULTS
        # ==============================================

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
                    "Fraudulent Transaction Detected"
                )

            else:

                st.info(
                    "Legitimate Transaction"
                )

    except Exception as e:

        st.error(f"Prediction Error: {e}")

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown(
    "Built with Streamlit and Random Forest"
)
```
