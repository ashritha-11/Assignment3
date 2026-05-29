```python
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
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    model = joblib.load("fraud_model.pkl")

    return model

model = load_model()

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

section[data-testid="stSidebar"] {
    background-color: #111827;
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

.stButton>button {
    width: 100%;
    background: linear-gradient(to right, #ff1744, #ff5252);
    color: white;
    border-radius: 12px;
    height: 3.2em;
    font-size: 18px;
    font-weight: bold;
    border: none;
}

.stButton>button:hover {
    background: linear-gradient(to right, #ff5252, #ff1744);
}

h1, h2, h3 {
    color: white;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================

st.title("💳 AI Credit Card Fraud Detection System")

st.markdown(
    "Real-time fraud detection using Machine Learning and Banking AI."
)

st.markdown("---")

# =========================================================
# SIDEBAR INPUTS
# =========================================================

st.sidebar.header("Transaction Details")

amount = st.sidebar.number_input(
    "Transaction Amount",
    min_value=1.0,
    max_value=100000.0,
    value=1200.0
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
# EXACT TRAINING COLUMNS
# =========================================================

input_dict = {

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

}

input_data = pd.DataFrame([input_dict])

# =========================================================
# DASHBOARD METRICS
# =========================================================

c1, c2, c3 = st.columns(3)

with c1:

    st.metric(
        "Transaction Amount",
        f"${amount:,.2f}"
    )

with c2:

    st.metric(
        "V1 Score",
        round(v1, 2)
    )

with c3:

    st.metric(
        "V2 Score",
        round(v2, 2)
    )

st.markdown("---")

# =========================================================
# PREDICTION
# =========================================================

if st.button("Detect Fraud"):

    try:

        prediction = model.predict(input_data)[0]

        probability = model.predict_proba(
            input_data
        )[0][1] * 100

        # =================================================
        # RISK LEVEL
        # =================================================

        if probability >= 80:

            risk = "🔴 CRITICAL"

            recommendation = """
            Immediate fraud investigation required.
            """

        elif probability >= 50:

            risk = "🟠 HIGH"

            recommendation = """
            Transaction should be manually verified.
            """

        else:

            risk = "🟢 LOW"

            recommendation = """
            Transaction appears legitimate.
            """

        # =================================================
        # RESULTS
        # =================================================

        st.subheader("Fraud Detection Result")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                label="Fraud Probability",
                value=f"{probability:.2f}%"
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

        # =================================================
        # RECOMMENDATION
        # =================================================

        st.markdown("## Security Recommendation")

        st.write(recommendation)

    except Exception as e:

        st.error(f"Prediction Error: {e}")

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown(
    "Built with Streamlit • Random Forest • Fraud Detection AI"
)
```
