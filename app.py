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

/* =====================================================
BACKGROUND
===================================================== */

.main {
    background: linear-gradient(to right, #0f172a, #111827);
    color: white;
}

/* =====================================================
SIDEBAR
===================================================== */

section[data-testid="stSidebar"] {
    background: #0B1120;
    border-right: 1px solid #1f2937;
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

/* =====================================================
TITLE
===================================================== */

.main-title {
    font-size: 42px;
    font-weight: 800;
    color: white;
    margin-bottom: 5px;
}

.subtitle {
    color: #9ca3af;
    font-size: 18px;
    margin-bottom: 30px;
}

/* =====================================================
CARDS
===================================================== */

.card {
    background: #111827;
    padding: 25px;
    border-radius: 18px;
    border: 1px solid #1f2937;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.4);
    text-align: center;
}

.card-title {
    color: #9ca3af;
    font-size: 16px;
}

.card-value {
    color: white;
    font-size: 28px;
    font-weight: bold;
    margin-top: 10px;
}

/* =====================================================
BUTTON
===================================================== */

.stButton>button {
    width: 100%;
    background: linear-gradient(to right, #ff1744, #ff4b2b);
    color: white;
    border-radius: 12px;
    height: 3.3em;
    font-size: 18px;
    font-weight: bold;
    border: none;
    transition: 0.3s ease-in-out;
}

.stButton>button:hover {
    transform: scale(1.02);
    background: linear-gradient(to right, #ff4b2b, #ff1744);
}

/* =====================================================
METRICS
===================================================== */

[data-testid="metric-container"] {
    background: #111827;
    border: 1px solid #1f2937;
    padding: 20px;
    border-radius: 15px;
}

/* =====================================================
FOOTER
===================================================== */

.footer {
    text-align: center;
    color: #9ca3af;
    padding-top: 20px;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
    <div class="main-title">
        💳 AI Credit Card Fraud Detection System
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
        Real-time banking fraud detection using Artificial Intelligence and Machine Learning
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.markdown("## 🧾 Transaction Details")

amount = st.sidebar.number_input(
    "Transaction Amount ($)",
    min_value=1.0,
    max_value=100000.0,
    value=1200.0
)

v1 = st.sidebar.slider(
    "V1 Feature",
    -30.0,
    30.0,
    -1.23
)

v2 = st.sidebar.slider(
    "V2 Feature",
    -30.0,
    30.0,
    2.56
)

st.sidebar.markdown("---")

st.sidebar.info(
    "Adjust transaction parameters to analyze fraud risk."
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
# TOP DASHBOARD
# =========================================================

col1, col2, col3 = st.columns(3)

with col1:

    st.markdown(f"""
    <div class="card">
        <div class="card-title">Transaction Amount</div>
        <div class="card-value">${amount:,.2f}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:

    st.markdown(f"""
    <div class="card">
        <div class="card-title">V1 Score</div>
        <div class="card-value">{v1:.2f}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:

    st.markdown(f"""
    <div class="card">
        <div class="card-title">V2 Score</div>
        <div class="card-value">{v2:.2f}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# =========================================================
# PREDICTION BUTTON
# =========================================================

if st.button("🔍 Analyze Transaction"):

    try:

        prediction = model.predict(input_data)[0]

        probability = model.predict_proba(
            input_data
        )[0][1] * 100

        # =================================================
        # RISK LEVEL
        # =================================================

        if probability >= 80:

            risk = "🔴 CRITICAL RISK"

            recommendation = """
            Immediate fraud investigation required.
            Block the transaction temporarily.
            """

        elif probability >= 50:

            risk = "🟠 HIGH RISK"

            recommendation = """
            Manual verification recommended before approval.
            """

        else:

            risk = "🟢 LOW RISK"

            recommendation = """
            Transaction appears safe and legitimate.
            """

        # =================================================
        # RESULTS
        # =================================================

        st.markdown("## 📊 Fraud Analysis Report")

        result1, result2 = st.columns(2)

        with result1:

            st.metric(
                "Fraud Probability",
                f"{probability:.2f}%"
            )

            st.progress(int(probability))

        with result2:

            st.metric(
                "Risk Category",
                risk
            )

            if prediction == 1:

                st.error(
                    "⚠ Fraudulent Transaction Detected"
                )

            else:

                st.success(
                    "✔ Legitimate Transaction"
                )

        st.markdown("---")

        # =================================================
        # SECURITY RECOMMENDATION
        # =================================================

        st.markdown("## 🛡 Security Recommendation")

        st.info(recommendation)

        # =================================================
        # TRANSACTION SUMMARY
        # =================================================

        st.markdown("## 📋 Transaction Summary")

        summary_df = pd.DataFrame({

            "Feature": [
                "Transaction Amount",
                "V1 Score",
                "V2 Score"
            ],

            "Value": [
                amount,
                v1,
                v2
            ]

        })

        st.dataframe(
            summary_df,
            use_container_width=True
        )

    except Exception as e:

        st.error(f"Prediction Error: {e}")

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown(
    """
    <div class="footer">
        Built with Streamlit • Scikit-Learn • Random Forest • Banking AI
    </div>
    """,
    unsafe_allow_html=True
)
```
