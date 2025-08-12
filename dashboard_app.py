import streamlit as st

# === Import individual app pages ===
import fraud_claim
import high_risk_customer
import customer_churn
import claim_forecast
import predict_will_claim

# === Sidebar navigation ===
st.sidebar.title("📌 Insurance ML Dashboard")

page = st.sidebar.selectbox(
    "Choose analysis",
    [
        "🏠 Home",
        "🕵️‍♀️ Fraud Claim Prediction",
        "⚠️ High Risk Customer",
        "📉 Customer Churn",
        "📊 Claim Forecast",
        "❓ Predict Will Claim"
    ]
)

# === Routing logic ===
if page == "🏠 Home":
    st.title("🏠 Welcome to the Insurance ML Dashboard")
    st.markdown("""
    This dashboard integrates multiple machine learning models to provide
    actionable insights for the insurance industry.

    **Available Modules:**
    - 🕵️‍♀️ Fraud Claim Prediction
    - ⚠️ High Risk Customer
    - 📉 Customer Churn
    - 📊 Claim Forecast
    - ❓ Predict Will Claim

    Select an option from the left sidebar to begin.
    """)

elif page == "🕵️‍♀️ Fraud Claim Prediction":
    fraud_claim.show_page()

elif page == "⚠️ High Risk Customer":
    high_risk_customer.show_page()

elif page == "📉 Customer Churn":
    customer_churn.show_page()

elif page == "📊 Claim Forecast":
    claim_forecast.show_page()

elif page == "❓ Predict Will Claim":
    predict_will_claim.show_page()
