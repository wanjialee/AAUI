import streamlit as st

# import all pages
import fraud_claim
import high_risk_customer
import customer_churn
import claim_forecast
import predict_will_claim

st.sidebar.title("Insurance ML Dashboard")

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

if page == "🏠 Home":
    st.title("Welcome to the Insurance ML Dashboard")
    st.markdown("""
    Use the sidebar to pick an analysis.
    """)

elif page == "🕵️‍♀️ Fraud Claim Prediction":
    fraud_claim.show_fraud_claim_page()

elif page == "⚠️ High Risk Customer":
    high_risk_customer.show_high_risk_customer_page()

elif page == "📉 Customer Churn":
    customer_churn.show_customer_churn_page()

elif page == "📊 Claim Forecast":
    claim_forecast.show_claim_forecast_page()

elif page == "❓ Predict Will Claim":
    predict_will_claim.show_predict_will_claim_page()
