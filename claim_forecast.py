import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Claim Forecast", layout="wide")

def show_claim_forecast_page():
    st.subheader("📊 Claim Forecast")

    @st.cache_data(show_spinner="Loading data...", ttl=600)
    def load_data():
        df_claim = pd.read_excel('DM_Claim.xlsx', engine='openpyxl')
        df_claim['Claim_Date'] = pd.to_datetime(df_claim['Claim_Date'], dayfirst=True, errors='coerce')
        df_claim = df_claim.dropna(subset=['Claim_Date'])
        return df_claim

    df_claim = load_data()
    st.write("✅ Data loaded:", df_claim.shape)

    # Aggregate daily sums
    daily_agg = df_claim.groupby('Claim_Date').agg({
        'Claim_Amount': 'sum',
        'Repair_Cost_Estimate': 'sum',
        'Approved_Amount': 'sum',
        'Time_to_Report': 'sum',
        'Claim_Frequency_Customer': 'sum'
    }).reset_index()

    # Lag & rolling features
    daily_agg['lag_1'] = daily_agg['Claim_Amount'].shift(1)
    daily_agg['lag_7'] = daily_agg['Claim_Amount'].shift(7)
    daily_agg['rolling_7'] = daily_agg['Claim_Amount'].rolling(window=7).mean()
    daily_agg['rolling_30'] = daily_agg['Claim_Amount'].rolling(window=30).mean()
    daily_agg = daily_agg.fillna(daily_agg.median())

    X = daily_agg.drop(columns=['Claim_Date', 'Claim_Amount'])
    y = daily_agg['Claim_Amount']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    @st.cache_resource
    def train_model(X_train, y_train):
        model = XGBRegressor(
            random_state=42,
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6
        )
        model.fit(X_train, y_train)
        return model

    model = train_model(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    st.markdown(f"### ✅ Model Performance")
    st.write(f"**RMSE:** {rmse:.2f}")
    st.write(f"**R² Score:** {r2:.4f}")

    # Build test_df
    test_df = pd.DataFrame({
        'Date': daily_agg.iloc[y_test.index]['Claim_Date'],
        'Actual': y_test.values,
        'Predicted': y_pred
    }).sort_values('Date')

    # Forecast next 30 days
    future_dates = pd.date_range(start=daily_agg['Claim_Date'].max() + pd.Timedelta(days=1), periods=30)
    last_known = daily_agg.iloc[-1].copy()
    future_preds = []

    for date in future_dates:
        row = {
            'Repair_Cost_Estimate': last_known['Repair_Cost_Estimate'],
            'Approved_Amount': last_known['Approved_Amount'],
            'Time_to_Report': last_known['Time_to_Report'],
            'Claim_Frequency_Customer': last_known['Claim_Frequency_Customer'],
            'lag_1': last_known['Claim_Amount'],
            'lag_7': last_known['lag_7'],
            'rolling_7': last_known['rolling_7'],
            'rolling_30': last_known['rolling_30']
        }
        pred = model.predict(pd.DataFrame([row]))[0]
        future_preds.append(pred)

        # Update rolling stats
        last_known['lag_7'] = last_known['lag_1']
        last_known['lag_1'] = pred
        last_known['rolling_7'] = (last_known['rolling_7'] * 6 + pred) / 7
        last_known['rolling_30'] = (last_known['rolling_30'] * 29 + pred) / 30
        last_known['Claim_Amount'] = pred

    # Plot 1
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(daily_agg['Claim_Date'], daily_agg['Claim_Amount'], label='Historical Actual', color='blue')
    ax1.plot(future_dates, future_preds, label='Forecast (next 30 days)', color='orange')
    ax1.set_title('Claim Amount Forecast')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Total Daily Claim Amount')
    ax1.legend()
    st.pyplot(fig1)
    plt.close(fig1)

    # Plot 2
    last_60_test = test_df.sort_values('Date').iloc[-60:]
    test_actual_part = pd.DataFrame({
        'Date': last_60_test['Date'],
        'Value': last_60_test['Actual'],
        'Type': 'Test Actual'
    })

    predicted_dates = list(last_60_test['Date']) + list(future_dates)
    predicted_values = list(last_60_test['Predicted']) + future_preds
    predicted_part = pd.DataFrame({
        'Date': predicted_dates,
        'Value': predicted_values,
        'Type': 'Test Predicted + Forecast'
    })

    plot_df = pd.concat([test_actual_part, predicted_part]).sort_values('Date')

    fig2, ax2 = plt.subplots(figsize=(14, 6))
    for label, part in plot_df.groupby('Type'):
        ax2.plot(part['Date'], part['Value'], label=label)
    ax2.set_title('Last 2 months Test Actual + Test Predicted + Next 30 Days Forecast (connected)')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Daily Claim Amount')
    ax2.legend()
    st.pyplot(fig2)
    plt.close(fig2)


if __name__ == "__main__":
    show_claim_forecast_page()
