from google.cloud import bigquery, aiplatform
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import streamlit as st

def show_high_risk_customer_page():
    st.subheader("⚠️ High Risk Customer")

    # === Step 1: Setup ===
    PROJECT_ID = 'aaui-464809'
    REGION = 'asia-southeast1'
    aiplatform.init(project=PROJECT_ID, location=REGION)
    client = bigquery.Client(project=PROJECT_ID)

    # === Step 2: Load data from BigQuery ===
    @st.cache_data(show_spinner="Loading data...")
    def load_data():
        df_customer = client.query("SELECT * FROM `aaui-464809.Datamart.DMT_Customer`").to_dataframe()
        df = client.query("SELECT * FROM `aaui-464809.Datamart.DMT_Risk`").to_dataframe()
        df_underwriting = client.query("SELECT * FROM `aaui-464809.Datamart.DMT_Underwriting`").to_dataframe()
        df_policy = client.query("SELECT * FROM `aaui-464809.Datamart.DMT_Policy`").to_dataframe()
        df_claim = client.query("SELECT * FROM `aaui-464809.Datamart.DMT_Claim`").to_dataframe()
        return df_customer, df, df_underwriting, df_policy, df_claim

    df_customer, df, df_underwriting, df_policy, df_claim = load_data()

    # === Step 3: Merge tables ===
    master_df = pd.merge(df, df_customer, on='Customer_ID', how='left')
    master_df = pd.merge(master_df, df_policy, on='Customer_ID', how='left')

    overlap_cols = set(master_df.columns) & set(df_underwriting.columns)
    overlap_cols.discard('Customer_ID')
    master_df = master_df.drop(columns=overlap_cols)
    master_df = pd.merge(master_df, df_underwriting, on='Customer_ID', how='left')

    overlap_cols = set(master_df.columns) & set(df_claim.columns)
    overlap_cols.discard('Customer_ID')
    master_df = master_df.drop(columns=overlap_cols)
    master_df = pd.merge(master_df, df_claim, on='Customer_ID', how='left')

    # === Step 4: Create target ===
    master_df['Risk_Category'] = pd.qcut(master_df['Total_Risk_Score'], q=3, labels=['Low Risk', 'Moderate Risk', 'High Risk'])
    target_col = 'Risk_Category'

    # === Step 5: Drop columns ===
    drop_cols = [
        target_col, 'Churn_Flag', 'Cancellation_Reason', 'Is_Fraud_Flag', 'Claim_Status', 'Approved_Amount',
        'Claim_Frequency_Customer', 'Vehicle_Damage_Type', 'Case_Type', 'Claim_Type', 'Repair_Cost_Estimate', 'Claim_Amount',
        'Time_to_Report', 'Claims_History_Flag', 'Load_Date_customer', 'Load_Date_policy', 'Load_Date', 'Total_Risk_Score'
    ]
    id_cols = [col for col in master_df.columns if 'id' in col.lower()]
    datetime_cols = master_df.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]']).columns.tolist()
    all_drop_cols = list(set(drop_cols + id_cols + datetime_cols))
    all_drop_cols_exist = [col for col in all_drop_cols if col in master_df.columns]

    # === Step 6: Features & encode ===
    X = master_df.drop(columns=all_drop_cols_exist, errors='ignore')
    y = master_df[target_col]

    cat_cols = X.select_dtypes(include='object').columns
    le_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        le_dict[col] = le

    # === Step 7: Train model ===
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # === Step 8: Evaluate ===
    accuracy = accuracy_score(y_test, model.predict(X_test))
    feature_importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values(by='importance', ascending=False)
    top_10_features = feature_importance_df.head(10)['feature'].tolist()

    # === Step 9: Streamlit UI ===
    st.write(f"✅ Model accuracy: **{accuracy:.2f}**")
    st.subheader("Input data for prediction:")
    user_input = {}

    for feat in top_10_features:
        if feat in le_dict:
            options = list(le_dict[feat].classes_)
            val = st.selectbox(f"{feat}:", options)
            user_input[feat] = le_dict[feat].transform([val])[0]
        else:
            val = st.number_input(f"{feat}:", value=0.0)
            user_input[feat] = val

    if st.button("Predict Risk Category"):
        input_df = pd.DataFrame([user_input], columns=X.columns).fillna(0)
        pred = model.predict(input_df)[0]
        st.success(f"🔮 Predicted Risk Category: **{pred}**")
