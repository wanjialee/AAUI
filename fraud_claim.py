import streamlit as st
from google.cloud import bigquery, aiplatform
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import xgboost as xgb

def show_fraud_claim_page():
    st.subheader("🚩 Fraud Claim Detection")

    # === Step 1: Setup ===
    PROJECT_ID = 'aaui-464809'
    REGION = 'asia-southeast1'
    aiplatform.init(project=PROJECT_ID, location=REGION)
    client = bigquery.Client(project=PROJECT_ID)

    @st.cache_data(show_spinner="Loading data...")
    def load_data():
        df_customer = client.query("SELECT * FROM `aaui-464809.Datamart.DMT_Customer`").to_dataframe()
        df = client.query("SELECT * FROM `aaui-464809.Datamart.DMT_Risk`").to_dataframe()
        df_underwriting = client.query("SELECT * FROM `aaui-464809.Datamart.DMT_Underwriting`").to_dataframe()
        df_policy = client.query("SELECT * FROM `aaui-464809.Datamart.DMT_Policy`").to_dataframe()
        df_claim = client.query("SELECT * FROM `aaui-464809.Datamart.DMT_Claim`").to_dataframe()
        return df_customer, df, df_underwriting, df_policy, df_claim

    df_customer, df, df_underwriting, df_policy, df_claim = load_data()

    # === Step 3: Merge ===
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

    # === Step 4: Target & clean ===
    target_col = 'Is_Fraud_Flag'
    master_df = master_df.dropna(subset=[target_col])
    y = master_df[target_col]

    drop_cols = [
        target_col, 'Cancellation_Reason', 'Claim_Status', 'Load_Date', 'Load_Date_customer',
        'Load_Date_policy', 'Load_Date_claim', 'Policy_Status', 'Churn_Flag', 'Renewal_Flag',
        'Underwriter_Decision', 'Underwriting_Comments', 'Approved_Amount', 'Loading_Percentage',
        'Case_Type', 'Claim_Amount', 'Repair_Cost_Estimate', 'Claim_Type', 'Vehicle_Damage_Type',
        'Claim_Frequency_Customer'
    ]
    id_cols = [col for col in master_df.columns if 'id' in col.lower()]
    datetime_cols = master_df.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]']).columns.tolist()
    all_drop_cols = list(set(drop_cols + id_cols + datetime_cols))
    all_drop_cols_exist = [col for col in all_drop_cols if col in master_df.columns]

    X = master_df.drop(columns=all_drop_cols_exist, errors='ignore')

    # === Step 5: Encode ===
    le_dict = {}
    for col in X.columns:
        if pd.api.types.is_object_dtype(X[col]) or pd.api.types.is_categorical_dtype(X[col]):
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            le_dict[col] = le

    # === Step 6: Train ===
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
    model_xgb = xgb.XGBClassifier(random_state=42, scale_pos_weight=scale_pos_weight, eval_metric='logloss')
    model_xgb.fit(X_train, y_train)

    y_pred = model_xgb.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"✅ Model accuracy: {round(accuracy,4)}")

    importances = model_xgb.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': importances
    }).sort_values(by='importance', ascending=False)
    top_10_features = feature_importance_df.head(10)['feature'].tolist()

    # === Step 7: UI ===
    st.write("### Input features to predict if a claim is fraud:")
    user_input = {}
    for feat in top_10_features:
        if feat == 'Annual_Mileage':
            val = st.number_input(f"{feat} (km):", min_value=0, max_value=50000, step=1000, value=12000)
            user_input[feat] = val
        else:
            unique_vals = master_df[feat].dropna().unique()
            if pd.api.types.is_numeric_dtype(master_df[feat]):
                if sorted(unique_vals.tolist()) == [0, 1]:
                    option = st.selectbox(f"{feat}:", ["No", "Yes"])
                    user_input[feat] = 0 if option.startswith("No") else 1
                else:
                    option = st.selectbox(f"{feat}:", sorted(unique_vals))
                    user_input[feat] = option
            elif feat in le_dict:
                options = list(le_dict[feat].classes_)
                val = st.selectbox(f"{feat}:", options)
                user_input[feat] = le_dict[feat].transform([val])[0]
            else:
                val = st.number_input(f"{feat}:", step=1.0)
                user_input[feat] = val

    if st.button("Predict Fraud"):
        input_df = pd.DataFrame(columns=X_train.columns)
        input_df.loc[0] = 0
        for feat, val in user_input.items():
            if feat in input_df.columns:
                input_df.at[0, feat] = val
        prediction = model_xgb.predict(input_df)[0]
        if prediction == 1:
            st.error("⚠️ Prediction: Claim is likely FRAUD")
        else:
            st.success("✅ Prediction: Claim is likely legitimate")
