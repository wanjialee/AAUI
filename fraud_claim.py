import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import xgboost as xgb

st.set_page_config(page_title="Fraud Claim Detection", layout="wide")
st.subheader("🚩 Fraud Claim Detection")

@st.cache_data(show_spinner="Loading data...")
def load_data():
    df_policy = pd.read_excel('DM_Policy.xlsx')
    df_underwriting = pd.read_excel('DM_Underwriting.xlsx')
    df_claim = pd.read_excel('DM_Claim.xlsx')
    df_customer = pd.read_excel('DM_Customer.xlsx')
    df_risk = pd.read_excel('DM_Risk.xlsx')
    return df_customer, df_risk, df_underwriting, df_policy, df_claim

# === Load ===
df_customer, df_risk, df_underwriting, df_policy, df_claim = load_data()

# === Merge ===
master_df = pd.merge(df_risk, df_customer, on='Customer_ID', how='left')
master_df = pd.merge(master_df, df_policy, on='Customer_ID', how='left')

overlap_cols = set(master_df.columns) & set(df_underwriting.columns)
overlap_cols.discard('Customer_ID')
master_df.drop(columns=overlap_cols, inplace=True)
master_df = pd.merge(master_df, df_underwriting, on='Customer_ID', how='left')

overlap_cols = set(master_df.columns) & set(df_claim.columns)
overlap_cols.discard('Customer_ID')
master_df.drop(columns=overlap_cols, inplace=True)
master_df = pd.merge(master_df, df_claim, on='Customer_ID', how='left')

# === Prepare ===
target_col = 'Is_Fraud_Flag'
master_df.dropna(subset=[target_col], inplace=True)
y = master_df[target_col]

drop_cols = [
    target_col, 'Cancellation_Reason', 'Claim_Status', 'Load_Date', 'Load_Date_customer',
    'Load_Date_policy', 'Load_Date_claim', 'Policy_Status', 'Churn_Flag', 'Renewal_Flag',
    'Underwriter_Decision', 'Underwriting_Comments', 'Approved_Amount', 'Loading_Percentage',
    'Case_Type', 'Claim_Amount', 'Repair_Cost_Estimate', 'Claim_Type', 'Vehicle_Damage_Type',
    'Claim_Frequency_Customer'
]
id_cols = [c for c in master_df.columns if 'id' in c.lower()]
datetime_cols = master_df.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]']).columns.tolist()
all_drop_cols = [c for c in set(drop_cols + id_cols + datetime_cols) if c in master_df.columns]

X = master_df.drop(columns=all_drop_cols)

# === Encode ===
le_dict = {}
for col in X.columns:
    if pd.api.types.is_object_dtype(X[col]) or pd.api.types.is_categorical_dtype(X[col]):
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        le_dict[col] = le

# === Train ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
model_xgb = xgb.XGBClassifier(random_state=42, scale_pos_weight=scale_pos_weight, eval_metric='logloss')
model_xgb.fit(X_train, y_train)

y_pred = model_xgb.predict(X_test)
st.write(f"✅ Model accuracy: **{round(accuracy_score(y_test, y_pred), 4)}**")

# === Top features ===
top_10_features = pd.DataFrame({
    'feature': X.columns,
    'importance': model_xgb.feature_importances_
}).sort_values(by='importance', ascending=False).head(10)['feature'].tolist()

st.write("### Input features to predict if a claim is fraud:")
user_input = {}
for feat in top_10_features:
    if feat == 'Annual_Mileage':
        user_input[feat] = st.number_input(f"{feat} (km):", min_value=0, max_value=50000, step=1000, value=12000)
    else:
        if pd.api.types.is_numeric_dtype(master_df[feat]):
            vals = sorted(master_df[feat].dropna().unique())
            if vals == [0, 1]:
                user_input[feat] = 1 if st.selectbox(f"{feat}:", ["No", "Yes"]) == "Yes" else 0
            else:
                user_input[feat] = st.selectbox(f"{feat}:", vals)
        elif feat in le_dict:
            options = list(le_dict[feat].classes_)
            selected = st.selectbox(f"{feat}:", options)
            user_input[feat] = le_dict[feat].transform([selected])[0]
        else:
            user_input[feat] = st.number_input(f"{feat}:", step=1.0)

if st.button("Predict Fraud"):
    input_df = pd.DataFrame(columns=X_train.columns)
    input_df.loc[0] = 0
    for feat, val in user_input.items():
        if feat in input_df.columns:
            input_df.at[0, feat] = val
    pred = model_xgb.predict(input_df)[0]
    st.error("⚠️ Claim is likely FRAUD") if pred == 1 else st.success("✅ Claim is likely legitimate")
