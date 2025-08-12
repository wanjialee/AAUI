import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import streamlit as st

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📉")
st.subheader("📉 Customer Churn Prediction")

# === Step 2: Load data ===
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

# === Step 6: Prepare target & drop cols ===
target_col = 'Churn_Flag'
y = master_df[target_col]
drop_cols = [
    target_col, 'Cancellation_Reason', 'Claim_Status', 'Case_Type', 'Claim_Amount', 'Approved_Amount',
    'Repair_Cost_Estimate', 'Claim_Type', 'Load_Date', 'Load_Date_customer',
    'Load_Date_policy', 'Load_Date_claim', 'Policy_Status', 'Renewal_Flag',
    'Underwriting_Comments','Underwriter_Decision','Renewal_History','Is_Fraud_Flag','Total_Risk_Score'
]
id_cols = [col for col in master_df.columns if 'id' in col.lower()]
datetime_cols_left = master_df.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]']).columns.tolist()
dbdate_cols = [col for col in master_df.columns if str(master_df[col].dtype).lower() in ['dbdate', 'db_datetime']]
all_drop_cols = list(set(drop_cols + id_cols + datetime_cols_left + dbdate_cols))
all_drop_cols_exist = [col for col in all_drop_cols if col in master_df.columns]

# === Step 7: Define X and y ===
X_raw = master_df.drop(columns=all_drop_cols_exist, errors='ignore')
X = X_raw

# === Step 8: Encode categoricals ===
le_dict = {}
for col in X.columns:
    if pd.api.types.is_object_dtype(X[col]) or pd.api.types.is_categorical_dtype(X[col]):
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        le_dict[col] = le

# === Step 9: Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === Step 10: Train XGBoost ===
scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
model_xgb = xgb.XGBClassifier(random_state=42, scale_pos_weight=scale_pos_weight, eval_metric='logloss')
model_xgb.fit(X_train, y_train)

# === Step 11: Evaluate ===
y_pred = model_xgb.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)
st.write(f"✅ Model accuracy on test set: **{accuracy:.4f}**")

# === Step 12: Top 10 features ===
importances = model_xgb.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': importances
}).sort_values(by='importance', ascending=False)
top_10_features = feature_importance_df.head(10)['feature'].tolist()
st.write("Top 10 important features:", top_10_features)

# === Step 13: Streamlit UI ===
st.subheader("Input values for prediction (top 10 features):")
user_input = {}

for feat in top_10_features:
    if feat == 'Annual_Mileage':
        val = st.number_input(
            f"{feat} (Enter annual mileage in km):",
            min_value=0, max_value=50000,
            step=1000, value=12000
        )
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

if st.button("Predict Churn"):
    input_df = pd.DataFrame(columns=X.columns)
    input_df.loc[0] = 0
    for feat, val in user_input.items():
        if feat in input_df.columns:
            input_df.at[0, feat] = val
    prediction = model_xgb.predict(input_df)[0]
    if prediction == 1:
        st.warning("⚠️ Prediction: Customer is likely to churn")
    else:
        st.success("✅ Prediction: Customer is likely to stay")
