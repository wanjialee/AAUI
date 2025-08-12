from google.cloud import bigquery, aiplatform
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import xgboost as xgb
import streamlit as st

st.set_page_config(page_title="High Risk Customer", layout="wide")
st.subheader("⚠️ High Risk Customer")

# =====================
# Data Loading
# =====================
@st.cache_data(show_spinner="Loading data...")
def load_data():
    df_policy = pd.read_excel('DM_Policy.xlsx')
    df_underwriting = pd.read_excel('DM_Underwriting.xlsx')
    df_claim = pd.read_excel('DM_Claim.xlsx')
    df_customer = pd.read_excel('DM_Customer.xlsx')
    df_risk = pd.read_excel('DM_Risk.xlsx')
    return df_customer, df_risk, df_underwriting, df_policy, df_claim

df_customer, df_risk, df_underwriting, df_policy, df_claim = load_data()

# =====================
# Data Merging
# =====================
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

# Create target: Will_Claim
claim_counts = df_claim.groupby('Customer_ID').size().reset_index(name='Number_of_Claim')
master_df = master_df.merge(claim_counts, on='Customer_ID', how='left')
master_df['Number_of_Claim'] = master_df['Number_of_Claim'].fillna(0).astype(int)
master_df['Will_Claim'] = master_df['Number_of_Claim'].apply(lambda x: 1 if x > 0 else 0)

# Drop leakage & ID/date columns
target_col = 'Will_Claim'
drop_cols = [
    target_col, 'Number_of_Claim','Churn_Flag','Cancellation_Reason','Is_Fraud_Flag','Claim_Status','Approved_Amount',
    'Claim_Frequency_Customer','Vehicle_Damage_Type','Case_Type','Claim_Type','Repair_Cost_Estimate','Claim_Amount',
    'Time_to_Report','Claims_History_Flag','Load_Date_customer','Load_Date_policy','Load_Date',
    'Underwriter_Decision','Underwriting_Comments', 'Policy_Status'
]
id_cols = [col for col in master_df.columns if 'id' in col.lower()]
datetime_cols_left = master_df.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]']).columns.tolist()
dbdate_cols = [col for col in master_df.columns if str(master_df[col].dtype).lower() in ['dbdate', 'db_datetime']]
all_drop_cols = list(set(drop_cols + id_cols + datetime_cols_left + dbdate_cols))
all_drop_cols_exist = [col for col in all_drop_cols if col in master_df.columns]

# === Step 7: Define X and y ===
X_raw = master_df.drop(columns=all_drop_cols_exist, errors='ignore')
y = master_df[target_col]
X=X_raw

# Encode categoricals
cat_cols = X.select_dtypes(include='object').columns
le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    le_dict[col] = le

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train XGBoost classifier
scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
model = XGBClassifier(random_state=42, scale_pos_weight=scale_pos_weight, eval_metric='logloss')
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n✅ Model Evaluation:")
print(f"Accuracy : {accuracy:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))

# === Step 9: Top 10 features ===
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': importances
}).sort_values(by='importance', ascending=False)
top_10_features = feature_importance_df.head(10)['feature'].tolist()
print("✅ Top 10 important features:", top_10_features)


# === Streamlit UI ===
st.title("🚗 Will Customer Claim (XGBoost)")
st.write(f"✅ Accuracy: {accuracy:.4f}")

user_input = {}
for feat in top_10_features:
    if feat == 'Annual_Mileage':
        # Always use number input for Annual_Mileage
        val = st.number_input(
            f"{feat} (Enter annual mileage in km):",
            min_value=0, max_value=50000,
            step=1000, value=12000
        )
        user_input[feat] = val
    else:
        unique_vals = master_df[feat].dropna().unique()
        if pd.api.types.is_numeric_dtype(master_df[feat]):
            # If numeric and only 0 and 1 → No/Yes
            if sorted(unique_vals.tolist()) == [0, 1]:
                option = st.selectbox(f"{feat}:", ["No (0)", "Yes (1)"])
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

if st.button("Predict"):
    input_df = pd.DataFrame(columns=X.columns)
    input_df.loc[0] = 0
    for feature, value in user_input.items():
        if feature in input_df.columns:
            input_df.at[0, feature] = value
    prediction = model.predict(input_df)[0]
    st.subheader(f"🔮 Predicted: {'Will Claim' if prediction==1 else 'Will Not Claim'}")
