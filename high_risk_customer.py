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

# === Step 3: Merge tables carefully ===
master_df = pd.merge(df, df_customer, on='Customer_ID', how='left', suffixes=('', '_customer'))
master_df = pd.merge(master_df, df_policy, on='Customer_ID', how='left', suffixes=('', '_policy'))

overlap_cols = set(master_df.columns) & set(df_underwriting.columns)
overlap_cols.discard('Customer_ID')
master_df = master_df.drop(columns=overlap_cols)
master_df = pd.merge(master_df, df_underwriting, on='Customer_ID', how='left')

overlap_cols = set(master_df.columns) & set(df_claim.columns)
overlap_cols.discard('Customer_ID')
master_df = master_df.drop(columns=overlap_cols)
master_df = pd.merge(master_df, df_claim, on='Customer_ID', how='left')

# === Step 5: Create target column ===
master_df['Risk_Category'] = pd.qcut(master_df['Total_Risk_Score'], q=3, labels=['Low Risk', 'Moderate Risk', 'High Risk'])
target_col = 'Risk_Category'

# === Step 6: Identify columns to drop ===
drop_cols = [
    target_col, 'Churn_Flag', 'Cancellation_Reason', 'Is_Fraud_Flag', 'Claim_Status', 'Approved_Amount',
    'Claim_Frequency_Customer', 'Vehicle_Damage_Type', 'Case_Type', 'Claim_Type', 'Repair_Cost_Estimate','Claim_Amount',
    'Time_to_Report', 'Claims_History_Flag','Load_Date_customer', 'Load_Date_policy', 'Load_Date', 'Total_Risk_Score'
]
id_cols = [col for col in master_df.columns if 'id' in col.lower()]
# Remove raw datetime columns and dbdate/db_datetime columns
datetime_cols_left = master_df.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]']).columns.tolist()
dbdate_cols = [col for col in master_df.columns if str(master_df[col].dtype).lower() in ['dbdate', 'db_datetime']]
all_drop_cols = list(set(drop_cols + id_cols + datetime_cols_left + dbdate_cols))
all_drop_cols_exist = [col for col in all_drop_cols if col in master_df.columns]

# === Step 7: Define X and y ===
X_raw = master_df.drop(columns=all_drop_cols_exist, errors='ignore')
y = master_df[target_col]
X=X_raw

# === Step 8: Encode categorical features ===
cat_cols = X.select_dtypes(include='object').columns
le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    le_dict[col] = le

# === Step 9: Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(X_raw, y, test_size=0.2, random_state=42)

# === Step 10: Train model ===
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# === Step 11: Evaluate ===
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

# === Step 12: Feature importance ===
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': importances
}).sort_values(by='importance', ascending=False)
top_10_features = feature_importance_df.head(10)['feature'].tolist()
print("✅ Top 10 important features:", top_10_features)

# === Step 13: Streamlit UI ===
def show_ui():

st.title("🚗 High Risk Customer Prediction")
st.write(f"✅ Model accuracy on test data: **{accuracy:.2f}**")

st.subheader("Input customer data for prediction:")
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

if st.button("Predict Risk Category"):
    input_df = pd.DataFrame(columns=X.columns)
    input_df.loc[0] = 0
    for feat, val in user_input.items():
        if feat in input_df.columns:
            input_df.at[0, feat] = val
    pred = model.predict(input_df)[0]
    st.subheader(f"🔮 Predicted Risk Category: **{pred}**")



