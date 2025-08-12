import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import streamlit as st

# =====================
# Streamlit UI Settings
# =====================
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

# =====================
# Target & Feature Prep
# =====================
master_df['Risk_Category'] = pd.qcut(
    master_df['Total_Risk_Score'],
    q=3,
    labels=['Low Risk', 'Moderate Risk', 'High Risk']
)
target_col = 'Risk_Category'

drop_cols = [
    target_col, 'Churn_Flag', 'Cancellation_Reason', 'Is_Fraud_Flag',
    'Claim_Status', 'Approved_Amount', 'Claim_Frequency_Customer',
    'Vehicle_Damage_Type', 'Case_Type', 'Claim_Type', 'Repair_Cost_Estimate',
    'Claim_Amount', 'Time_to_Report', 'Claims_History_Flag',
    'Load_Date_customer', 'Load_Date_policy', 'Load_Date', 'Total_Risk_Score'
]
id_cols = [c for c in master_df.columns if 'id' in c.lower()]
datetime_cols = master_df.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]']).columns.tolist()
all_drop_cols = [c for c in set(drop_cols + id_cols + datetime_cols) if c in master_df.columns]

X = master_df.drop(columns=all_drop_cols)
y = master_df[target_col]

# Encode categorical variables
le_dict = {}
for col in X.columns:
    if pd.api.types.is_object_dtype(X[col]) or pd.api.types.is_categorical_dtype(X[col]):
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        le_dict[col] = le

# =====================
# Model Training
# =====================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))
feature_importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values(by='importance', ascending=False)
top_10_features = feature_importance_df.head(10)['feature'].tolist()

# =====================
# Streamlit Input Form
# =====================
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
