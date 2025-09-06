import pandas as pd

df = pd.read_excel('telco_data/Telco_Customer_Churn.xlsx')

# List of features to match (excluding 'TotalCharges' and 'customerID')
features = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod'
]

# Find rows with missing TotalCharges
missing = df[df['TotalCharges'].isnull()]

for idx, row in missing.iterrows():
    # Find other customers with the same feature values
    mask = (df[features] == row[features]).all(axis=1) & df['TotalCharges'].notnull()
    similar_customers = df[mask]
    if not similar_customers.empty:
        # Fill with the mean TotalCharges of similar customers
        mean_charge = similar_customers['TotalCharges'].mean()
        df.at[idx, 'TotalCharges'] = mean_charge

# Save the filled DataFrame
df.to_excel('telco_data/Telco_Customer_Churn_filled.xlsx', index=False)