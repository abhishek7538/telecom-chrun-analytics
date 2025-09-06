import pandas as pd

# Load dataset
df = pd.read_excel('telco_data/Telco_Customer_Churn.xlsx')

# Convert TotalCharges to numeric and check variance before filling
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
variance_before = df['TotalCharges'].var()
print(f"Variance before filling: {variance_before}")

# Fill missing TotalCharges with 0 (for tenure=0 customers)
df['TotalCharges'].fillna(0, inplace=True)

# Fill categorical missing values with mode
for col in df.select_dtypes(include='object').columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Fill numeric missing values with median
for col in df.select_dtypes(include='number').columns:
    df[col].fillna(df[col].median(), inplace=True)

# Check variance after filling
variance_after = df['TotalCharges'].var()
print(f"Variance after filling: {variance_after}")

print(df.isnull().sum())  # Check again
df.to_excel('telco_data/Telco_Customer_Churn_final_processed.xlsx', index=False)