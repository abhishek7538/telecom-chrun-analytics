import pandas as pd

df = pd.read_excel('telco_data/Telco_Customer_Churn.xlsx')

conditions = {
    'gender': 'Male',
    'SeniorCitizen': 0,
    'Partner': 'No',
    'Dependents': 'Yes',
    'PhoneService': 'Yes',
    'MultipleLines': 'No',
    'InternetService': 'No',
    'OnlineSecurity': 'No phone service',
    'OnlineBackup': 'No phone service',
    'DeviceProtection': 'No phone service',
    'TechSupport': 'No phone service',
    'StreamingTV': 'No phone service',
    'StreamingMovies': 'No phone service',
    'Contract': 'Two year',
    'PaperlessBilling': 'No',
    'PaymentMethod': 'Mailed check'
    
}

filtered = df
for col, val in conditions.items():
    filtered = filtered[filtered[col] == val]

print(filtered['customerID'])