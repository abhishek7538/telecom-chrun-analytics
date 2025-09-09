import pandas as pd
#load
df = pd.read_excel(r"C:\Users\ASUS\Desktop\Clg\Abhishek\newcoding\telco_data\Telco_Customer_Churn_final_processed.xlsx")
#quick checks
print(df.shape)
print(df.head())
#Convert TotalCharges to numeric if needed
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
#simple missing imputation example (do this earlier in cleaning step)
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
#usually drop customerID (identifier).
df = df.drop(columns=['customerID'])
#map target
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
#binary columns to map (adjust to your dataset)
binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
#map gender explicitly (Male=1, Female=0)
df['gender'] = df['gender'].map({'Male':1, 'Female':0})
#for rest (Yes/No -> 1/0)
for c in ['Partner','Dependents','PhoneService','PaperlessBilling']:
    df[c] = df[c].map({'Yes':1, 'No':0})    
#Typical multi-category features in Telco
multi_cat = [
    'MultipleLines','InternetService','OnlineSecurity','OnlineBackup',
    'DeviceProtection','TechSupport','StreamingTV','StreamingMovies',
    'Contract','PaymentMethod'
]
#Double-check these columns exist in your DataFrame
[col for col in multi_cat if col not in df.columns]
numeric_cols = ['tenure','MonthlyCharges','TotalCharges']
#ensure numeric dtype
df[numeric_cols] = df[numeric_cols].astype(float)
#Fit preprocessing only on training data:
from sklearn.model_selection import train_test_split
X = df.drop(columns=['Churn'])
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(X_train.shape, X_test.shape)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
#numeric pipeline
numeric_transformer = Pipeline([
    ('scaler', StandardScaler())         #or MinMaxScaler()
])
#categorical pipeline for multi-category columns
categorical_transformer = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
])
#remainder columns (those not listed in numeric_cols or multi_cat) will be passed through.
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, multi_cat)
    ],
    remainder='passthrough'  #pass through binary cols and other columns unchanged
)
#Fit on training data
preprocessor.fit(X_train)
#Transform train & test
X_train_proc = preprocessor.transform(X_train)
X_test_proc  = preprocessor.transform(X_test)
#Build feature names for result DataFrame (sklearn >= 1.0)
#Get OHE feature names
ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
ohe_feature_names = ohe.get_feature_names_out(multi_cat).tolist()
#remainder (passed-through) features order:
remainder_features = [c for c in X_train.columns if c not in numeric_cols + multi_cat]
feature_names = numeric_cols + ohe_feature_names + remainder_features
import pandas as pd
X_train_df = pd.DataFrame(X_train_proc, columns=feature_names, index=X_train.index)
X_test_df  = pd.DataFrame(X_test_proc, columns=feature_names, index=X_test.index)
print("Processed train shape:", X_train_df.shape)
X_train_df.head()
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_df, y_train)
pred = clf.predict(X_test_df)
print(classification_report(y_test, pred))
print("ROC AUC:", roc_auc_score(y_test, clf.predict_proba(X_test_df)[:,1])
import joblib
joblib.dump(preprocessor, "telco_preprocessor.joblib")
#and save the model
joblib.dump(clf, "telco_logreg_model.joblib")
new_df = pd.read_csv("new_customers.csv")   #must have same columns before preprocessing
#perform same binary mappings used earlier:
new_df['gender'] = new_df['gender'].map({'Male':1,'Female':0})
#map other binary cols similarly...
#then:
X_new_proc = preprocessor.transform(new_df)
#convert to DataFrame if needed:
X_new_df = pd.DataFrame(X_new_proc, columns=feature_names)
#get predictions
preds = clf.predict(X_new_df)
X_all = pd.concat([X_train_df, X_test_df], axis=0)
y_all = pd.concat([y_train, y_test], axis=0)
final_df = X_all.copy()
final_df['Churn'] = y_all
final_df.to_excel("Telco_Customer_Churn_Processed_ML_Model.xlsx", index=False)
