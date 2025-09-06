# telecom-chrun-analytics

## Project Overview
The project emphasizes the analysis of telecom customer churn based on a blend of Python for data preprocessing, exploratory data analysis (EDA), and machine learning modeling, and Power BI to develop interactive dashboards.
The primary objective is to forecast the customers who will churn, identify the most influential factors leading to churn, and offer actionable recommendations for enhancing customer retention. Analysis includes consideration of customer demographics, contract status, payment schemes, usage of the service, and the amount billed each month.
Some of the key deliverables of the project are:
- Cleaning and preprocessing raw data to prepare it for analysis and modeling
- Investigating patterns and relationships between feature and churn through visualizations including bar charts, pie charts, scatter plots, and heatmaps
- Developing a logistic regression model to predict churn and measure performance with metrics such as accuracy and ROC-AUC
- Developing a Power BI dashboard with KPIs and charts to present an exhaustive view of customer behavior
- Providing retention recommendations based on data-driven insights to enable telecom companies to minimize churn
  
This end-to-end solution shows data analysis and business intelligence capabilities, integrating Python-based analysis with interactive Power BI visualizations to drive decision-making.

## Table of Contents
- [Project Overview](#Project-Overview)
- [Dataset](#Dataset)
- [Data Preprocessing & Cleaning](#Data-Preprocessing-&-Cleaning)
- [Exploratory Data Analysis (EDA)](#Exploratory-Data-Analysis)
  - [Contract vs Churn (Bar Chart)](#Contract-vs-Churn)
  - [Customer Churn by Payment Method (Bar Chart)](#Customer-Churn-by-Payment-Method)
  - [Feature Correlation (Heatmap)](#Heatmap---Feature-Correlation)
- [Modeling](#Modeling)
- [Insights & Observations](#Insights-&-Observations)
- [Power BI Dashboard](#Power-BI-Dashboard)
- [How to Run](#How-to-Run)

  ## Dataset
- Source: Telco Customer Churn dataset
- Rows: 7,000+ customers, Columns: 21
- Key Features: tenure, MonthlyCharges, TotalCharges, Contract, PaymentMethod, Churn

## Data Preprocessing & Cleaning
- Converted TotalCharges to numeric and filled missing values
- Dropped irrelevant columns (e.g., customerID)
- Mapped binary columns (Yes/No → 1/0, Gender: Male=1, Female=0)
- One-hot encoded multi-category columns (Contract, PaymentMethod, InternetService, etc.)
- Scaled numeric columns: tenure, MonthlyCharges, TotalCharges

### Note:
- Some operations of EDA on Dataset are executed and related files are attached 'tel.py', 'tel2.py', tel3.py' which include fillna to fill the missing valures of features and dropna for non-important features
## Exploratory Data Analysis

### Contract vs Churn
- Observes which contract types have higher churn rates.
<img width="600" height="500" alt="Bar Chart -  Contract vs Churn" src="https://github.com/user-attachments/assets/7a618aa8-3d0b-453e-aa81-e66f37dbb373" />

### Customer Churn by Payment Method
- Highlights payment methods linked with higher churn.
<img width="600" height="500" alt="Bar Chart - Customer Churn by Payment Method" src="https://github.com/user-attachments/assets/fb0735cc-b990-479e-a955-5613dbc271f6" />

### Heatmap - Feature Correlation
- Shows correlation between numeric features.
<img width="600" height="500" alt="Heatmap - Feature Correlation" src="https://github.com/user-attachments/assets/e1cc0044-bbf0-4981-afaf-e99ac13f199d" />


## Modeling
- Logistic Regression model for churn prediction
- Train/Test split: 80/20
- Performance:
  - Accuracy: 80%
  - ROC-AUC: 0.85
- Saved preprocessing pipeline (`.joblib`) and model (`.joblib`) for reuse

## Insights & Observations
- Customers on Month-to-Month contracts churn more.
- Fiber Optic internet users have higher churn than DSL.
- Electronic Check payment users churn more frequently.
- Shorter tenure customers tend to leave earlier.
- Higher MonthlyCharges correlate slightly with churn.

## Power BI Dashboard
- Processed dataset: `DA_telec_customer_churn.xlsx`
- Dashboard file: `Final_Report.pbix`
- Includes:
  - Churn rate KPI card
  - Churn by Internet Service, Contract, Payment Method
  - Retention suggestion via Smart Narrative

## How to Run
1. Open `tele_finalML-checkpoint.ipynb` or notebook to explore preprocessing & modeling
2. Load `DA_telec_customer_churn.xlsx` in Power BI
3. Open `Final_Report.pbix` to explore dashboards and KPIs
