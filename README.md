# 🍱 Smart Canteen Demand Prediction & Food-Waste Analytics

An end-to-end Machine Learning and Data Analytics project designed to predict canteen food demand and identify potential food-waste risk using historical food-demand data.

The project combines data preprocessing, feature engineering, Machine Learning, model evaluation, demand analytics, model explainability, and an interactive Streamlit dashboard.

It also supports uploading a new month's data and generating predictions using the already trained Machine Learning model.

---

## 📌 Project Overview

Smart Canteen Demand Prediction & Food-Waste Analytics is a Machine Learning based project developed to help canteen management understand food-demand patterns and make better preparation and planning decisions.

The system processes historical food-demand data, performs feature engineering, trains a Random Forest Regression model, predicts food demand, and presents analytical insights through an interactive dashboard.

The project also provides a potential food-waste-risk indicator based on predicted demand.

The complete workflow is:

```text
Historical Data
      ↓
Data Processing
      ↓
Feature Engineering
      ↓
Machine Learning Model
      ↓
Demand Prediction
      ↓
Demand Analytics
      ↓
Potential Food-Waste Risk Analysis
      ↓
Interactive Dashboard
      ↓
New-Month CSV Prediction
```

> The project uses predicted demand as a proxy for potential food-waste risk because the dataset does not contain actual prepared quantity or discarded food quantity.

---

# ✨ Features

## 📊 Food Demand Prediction

- Predict expected food orders using Machine Learning
- Random Forest Regression model
- Uses meal, pricing, promotion, location and time-related features
- Predicts demand for individual records
- Classifies predicted demand into High, Medium and Low levels

---

## ⚠️ Potential Food-Waste Risk Analysis

The project provides a potential food-waste-risk indicator based on predicted demand.

It supports:

- High Risk
- Medium Risk
- Low Risk
- Meal-level risk analysis
- Center-level risk analysis
- Demand-risk distribution

> This is a potential risk proxy and not actual measured food waste percentage.

---

## 📈 Demand Analytics

The project provides:

- Weekly demand analysis
- Category-wise demand
- Cuisine-wise demand
- Top-demand meals
- Top fulfilment centers
- Center-type analysis
- Promotion impact analysis
- Homepage-feature analysis
- Price vs demand analysis

---

## 📊 Interactive Dashboard

The Streamlit dashboard provides:

- Total predicted orders
- Average predicted orders
- Demand distribution
- Potential food-waste-risk distribution
- Weekly demand trend
- Category demand
- Top 10 meals
- Top 10 fulfilment centers
- Center-wise potential food-waste risk
- Interactive filters
- Model performance
- Feature importance
- Data summary

---

## 📥 New-Month Prediction

The dashboard supports uploading a new month's CSV data.

The system automatically performs:

- Input validation
- Duplicate ID checking
- Metadata merging
- Feature engineering
- Prediction using the existing trained model
- Demand classification
- Potential food-waste-risk classification
- Downloadable prediction results

This allows the application to be reused for future monthly data without retraining the model every time.

---

# 🧠 Machine Learning

## Algorithm

The project uses:

**Random Forest Regressor**

The prediction target is:

```text
num_orders
```

---

## Model Configuration

```text
n_estimators = 100
max_depth = 20
min_samples_split = 5
min_samples_leaf = 2
random_state = 42
n_jobs = -1
```

---

## Model Performance

The final trained model achieved:

| Metric | Score |
|---|---:|
| MAE | 69.20 |
| RMSE | 142.55 |
| R² Score | 0.8668 |

### Interpretation

- **MAE = 69.20** means the model's predictions differ from actual orders by approximately 69 orders on average.
- **RMSE = 142.55** gives more weight to larger prediction errors.
- **R² = 0.8668** means the model explains approximately 86.68% of the variance in the target variable.

---

# ⚙️ Data Processing

The raw datasets are merged with meal and fulfilment-center metadata.

Main data files:

```text
train.csv
food_Demand_test.csv
meal_info.csv
fulfilment_center_info.csv
```

The processing pipeline performs:

- Data loading
- Data validation
- Metadata merging
- Missing-value checking
- Duplicate checking
- Feature creation
- Final dataset preparation

---

# 🔧 Feature Engineering

The following features are created:

```text
discount_amount
discount_percentage
price_ratio
month
quarter
week_of_year
total_promotion
```

These features help capture:

- Pricing patterns
- Discounts
- Promotions
- Seasonal patterns
- Weekly demand patterns

---

# 🚨 Target Leakage Prevention

Target leakage was specifically checked during model development.

The feature:

```text
orders_per_area
```

was identified as target leakage because it is calculated using the target variable:

```text
num_orders
```

Therefore, `orders_per_area` is excluded from:

- Model training
- Model evaluation
- New-month prediction

This prevents the model from receiving information derived from the target variable.

---

# 🔍 Model Explainability

Feature importance analysis is performed to understand which features contribute most to demand prediction.

## Top Important Features

| Rank | Feature |
|---:|---|
| 1 | Operating Area |
| 2 | Category - Rice Bowl |
| 3 | Total Promotion |
| 4 | Checkout Price |
| 5 | Base Price |
| 6 | Category - Sandwich |
| 7 | Meal ID |
| 8 | Center ID |
| 9 | City Code |
| 10 | Week of Year |

Feature importance results are generated as CSV files and visualized through a feature-importance chart.

---

# ⚙️ Model Preprocessing

The Machine Learning pipeline uses:

- ColumnTransformer
- OneHotEncoder
- Numeric feature passthrough
- Scikit-learn Pipeline

Categorical features include:

```text
category
cuisine
center_type
```

Categorical values are processed using:

```text
OneHotEncoder(handle_unknown="ignore")
```

This also allows the model to handle previously unseen categorical values during new-month prediction.

---

# 🏗️ System Architecture

```text
                ┌─────────────────────────┐
                │     Historical Data     │
                └────────────┬────────────┘
                             │
                             ↓
                ┌─────────────────────────┐
                │    Data Processing      │
                └────────────┬────────────┘
                             │
                             ↓
                ┌─────────────────────────┐
                │   Feature Engineering   │
                └────────────┬────────────┘
                             │
                             ↓
                ┌─────────────────────────┐
                │ Target Leakage Removal  │
                └────────────┬────────────┘
                             │
                             ↓
                ┌─────────────────────────┐
                │ Random Forest Regressor │
                └────────────┬────────────┘
                             │
                             ↓
                ┌─────────────────────────┐
                │    Demand Prediction    │
                └────────────┬────────────┘
                             │
                  ┌──────────┴──────────┐
                  ↓                     ↓
          Demand Analytics       Waste-Risk Proxy
                  │                     │
                  └──────────┬──────────┘
                             ↓
                ┌─────────────────────────┐
                │ Streamlit Dashboard     │
                └────────────┬────────────┘
                             │
                             ↓
                ┌─────────────────────────┐
                │ New-Month CSV Upload    │
                └────────────┬────────────┘
                             │
                             ↓
                ┌─────────────────────────┐
                │ Existing ML Model       │
                └────────────┬────────────┘
                             │
                             ↓
                ┌─────────────────────────┐
                │ Prediction Results      │
                └─────────────────────────┘
```

---

# 📂 Project Structure

```text
Smart-Canteen-Demand-Prediction-and-Food-Waste-Analytics
│
├── data
│   ├── raw
│   │   ├── train.csv
│   │   ├── food_Demand_test.csv
│   │   ├── meal_info.csv
│   │   └── fulfilment_center_info.csv
│   │
│   ├── processed
│   │   ├── food_demand_merged.csv
│   │   └── food_demand_features.csv
│   │
│   └── new_month
│       └── new_month_data.csv
│
├── models
│   └── smart_canteen_demand_model_final.pkl
│
├── outputs
│   ├── analytics
│   │   ├── meal_demand_analysis.csv
│   │   ├── category_demand_analysis.csv
│   │   ├── center_demand_analysis.csv
│   │   ├── weekly_demand_analysis.csv
│   │   ├── potential_food_waste_risk.csv
│   │   └── center_food_waste_risk.csv
│   │
│   ├── charts
│   │   ├── 01_weekly_demand.png
│   │   ├── 02_category_demand.png
│   │   ├── 03_cuisine_demand.png
│   │   ├── 04_center_type_demand.png
│   │   ├── 05_promotion_demand.png
│   │   ├── 06_homepage_demand.png
│   │   ├── 07_top_10_meals.png
│   │   ├── 08_top_10_centers.png
│   │   ├── 09_price_vs_demand.png
│   │   └── 10_feature_importance.png
│   │
│   ├── model_analysis
│   │   ├── feature_importance.csv
│   │   ├── feature_importance_all.csv
│   │   ├── model_performance.csv
│   │   └── top_10_features.csv
│   │
│   └── new_month
│       └── new_month_predictions.csv
│
├── src
│   ├── data_processing
│   │   ├── 02_merge_data.py
│   │   └── 04_feature_engineering.py
│   │
│   ├── modeling
│   │   └── 05_train_model.py
│   │
│   ├── prediction
│   │   ├── 06_predict_demand.py
│   │   └── 07_new_month_prediction.py
│   │
│   ├── analytics
│   │   ├── 07_prediction_analysis.py
│   │   └── 09_model_analysis.py
│   │
│   └── dashboard
│       └── 08_dashboard.py
│
├── requirements.txt
├── .gitignore
├── .gitattributes
└── README.md
```

> Large datasets, generated prediction files, and other local artifacts may be excluded from Git using `.gitignore`.

---

# 🔄 Project Workflow

## Step 1 — Data Collection

Food-demand datasets are collected from the Kaggle Food Demand Forecasting dataset.

## Step 2 — Data Processing

Raw datasets are cleaned and merged with meal and fulfilment-center information.

## Step 3 — Feature Engineering

Additional pricing, promotion and time-based features are generated.

## Step 4 — Target Leakage Prevention

Features derived from the target variable are removed before model training.

## Step 5 — Model Training

A Random Forest Regression model is trained using the processed dataset.

## Step 6 — Model Evaluation

The model is evaluated using MAE, RMSE and R² Score.

## Step 7 — Demand Prediction

The trained model predicts expected food orders for test data.

## Step 8 — Analytics

Prediction results are analyzed by meal, category, cuisine, center, week, promotion and pricing.

## Step 9 — Model Explainability

Feature importance analysis is performed to understand the major drivers of demand.

## Step 10 — Dashboard

All major insights are presented through an interactive Streamlit dashboard.

---

# 📥 New-Month Prediction Workflow

The application allows users to upload a new month's CSV file.

## Required Columns

```text
id
week
center_id
meal_id
checkout_price
base_price
emailer_for_promotion
homepage_featured
```

## Processing Flow

```text
New Month CSV
      ↓
Input Validation
      ↓
Duplicate ID Check
      ↓
Meal Metadata Merge
      ↓
Center Metadata Merge
      ↓
Feature Engineering
      ↓
Existing Trained Model
      ↓
Predicted Orders
      ↓
Demand Level
      ↓
Potential Waste-Risk Level
      ↓
Download Results
```

The existing trained model is reused for prediction.

Retraining is treated as a separate process and can be performed when sufficient new historical data becomes available.

---

# 📊 Demand Classification

| Predicted Orders | Demand Level |
|---:|---|
| < 150 | Low |
| 150 - 499 | Medium |
| ≥ 500 | High |

---

# ⚠️ Potential Food-Waste Risk Classification

| Predicted Orders | Potential Risk |
|---:|---|
| < 150 | Low Risk |
| 150 - 499 | Medium Risk |
| ≥ 500 | High Risk |

> These thresholds are analytical rules defined for this project and do not represent actual measured food waste.

---

# 📊 Generated Analytics

The project generates:

- Meal demand analysis
- Category demand analysis
- Center demand analysis
- Weekly demand analysis
- Potential food-waste-risk analysis
- Center food-waste-risk analysis
- Feature importance
- Model performance

---

# 📈 Generated Visualizations

The project generates visualizations for:

- Weekly demand
- Category demand
- Cuisine demand
- Center-type demand
- Promotion impact
- Homepage-feature impact
- Top 10 meals
- Top 10 centers
- Price vs demand
- Feature importance

---

# 🛠️ Technology Stack

## Programming Language

- Python

## Data Processing

- Pandas
- NumPy

## Machine Learning

- Scikit-learn
- Random Forest Regressor

## Visualization

- Plotly
- Matplotlib

## Dashboard

- Streamlit

## Version Control

- Git
- GitHub
- Git LFS

## Deployment

- Streamlit Community Cloud

---

# ⚙️ Installation Guide

## 1. Clone the Repository

```bash
git clone https://github.com/Namrata9908/Smart-Canteen-Demand-Prediction-and-Food-Waste-Analytics.git
```

## 2. Navigate to the Project

```bash
cd Smart-Canteen-Demand-Prediction-and-Food-Waste-Analytics
```

## 3. Create Virtual Environment

```bash
python -m venv venv
```

## 4. Activate Virtual Environment

### Windows

```bash
venv\Scriptsctivate
```

## 5. Install Dependencies

```bash
pip install -r requirements.txt
```

## 6. Run the Streamlit Dashboard

```bash
python -m streamlit run src/dashboard/08_dashboard.py
```

The dashboard will open locally at:

```text
http://localhost:8501
```

---

# ▶️ Running the Project Scripts

## Data Processing

```bash
python src/data_processing/02_merge_data.py
```

## Feature Engineering

```bash
python src/data_processing/04_feature_engineering.py
```

## Model Training

```bash
python src/modeling/05_train_model.py
```

## Demand Prediction

```bash
python src/prediction/06_predict_demand.py
```

## New-Month Prediction

```bash
python src/prediction/07_new_month_prediction.py
```

## Prediction Analytics

```bash
python src/analytics/07_prediction_analysis.py
```

## Model Analysis

```bash
python src/analytics/09_model_analysis.py
```

## Dashboard

```bash
python -m streamlit run src/dashboard/08_dashboard.py
```

---

# 💼 Business Value

The project can help canteen management with:

## Demand Planning

Predict expected food demand before preparation.

## Meal Planning

Identify high-demand meals and prepare resources accordingly.

## Inventory Planning

Improve planning of ingredients and other resources.

## Fulfilment Center Planning

Identify centers with consistently high predicted demand.

## Promotion Analysis

Understand how promotions are associated with demand.

## Potential Waste-Risk Monitoring

Identify high-demand situations that may require better operational planning.

---

# ⚠️ Limitations

## Actual Food Waste

The dataset does not contain:

- Prepared quantity
- Sold quantity
- Discarded quantity

Therefore, actual food-waste percentage cannot be calculated from this dataset.

## External Factors

The current model does not explicitly include:

- Weather
- Holidays
- Festivals
- Special events
- Local events
- Sudden demand changes

## Risk Classification

The High / Medium / Low risk levels are analytical thresholds created for this project.

They can be adjusted based on real-world canteen requirements.

---

# 🔮 Future Enhancements

- Integration of actual food-waste data
- Weather data integration
- Holiday and festival features
- Time-series forecasting
- XGBoost / LightGBM comparison
- Hyperparameter tuning
- Automated model retraining
- Model monitoring
- SHAP-based explainability
- Database integration
- Automated monthly prediction pipeline
- Cloud database integration
- Real-time demand monitoring
- Advanced food-waste optimization recommendations

---

# 🎯 Learning Outcomes

Through this project, I learned:

- Data cleaning and preprocessing
- Data integration
- Feature engineering
- Exploratory data analysis
- Regression Machine Learning
- Random Forest implementation
- Model evaluation
- Target leakage detection
- Model explainability
- Data visualization
- Business analytics
- Streamlit dashboard development
- Git and GitHub
- Git LFS
- Cloud deployment
- End-to-end Machine Learning project development

---

# 🏆 Project Highlights

- End-to-end Machine Learning pipeline
- Random Forest demand prediction
- R² Score of **0.8668**
- MAE of **69.20**
- RMSE of **142.55**
- Target leakage prevention
- Feature engineering
- Model explainability
- Demand analytics
- Potential food-waste-risk analytics
- Interactive Streamlit dashboard
- New-month CSV prediction
- Downloadable prediction results
- GitHub version control
- Git LFS for large Machine Learning model
- Live cloud deployment

---

# 🌐 Live Demo

🚀 **Streamlit Dashboard:**

https://smart-canteen-demand-analytics.streamlit.app

The live application provides:

- Demand analytics
- Potential food-waste-risk analysis
- Interactive charts
- Model performance
- Feature importance
- New-month CSV upload
- Demand prediction
- Downloadable prediction results

---

# 💻 GitHub Repository

🔗 **GitHub:**

https://github.com/Namrata9908/Smart-Canteen-Demand-Prediction-and-Food-Waste-Analytics

---

# 👩‍💻 Author

## Namrata

Machine Learning & Data Analytics Portfolio Project

This project demonstrates an end-to-end Machine Learning workflow from raw data processing and feature engineering to model development, analytics, dashboard creation and cloud deployment.

---

# ⭐ Project Summary

Smart Canteen Demand Prediction & Food-Waste Analytics transforms historical food-demand data into useful demand predictions and potential food-waste-risk insights.

The project combines:

```text
Data Processing
       +
Feature Engineering
       +
Machine Learning
       +
Model Evaluation
       +
Model Explainability
       +
Demand Analytics
       +
Visualization
       +
Interactive Dashboard
       +
New-Month Prediction
       +
Cloud Deployment
```

to create a complete end-to-end Machine Learning and Data Analytics solution for smart canteen demand planning.

---

## 🚀 Live Application

https://smart-canteen-demand-analytics.streamlit.app

⭐ If you find this project useful, consider starring the repository.
