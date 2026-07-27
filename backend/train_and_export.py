import os
import joblib
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from xgboost import XGBRegressor

# 1. Setup MLflow & Directories
os.makedirs("artifacts", exist_ok=True)
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Customer_Analytics_Deployment")

# 2. Load & Preprocess Data
df = pd.read_csv('marketing_campaign.csv', sep='\t')

# Feature Engineering
df['Age'] = 2026 - df['Year_Birth']
spend_cols = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
df['Expense_amount'] = df[spend_cols].sum(axis=1)

campaign_cols = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'Response']

# --- A. TRAIN REGRESSION MODEL ---
num_features = ['Income', 'Recency', 'Kidhome', 'Teenhome', 'NumDealsPurchases', 
                'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 
                'NumWebVisitsMonth', 'Age']
cat_features = ['Education', 'Marital_Status']

X = df[num_features + cat_features]
y = df['Expense_amount']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocessor = ColumnTransformer(transformers=[
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), num_features),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
])

reg_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42))
])

with mlflow.start_run(run_name="XGBoost_Regression"):
    reg_pipeline.fit(X_train, y_train)
    mlflow.sklearn.log_model(
        sk_model=reg_pipeline,
        artifact_path="regression_model",
        skops_trusted_types=[
            "numpy.dtype",
            "xgboost.core.Booster",
            "xgboost.sklearn.XGBRegressor"
        ]
    )
    joblib.dump(reg_pipeline, 'artifacts/best_xgb_model.pkl')

# --- B. TRAIN PCA + K-MEANS CLUSTERING ---
cluster_cols = spend_cols + campaign_cols + ['Recency', 'Income']
X_cluster = df[cluster_cols].copy()

cluster_imputer = SimpleImputer(strategy='median')
cluster_scaler = StandardScaler()
cluster_pca = PCA(n_components=2, random_state=42)

X_cl_prep = cluster_scaler.fit_transform(cluster_imputer.fit_transform(X_cluster))
X_cl_pca = cluster_pca.fit_transform(X_cl_prep)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans.fit(X_cl_pca)

persona_map = {
    0: 'At-Risk Low Spenders',
    1: 'Campaign Champions',
    2: 'High-Value Loyalists',
    3: 'Moderate Budget Shoppers'
}

with mlflow.start_run(run_name="PCA_KMeans_Clustering"):
    mlflow.sklearn.log_model(kmeans, artifact_path="clustering_model")
    joblib.dump(cluster_imputer, 'artifacts/cluster_imputer.pkl')
    joblib.dump(cluster_scaler, 'artifacts/cluster_scaler.pkl')
    joblib.dump(cluster_pca, 'artifacts/cluster_pca.pkl')
    joblib.dump(kmeans, 'artifacts/cluster_model.pkl')
    joblib.dump(persona_map, 'artifacts/persona_mapping.pkl')

print("[SUCCESS] Trained models, logged to MLflow, and exported artifacts.")
