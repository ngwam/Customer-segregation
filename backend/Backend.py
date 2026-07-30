import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, TargetEncoder, FunctionTransformer
from sklearn.base import clone
from category_encoders import CountEncoder
from sklearn.linear_model import Ridge, Lasso
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score, root_mean_squared_error

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

#loading the csv file
df = pd.read_csv('marketing_campaign.csv', sep='\t')
df_copy = df.copy()

#combining columns
df_copy['Children'] = df_copy['Kidhome'] + df_copy['Teenhome']
df_copy = df_copy.drop(columns=['Kidhome', 'Teenhome'])

expense = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
df_copy['Expense_amount'] = df_copy[expense].sum(axis=1)
df_copy = df_copy.drop(columns=expense)

#dropping the columns having same value in all observations
df_copy = df_copy.drop(columns=['Z_Revenue', 'Z_CostContact'])

#converting year of birth to age
df_copy['Age'] = 2014 - df_copy['Year_Birth']
df_copy = df_copy.drop(columns=['Year_Birth'])

# encoding the categories
edu_map = {
            'Basic': 1,
            'Graduation': 2,
            '2n Cycle': 3,
            'Master': 3,
            'PhD': 4
        }

marital_map = {
            'Single': 0, 'Together': 1, 'Married': 1,
            'Divorced': 0, 'Widow': 0, 'Alone': 0,
            'Absurd': 0, 'YOLO': 0
        }

dt_years = pd.to_datetime(df_copy['Dt_Customer'], format='%d-%m-%Y', errors='coerce').dt.year
year_map = {2012: 1, 2013: 2, 2014: 3}

df_copy['Education'] = df_copy['Education'].map(edu_map)
df_copy['Marital_Status'] = df_copy['Marital_Status'].map(marital_map)
df_copy['Dt_Customer'] = dt_years.map(year_map)

# Age cannot be more than 100
df_copy.loc[(df_copy['Age'] > 100), 'Age'] = np.nan

# Splitting the dataset
X = df_copy.drop(columns=['ID', 'Expense_amount'])
y = df_copy['Expense_amount']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#Preprocessing pipeline
pass_col = [
    'Response', 'Complain', 'AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'Children',
    'Education', 'Marital_Status', 'Dt_Customer'
]

num = [
    'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth', 'Recency', 'Age'
]

outliers_col = ['Income']

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

outliers_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('log_transform', FunctionTransformer(np.log1p, validate=True)),
    ('scaler', StandardScaler())
])

pass_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent'))
])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_pipeline, num),
        ('Log_tranform', outliers_pipeline, outliers_col),
        ('passthrough', pass_pipeline, pass_col)
    ],
    remainder='drop'
)

preprocessor.fit(X_train)

X_train_processed = preprocessor.transform(X_train)
X_test_processed = preprocessor.transform(X_test)

#Implementing MLFlow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Regression_comparison")

print("Tracking URI:", mlflow.get_tracking_uri())
print("Active experiment:", mlflow.get_experiment_by_name("Regression_comparison").name)

models = {'Ridge Regressor': (Ridge, {'alpha':1.0}),
          'Lasso Regressor': (Lasso, {'alpha':1.0, 'random_state':42}),
          'XGBoost Regressor': (XGBRegressor, {'n_estimators':100, 'learning_rate':0.05, 'max_depth':4, 'random_state':42})}
list(models.keys())

run_ids = {}

for name, (ModelClass, params) in models.items():
    with mlflow.start_run(run_name=name) as run:
        mlflow.set_tag("algorithm", name)
        mlflow.log_params(params)

        model = ModelClass(**params)
        model.fit(X_train_processed, y_train)

        y_pred = model.predict(X_test_processed)
        proba = model.predict_proba(X_test_processed)[:, 1] if hasattr(model, "predict_proba") else None
        
        metrics = {
            'MAE': mean_absolute_error(y_test, y_pred),
            'RMSE': root_mean_squared_error(y_test, y_pred),
            'R2 Score': r2_score(y_test, y_pred),
            'MAPE': mean_absolute_percentage_error(y_test, y_pred)
            }
        
        mlflow.log_metrics(metrics)
        trusted_types = ["xgboost.core.Booster", "xgboost.sklearn.XGBRegressor"]
        
        mlflow.sklearn.log_model(
            sk_model=model, 
            artifact_path="model", 
            input_example=X_train_processed[:2],
            skops_trusted_types=trusted_types
        )
        run_ids[name] = run.info.run_id

experiment = mlflow.get_experiment_by_name("Regression_comparison")
runs_df = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

metric_cols = [c for c in runs_df.columns if c.startswith("metrics.")]
comparison = runs_df[["run_id", "tags.algorithm"] + metric_cols].sort_values(
    "metrics.MAE", ascending=True
).reset_index(drop=True)
comparison.columns = [c.replace("metrics.", "") for c in comparison.columns]

best_row = comparison.iloc[0]
best_run_id = best_row["run_id"]
best_algorithm = best_row["tags.algorithm"]

print(f"Best model: {best_algorithm}  (run_id={best_run_id},R2={best_row['R2 Score']:.3f})")

model_uri = f"runs:/{best_run_id}/model"
loaded_model = mlflow.sklearn.load_model(model_uri)

loaded_preds = loaded_model.predict(X_test_processed)
print("\nReloaded model matches its original test R2 Score:",
      np.isclose(r2_score(y_test, loaded_preds), best_row["R2 Score"]))

#Model registration
registered = mlflow.register_model(model_uri=model_uri, name="best_regression_model")
print(f"Registered '{registered.name}' as version {registered.version}")

# load it back by registry name + version, instead of by run id
registry_model = mlflow.sklearn.load_model(f"models:/{registered.name}/{registered.version}")
print("Loaded from registry OK:", r2_score(y_test, registry_model.predict(X_test_processed)))

#Clustering
spend_cols = ['MntWines', 'MntFruits', 'MntMeatProducts', 
              'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
campaign_cols = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 
                 'AcceptedCmp4', 'AcceptedCmp5', 'Response']
clustering_features = spend_cols + campaign_cols + ['Recency', 'Income']

X_raw = df[clustering_features].copy()

imputer = SimpleImputer(strategy='median')
scaler = StandardScaler()

X_imp = imputer.fit_transform(X_raw)
X_scaled = scaler.fit_transform(X_imp)

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
pca_full = PCA(n_components=len(clustering_features), random_state=42)
pca_full.fit(X_scaled)
exp_var = pca_full.explained_variance_ratio_
cum_exp_var = np.cumsum(exp_var)

for k in [3, 4, 5]:
    print(f"\n--- Clusters k={k} ---")
    
    # 1. K-Means
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km_labels = km.fit_predict(X_scaled)
    km_sil = silhouette_score(X_scaled, km_labels)
    
    # 2. Agglomerative
    agg = AgglomerativeClustering(n_clusters=k)
    agg_labels = agg.fit_predict(X_scaled)
    agg_sil = silhouette_score(X_scaled, agg_labels)
    
    # 3. Gaussian Mixture
    gmm = GaussianMixture(n_components=k, random_state=42)
    gmm_labels = gmm.fit_predict(X_scaled)
    gmm_sil = silhouette_score(X_scaled, gmm_labels)
    
    print(f"K-Means Silhouette Score:         {km_sil:.4f}")
    print(f"Agglomerative Silhouette Score:   {agg_sil:.4f}")
    print(f"Gaussian Mixture Silhouette Score:{gmm_sil:.4f}")

km = KMeans(n_clusters=4, random_state=42, n_init=10)
km_labels = km.fit_predict(X_scaled)
km_sil = silhouette_score(X_scaled, km_labels)
df['Cluster'] = km_labels

persona_mapping = {
    0: 'At-Risk Low Spenders',
    1: 'Unengaged High Spenders',
    2: 'High-Value Loyalists',
    3: 'Campaign Champions'
}

df['Persona'] = df['Cluster'].map(persona_mapping)

df['Total_Spend'] = df[spend_cols].sum(axis=1)
persona_summary = df.groupby('Persona').agg({
    'Income': 'mean',
    'Total_Spend': 'mean',
    'MntWines': 'mean',
    'Recency': 'mean',
    'Response': 'mean',
    'ID': 'count'
}).rename(columns={'ID': 'Customer_Count'}).round(2)

print("\n=== Persona Profiling Summary ===")
print(persona_summary)