import os
import joblib
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score, root_mean_squared_error,  silhouette_score
from sklearn.ensemble import RandomForestRegressor

# 1. Setup MLflow & Directories
os.makedirs("artifacts", exist_ok=True)
mlflow.set_tracking_uri("http://localhost:5000/")
mlflow.set_experiment("Customer_Analytics_Deployment")

# 2. Load & Preprocess Data
df = pd.read_csv('marketing_campaign.csv', sep='\t')

# Feature Engineering
df["Income"] = np.log1p(df["Income"])

df['Age'] = 2020 - df['Year_Birth']
spend_cols = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
df['Expense_amount'] = df[spend_cols].sum(axis=1)

campaign_cols = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'Response']
df["Campaigns"] = df["AcceptedCmp1"]+df["AcceptedCmp2"]+df["AcceptedCmp3"]+df["AcceptedCmp4"]+df["AcceptedCmp5"]+df["Response"]


edu_order = {"Basic":0, "Graduation": 1, "2n Cycle": 2, "Master": 2, "PhD": 3}
df["Education"] = df["Education"].map(edu_order)

marital_order = {"Absurd":0, "YOLO": 0, "Alone": 1, "Single": 1, "Divorced": 2, "Widow": 2, "Together": 3, "Married": 4}
df["Marital_Status"] = df["Marital_Status"].map(marital_order)

df['HalveIncomeIfComplain'] = df['Income']/(df['Complain']+1)

# --- A. TRAIN REGRESSION MODEL ---
num_features = ['Income', 'Recency', 'Kidhome', 'Teenhome', 'NumDealsPurchases',
                'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 
                'NumWebVisitsMonth', 'Age', 'Education', 'Marital_Status',
                'Complain', 'HalveIncomeIfComplain']

X = df[num_features]
y = df['Expense_amount']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocessor = ColumnTransformer(transformers=[
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), num_features)
])

reg_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42))
])

preprocessor.fit(X_train)

X_train_processed = preprocessor.transform(X_train)
X_test_processed = preprocessor.transform(X_test)

param_grid_ridge = {
    "alpha": [0.1,1,10]
}

param_grid_lasso = {
    "alpha": [0.1,1,10]
}

param_grid_rf = {
    "n_estimators": [100, 200, 300],
    "max_depth": [5, 8, None],
    "min_samples_split": [2, 5],
}

param_grid_xg = {
    "n_estimators": [100, 200, 300],
    "max_depth": [4, 6, 8, None],
    "learning_rate": [0.02, 0.05, 0.1],
}

models = {
    "Ridge": {
        "model": Ridge(),
        "params": param_grid_ridge,
    },
    "Lasso": {
        "model": Lasso(),
        "params": param_grid_lasso,
    },
    "Random Forest": {
        "model": RandomForestRegressor(n_jobs=-1),
        "params": param_grid_rf,
    },
    "XGBoost": {
        "model": XGBRegressor(n_jobs=-1),
        "params": param_grid_xg,
    },
}

list(models.keys())

run_ids = {}

for name, config in models.items():
    model = config["model"]
    param_grid = config["params"]

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )

    with mlflow.start_run(run_name=name) as run:
        mlflow.set_tag("algorithm", name)

        grid.fit(X_train_processed, y_train)

        best_model = grid.best_estimator_

        mlflow.log_params(grid.best_params_)

        y_pred = best_model.predict(X_test_processed)

        metrics = {
            "MAE": mean_absolute_error(y_test, y_pred),
            "RMSE": root_mean_squared_error(y_test, y_pred),
            "R2 Score": r2_score(y_test, y_pred),
            "MAPE": mean_absolute_percentage_error(y_test, y_pred),
        }

        mlflow.log_metrics(metrics)

        if name == "XGBoost":
            mlflow.xgboost.log_model(
                xgb_model=best_model,
                artifact_path="model",
                input_example=X_train_processed[:2],
            )
        else:
            mlflow.sklearn.log_model(
                sk_model=best_model,
                artifact_path="model",
                input_example=X_train_processed[:2],
            )

        run_ids[name] = run.info.run_id

        if name == "XGBoost":
            joblib.dump(best_model, "artifacts/best_xgb_model.pkl")


experiment = mlflow.get_experiment_by_name("Customer_Analytics_Deployment")
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

    mlflow.log_param("pca_components", cluster_pca.n_components)
    mlflow.log_param("kmeans_init", kmeans.n_init)

    mlflow.log_metric("inertia", kmeans.inertia_)
    mlflow.log_metric(
        "silhouette_score",
        silhouette_score(X_cl_pca, kmeans.labels_)
    )
    mlflow.log_metric(
        "explained_variance_pc1",
        cluster_pca.explained_variance_ratio_[0]
    )
    mlflow.log_metric(
        "explained_variance_pc2",
        cluster_pca.explained_variance_ratio_[1]
    )
    mlflow.log_metric(
        "total_explained_variance",
        cluster_pca.explained_variance_ratio_.sum()
    )

    mlflow.sklearn.log_model(kmeans, artifact_path="clustering_model")
    joblib.dump(cluster_imputer, 'artifacts/cluster_imputer.pkl')
    joblib.dump(cluster_scaler, 'artifacts/cluster_scaler.pkl')
    joblib.dump(cluster_pca, 'artifacts/cluster_pca.pkl')
    joblib.dump(kmeans, 'artifacts/cluster_model.pkl')
    joblib.dump(persona_map, 'artifacts/persona_mapping.pkl')

    mlflow.log_artifacts("artifacts")

print("[SUCCESS] Trained models, logged to MLflow, and exported artifacts.")
