import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="Customer Personality Analytics API",
    description="Endpoints for predicting total customer spend and retrieving persona segments.",
    version="1.0"
)

# Load Pre-trained Artifacts
try:
    reg_pipeline = joblib.load('/app/artifacts/best_xgb_model.pkl')
    cluster_imputer = joblib.load('/app/artifacts/cluster_imputer.pkl')
    cluster_scaler = joblib.load('/app/artifacts/cluster_scaler.pkl')
    cluster_pca = joblib.load('/app/artifacts/cluster_pca.pkl')
    cluster_model = joblib.load('/app/artifacts/cluster_model.pkl')
    persona_map = joblib.load('/app/artifacts/persona_mapping.pkl')
except Exception as e:
    print(f"Warning: Artifact loading issue: {e}")

CAMPAIGN_RECOMMENDATIONS = {
    'At-Risk Low Spenders': "Offer entry-level discount vouchers (e.g., 15% off first order) and budget-friendly bundle deals.",
    'Campaign Champions': "Send early-access invites to premium wine tasting events and high-tier promotional campaigns.",
    'High-Value Loyalists': "Enroll in VIP concierge service, exclusive rewards programs, and luxury personalized gifts.",
    'Moderate Budget Shoppers': "Target with seasonal cross-category offers and free shipping thresholds to increase basket size."
}

# Request Models
class PredictInput(BaseModel):
    Income: float = Field(..., example=58138.0)
    Recency: int = Field(..., example=58)
    Kidhome: int = Field(..., example=0)
    Teenhome: int = Field(..., example=0)
    NumDealsPurchases: int = Field(..., example=3)
    NumWebPurchases: int = Field(..., example=8)
    NumCatalogPurchases: int = Field(..., example=10)
    NumStorePurchases: int = Field(..., example=4)
    NumWebVisitsMonth: int = Field(..., example=7)
    Age: int = Field(..., example=45)
    Education: str = Field(..., example="Graduation")
    Marital_Status: str = Field(..., example="Single")
    Complain: int = Field(0, example=0)

class SegmentInput(BaseModel):
    Income: float = Field(..., example=58138.0)
    Recency: int = Field(..., example=58)
    MntWines: float = Field(..., example=635.0)
    MntFruits: float = Field(..., example=88.0)
    MntMeatProducts: float = Field(..., example=546.0)
    MntFishProducts: float = Field(..., example=172.0)
    MntSweetProducts: float = Field(..., example=88.0)
    MntGoldProds: float = Field(..., example=88.0)
    AcceptedCmp1: int = Field(0, example=0)
    AcceptedCmp2: int = Field(0, example=0)
    AcceptedCmp3: int = Field(0, example=0)
    AcceptedCmp4: int = Field(0, example=0)
    AcceptedCmp5: int = Field(0, example=0)
    Response: int = Field(0, example=1)

@app.get("/")
def home():
    return {"status": "Customer Analytics API operational"}

@app.post("/predict")
def predict_spend(profile: PredictInput):
    try:
        input_df = pd.DataFrame([profile.dict()])

        education_map = {
            "Basic":0,
            "Graduation":1,
            "2n Cycle":2,
            "Master":2,
            "PhD":3
        }

        marital_map = {
            "Absurd":0,
            "YOLO":0,
            "Alone":1,
            "Single":1,
            "Divorced":2,
            "Widow":2,
            "Together":3,
            "Married":4
        }

        input_df["Education"] = input_df["Education"].map(education_map)
        input_df["Marital_Status"] = input_df["Marital_Status"].map(marital_map)
        input_df["Income"] = np.log1p(input_df["Income"])
        input_df["HalveIncomeIfComplain"] = (
            input_df["Income"] / (input_df["Complain"] + 1)
        )
        predicted_spend = max(float(reg_pipeline.predict(input_df)[0]),0)
        return {
            "predicted_spend_usd": round(predicted_spend, 2),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/segment")
def get_segment(data: SegmentInput):
    try:
        input_dict = data.dict()
        features_order = [
            'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 
            'MntSweetProducts', 'MntGoldProds', 'AcceptedCmp1', 'AcceptedCmp2', 
            'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'Response', 'Recency', 'Income'
        ]

        input_dict["Income"] = np.log1p(input_dict["Income"])
        vector = np.array([[input_dict[f] for f in features_order]])
        
        vector_imp = cluster_imputer.transform(vector)
        vector_scaled = cluster_scaler.transform(vector_imp)
        vector_pca = cluster_pca.transform(vector_scaled)
        
        cluster_id = int(cluster_model.predict(vector_pca)[0])
        persona = persona_map.get(cluster_id, "Unknown Persona")
        campaign = CAMPAIGN_RECOMMENDATIONS.get(persona, "Standard Marketing Outreach")
        
        return {
            "cluster_id": cluster_id,
            "persona": persona,
            "suggested_campaign": campaign,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))