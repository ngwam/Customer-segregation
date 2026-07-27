import os
import requests
import streamlit as st

st.set_page_config(page_title="Customer Persona & Spend Predictor", layout="wide")

FASTAPI_URL = os.getenv("FASTAPI_URL", "http://backend:8000")

st.title("📊 Customer Personality Analytics Dashboard")
st.markdown("Input a customer profile to predict total spend, assign a persona segment, and generate a targeted campaign strategy.")

st.sidebar.header("Customer Demographic Input")
income = st.sidebar.number_input("Annual Income ($)", min_value=0.0, value=65000.0, step=1000.0)
age = st.sidebar.slider("Age", 18, 90, 42)
recency = st.sidebar.slider("Days Since Last Purchase (Recency)", 0, 100, 30)
education = st.sidebar.selectbox("Education Level", ["Graduation", "PhD", "Master", "Basic", "2n Cycle"])
marital_status = st.sidebar.selectbox("Marital Status", ["Single", "Together", "Married", "Divorced", "Widow"])

col1, col2 = st.sidebar.columns(2)
kidhome = col1.number_input("Kids at Home", 0, 5, 0)
teenhome = col2.number_input("Teens at Home", 0, 5, 0)

st.sidebar.subheader("Purchasing & Category Spend ($)")
mnt_wines = st.sidebar.number_input("Wine Spend ($)", 0.0, 2000.0, 400.0)
mnt_meat = st.sidebar.number_input("Meat Spend ($)", 0.0, 2000.0, 250.0)
mnt_fruits = st.sidebar.number_input("Fruit Spend ($)", 0.0, 500.0, 30.0)
mnt_fish = st.sidebar.number_input("Fish Spend ($)", 0.0, 500.0, 40.0)
mnt_sweet = st.sidebar.number_input("Sweet Spend ($)", 0.0, 500.0, 25.0)
mnt_gold = st.sidebar.number_input("Gold Spend ($)", 0.0, 500.0, 50.0)

st.sidebar.subheader("Channels & Campaign Activity")
web_purchases = st.sidebar.slider("Web Purchases", 0, 20, 5)
catalog_purchases = st.sidebar.slider("Catalog Purchases", 0, 20, 3)
store_purchases = st.sidebar.slider("Store Purchases", 0, 20, 6)
deals_purchases = st.sidebar.slider("Deals Purchases", 0, 15, 1)
web_visits = st.sidebar.slider("Monthly Web Visits", 0, 30, 4)

cmp_accepted = st.sidebar.multiselect(
    "Accepted Campaigns",
    ["Campaign 1", "Campaign 2", "Campaign 3", "Campaign 4", "Campaign 5", "Last Response"]
)

if st.button("🚀 Analyze Profile & Predict", use_container_width=True):
    predict_payload = {
        "Income": income,
        "Recency": recency,
        "Kidhome": kidhome,
        "Teenhome": teenhome,
        "NumDealsPurchases": deals_purchases,
        "NumWebPurchases": web_purchases,
        "NumCatalogPurchases": catalog_purchases,
        "NumStorePurchases": store_purchases,
        "NumWebVisitsMonth": web_visits,
        "Age": age,
        "Education": education,
        "Marital_Status": marital_status
    }
    
    segment_payload = {
        "Income": income,
        "Recency": recency,
        "MntWines": mnt_wines,
        "MntFruits": mnt_fruits,
        "MntMeatProducts": mnt_meat,
        "MntFishProducts": mnt_fish,
        "MntSweetProducts": mnt_sweet,
        "MntGoldProds": mnt_gold,
        "AcceptedCmp1": 1 if "Campaign 1" in cmp_accepted else 0,
        "AcceptedCmp2": 1 if "Campaign 2" in cmp_accepted else 0,
        "AcceptedCmp3": 1 if "Campaign 3" in cmp_accepted else 0,
        "AcceptedCmp4": 1 if "Campaign 4" in cmp_accepted else 0,
        "AcceptedCmp5": 1 if "Campaign 5" in cmp_accepted else 0,
        "Response": 1 if "Last Response" in cmp_accepted else 0,
    }
    
    try:
        res_pred = requests.post(f"{FASTAPI_URL}/predict", json=predict_payload)
        pred_data = res_pred.json()
        
        res_seg = requests.post(f"{FASTAPI_URL}/segment", json=segment_payload)
        seg_data = res_seg.json()
        
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Predicted Total Spend", f"${pred_data.get('predicted_spend_usd', 0.0):,.2f}")
        with c2:
            st.metric("Assigned Persona", seg_data.get('persona', 'N/A'))
            
        st.markdown("---")
        st.subheader("🎯 Suggested Campaign Recommendation")
        st.info(seg_data.get('suggested_campaign', 'No strategy recommendation available.'))
        
    except Exception as e:
        st.error(f"Error communicating with backend service: {e}")