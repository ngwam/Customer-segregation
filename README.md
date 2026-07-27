# Customer-segregation
Identifying different groups

# How to run
run uv sync in backend
then run train and export, ensure you have pkl files in artifacts
cd to frontend and run uv sync in frontend
then go to home directory and run `docker compose up --build`

# How to see results
Streamlit Frontend: http://localhost:8501
FastAPI Backend: http://localhost:8000/docs
MLflow Tracking Dashboard: http://localhost:5000

# Notes
If you make changes to development code, delete all images and containers from the docker desktop.