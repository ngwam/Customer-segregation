# Customer-segregation
Identifying different groups

# How to run
run ‘uv sync —link_mode =copy’ in /backend and the same in /frontend. then go back to the base directory (Customer-segregation) and run `docker compose up --build`.
check the docker images to make sure they are created and make sure the docker container is running - it will take a couple of minutes for that all to launch.

locally delete the artifacts , mlruns, ml db directories.

split the terminal and in the other terminal window go to /backend directory and run ‘uv run train_and_export.py’


# How to see results
Streamlit Frontend: http://localhost:8501
FastAPI Backend: http://localhost:8000/docs
MLflow Tracking Dashboard: http://localhost:5000

# Notes
If you make changes to development code, delete all images and containers from the docker desktop.
