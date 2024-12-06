import pandas as pd
import uvicorn
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from typing import Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load data
def load_data(file_path: str) -> pd.DataFrame:
    try:
        data = pd.read_csv(file_path)
        logger.info(f"Data loaded successfully from {file_path}")
        return data
    except FileNotFoundError:
        logger.error(f"Error: {file_path} not found")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Unexpected error loading data: {str(e)}")
        return pd.DataFrame()

data_file = 'insurance.csv'
df = load_data(data_file)

# FastAPI App
app = FastAPI(
    title="Medical Insurance Costs API",
    description="API for querying medical insurance cost data",
    version="1.0.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Route for index.html
@app.get("/", tags=["Frontend"])
async def serve_frontend():
    try:
        return FileResponse('index.html')
    except FileNotFoundError:
        logger.warning("Frontend file 'index.html' not found")
        raise HTTPException(status_code=404, detail="Frontend not found")

# Query endpoint
@app.get("/query", response_model=dict, tags=["Queries"])
def query_medical_costs(
    min_age: Optional[int] = Query(None, ge=0, description="Minimum age filter"),
    max_age: Optional[int] = Query(None, ge=0, description="Maximum age filter"),
    sex: Optional[str] = Query(None, description="Gender filter", regex="^(male|female)?$"),
    smoker: Optional[str] = Query(None, description="Smoker status filter", regex="^(yes|no)?$"),
    region: Optional[str] = Query(None, description="Region filter")
):
    if min_age is not None and max_age is not None and min_age > max_age:
        raise HTTPException(status_code=400, detail="Min age cannot be greater than max age")

    if df.empty:
        raise HTTPException(status_code=500, detail="Data not available")

    filtered_df = df.copy()

    try:
        if min_age is not None:
            filtered_df = filtered_df[filtered_df['age'] >= min_age]
        if max_age is not None:
            filtered_df = filtered_df[filtered_df['age'] <= max_age]
        if sex:
            filtered_df = filtered_df[filtered_df['sex'].str.lower() == sex.lower()]
        if smoker:
            filtered_df = filtered_df[filtered_df['smoker'].str.lower() == smoker.lower()]
        if region:
            filtered_df = filtered_df[filtered_df['region'].str.lower() == region.lower()]

        if filtered_df.empty:
            return {"message": "No records match the query filters"}

        result = {
            'total_records': len(filtered_df),
            'avg_charges': round(float(filtered_df['charges'].mean()), 2),
            'min_charges': round(float(filtered_df['charges'].min()), 2),
            'max_charges': round(float(filtered_df['charges'].max()), 2),
            'records': filtered_df.head(100).to_dict('records')
        }

        return result

    except KeyError as e:
        logger.error(f"Key error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Missing column in dataset: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error during query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Health check endpoint
@app.get("/health", tags=["System"])
def health_check():
    return {
        "status": "healthy",
        "data_loaded": not df.empty,
        "total_records": len(df) if not df.empty else 0
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
