import os
import json
import logging
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from config import Config
from azure_search_service import AzureSearchClient
from azure_openai_service import AzureOpenAIClient
from embedding import generate_embeddings, chunk_json, sanitize_data

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Application License Analysis Service")

#CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize clients
search_client = AzureSearchClient()
openai_client = AzureOpenAIClient()

# Models for request validation
class ApplicationAnalysisRequest(BaseModel):
    application_data: dict
    filter_expr: str = None
    top_similar: int = 3

def find_similar_applications(application_data, filter_expr=None, top=3):
    """
    Find similar applications from Azure AI Search.
    """
    json_chunks = chunk_json(application_data)
    embeddings = generate_embeddings(json_chunks)
    
    return search_client.search_similar_applications_with_embeddings(
        embeddings, 
        application_data, 
        filter_expr, 
        top
    )

@app.get("/", status_code=200)
async def root():
    """
    Enhanced root endpoint with comprehensive diagnostics
    """
    return {
        "message": "Application is running",
        "allowed_methods": ["GET", "POST"],
        "endpoints": [
            "/analyze-application (POST)"
        ],
        "status": "Active"
    }

@app.options("/analyze-application")
async def options_analyze_application():
    """
    Explicit OPTIONS handler for CORS preflight requests
    """
    return JSONResponse(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Accept"
        }
    )

@app.post("/analyze-application")
async def analyze_application_endpoint(request: ApplicationAnalysisRequest):
    """
    REST API endpoint for application analysis
    """
    try:
        # Sanitize input data
        sanitized_application_data = sanitize_data(request.application_data)
        
        # Find similar applications
        similar_applications = find_similar_applications(
            sanitized_application_data, 
            request.filter_expr, 
            request.top_similar
        )
        
        # Analyze application
        analysis_result = openai_client.analyze_application(sanitized_application_data, similar_applications)
        
        return {
            "analysis_result": sanitize_data(analysis_result)
        }
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    print(f"Starting server on port {port}...")
    print(f"REST endpoint: http://localhost:{port}/analyze-application")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)