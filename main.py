import os
import sys 
import time
import json
import logging
import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from opencensus.ext.azure.log_exporter import AzureLogHandler
from config import Config

from azure_search_service import AzureSearchClient
from azure_openai_service import AzureOpenAIClient
from embedding import generate_embeddings, chunk_json, clean_data

# Constants for error messages
ERROR_APP_DATA_DICT = "application_data must be a dictionary"
ERROR_FIELDS_LIST = "fields must be a list"
ERROR_FILTER_EXPR_STR = "filter_expr must be a string"
ERROR_TOP_POSITIVE = "top must be a positive integer"

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# First ensure configuration is loaded
try:
    Config.load_config()
except Exception as e:
    logger.critical(f"Failed to load configuration: {str(e)}")
    raise

# Get Application Insights key from Config
instrumentation_key = Config.APPLICATION_INSIGHTS_INSTRUMENTATION_KEY

# Add Azure Application Insights handler if key is available
if instrumentation_key:
    try:
        # The correct way to initialize AzureLogHandler with an instrumentation key
        azure_handler = AzureLogHandler(connection_string=f'InstrumentationKey={instrumentation_key}')
        logger.addHandler(azure_handler)
        logger.info("Application Insights logging enabled")
    except Exception as e:
        logger.error(f"Failed to initialize Application Insights logging: {str(e)}")
else:
    logger.warning("Application Insights instrumentation key not found - telemetry disabled")

# Initialize FastAPI app
app = FastAPI(title="Application License Analysis Service")

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize clients
try:
    search_client = AzureSearchClient()
    openai_client = AzureOpenAIClient()
    logger.info("Azure clients initialized successfully")
except Exception as e:
    logger.critical(f"Failed to initialize Azure clients: {str(e)}")
    raise

# Models for request validation without the validator
class ApplicationAnalysisRequest(BaseModel):
    """
    Request model for application analysis.
    """
    application_data: Dict[str, Any] = Field(..., description="Application data to analyze")
    filter_expr: Optional[str] = Field(None, description="Optional filter expression for search")
    top_similar: int = Field(3, description="Number of similar applications to retrieve", ge=1, le=10)


def extract_field_values(application_data, fields):
    """
    Extract values from specified fields in the application data.
    
    Args:
        application_data (dict): Dictionary containing application fields
        fields (list): List of field names to extract
        
    Returns:
        list: List of string values extracted from specified fields
        
    Raises:
        TypeError: If application_data is not a dictionary or fields is not a list
    """
    if not isinstance(application_data, dict):
        logger.error(ERROR_APP_DATA_DICT)
        raise TypeError(ERROR_APP_DATA_DICT)
        
    if not isinstance(fields, list):
        logger.error(ERROR_FIELDS_LIST)
        raise TypeError(ERROR_FIELDS_LIST)
    
    values = []
    for field in fields:
        value = application_data.get(field)
        if value:  # If the value exists and is not None/empty
            values.append(str(value))  # Convert all values to string
    return values


def find_similar_applications(application_data, filter_expr=None, top=3):
    """
    Find similar applications from Azure AI Search using hybrid search.
    
    Args:
        application_data (dict): Application data to find similar applications for
        filter_expr (str, optional): Optional filter expression for search
        top (int): Number of similar applications to retrieve
        
    Returns:
        list: List of similar applications
        
    Raises:
        TypeError: If parameters are of incorrect type
        ValueError: If parameters have invalid values
    """
    # Validate input parameters
    if not isinstance(application_data, dict):
        logger.error(ERROR_APP_DATA_DICT)
        raise TypeError(ERROR_APP_DATA_DICT)
        
    if filter_expr is not None and not isinstance(filter_expr, str):
        logger.error(ERROR_FILTER_EXPR_STR)
        raise TypeError(ERROR_FILTER_EXPR_STR)
        
    if not isinstance(top, int) or top <= 0:
        logger.error(ERROR_TOP_POSITIVE)
        raise ValueError(ERROR_TOP_POSITIVE)
    
    try:
        json_chunks = chunk_json(application_data)
        embeddings = generate_embeddings(json_chunks)
        
        # Define essential fields for comparison
        essential_fields = [
            "companyName", "companyOrigin", "companyCity", "companyOutput",
            "contributionAmount", "totalCapitalAmount", "shareholderCompanyPartnerName",
            "shareholderNationality", "valueOfEquityOrShares", "percentageOfEquityOrShares",
            "numberOfEquityOrShares", "totalInvestmentValue",
        ]
        
        # Extract values from essential fields
        field_values = extract_field_values(application_data, essential_fields)
        query_text = " ".join(field_values) if field_values else "*"
        
        return search_client.hybrid_search(
            query_text=query_text,
            embeddings=embeddings, 
            filter_expr=filter_expr, 
            top=top,
            app_data=application_data  # Pass application_data for fallback
        )
    except Exception as e:
        logger.error(f"Error finding similar applications: {str(e)}")
        raise


@app.get("/", status_code=status.HTTP_200_OK)
async def root():
    """
    Enhanced root endpoint with comprehensive diagnostics.
    
    Returns:
        dict: Service status information
    """
    try:
        return {
            "message": "Application is running",
            "allowed_methods": ["GET", "POST"],
            "endpoints": [
                "/analyze-application (POST)"
            ],
            "status": "Active"
        }
    except Exception as e:
        logger.error(f"Error in root endpoint: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@app.options("/analyze-application")
async def options_analyze_application():
    """
    Explicit OPTIONS handler for CORS preflight requests.
    
    Returns:
        JSONResponse: Empty response with CORS headers
    """
    try:
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type, Accept"
            }
        )
    except Exception as e:
        logger.error(f"Error in OPTIONS handler: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


def preprocess_application_data(application_data):
    """
    Preprocess application data to handle large text fields before further processing.
    
    Args:
        application_data (dict): Application data to preprocess
        
    Returns:
        dict: Preprocessed application data
        
    Raises:
        TypeError: If application_data is not a dictionary
    """
    if not isinstance(application_data, dict):
        logger.error(ERROR_APP_DATA_DICT)
        raise TypeError(ERROR_APP_DATA_DICT)
    
    try:
        processed_data = clean_data(application_data)
        
        # Define fields that might contain large text values
        large_text_fields = ["companyOutput", "termsAndConditions", "contributionType"]
        
        # Handle large text fields by truncating them if necessary
        for field in large_text_fields:
            if field in processed_data and isinstance(processed_data[field], str):
                text = processed_data[field]
                # If text is too large, truncate it
                if len(text) > 1000:  # Threshold for "large" text
                    processed_data[field] = text[:1000] + "..."
        
        return processed_data
    except Exception as e:
        logger.error(f"Error preprocessing application data: {str(e)}")
        raise


def sanitize_application_data(app_data):
    """
    Helper function to sanitize application data.
    
    Args:
        app_data (dict): Application data to sanitize
        
    Returns:
        dict: Sanitized application data
    """
    sanitized_data = {}
    large_text_fields = ["companyOutput", "termsAndConditions", "contributionType"]
    
    for key, value in app_data.items():
        # For large text fields, truncate immediately
        if key in large_text_fields and isinstance(value, str) and len(value) > 1000:
            sanitized_data[key] = value[:1000] + "..."
        else:
            sanitized_data[key] = value
            
    return sanitized_data


def prepare_search_query(app_data, essential_fields):
    """
    Helper function to prepare search query from application data.
    
    Args:
        app_data (dict): Application data
        essential_fields (list): List of essential fields to include in query
        
    Returns:
        tuple: (field_values, query_text)
    """
    field_values = []
    
    for field in essential_fields:
        value = app_data.get(field)
        if value:
            # Truncate any value to 200 characters max
            field_values.append(str(value)[:200])
    
    query_text = " ".join(field_values) if field_values else "*"
    return field_values, query_text


@app.post("/analyze-application")
async def analyze_application_endpoint(request: ApplicationAnalysisRequest):
    """
    Optimized REST API endpoint for application analysis.
    """
    try:
        # Log received request
        logger.info("Received application analysis request")

        # Preprocess data with minimal operations
        try:
            # Use helper function to sanitize data
            sanitized_application_data = sanitize_application_data(request.application_data)
        except TypeError as e:
            logger.error(f"Invalid application data: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid application data: {str(e)}"
            )
        
        # Find similar applications using hybrid search
        try:
            # Implement minimal chunking for large applications
            app_data_str = json.dumps(sanitized_application_data)
            json_chunks = chunk_json(sanitized_application_data) if len(app_data_str) > 8000 else [sanitized_application_data]
                
            # Generate embeddings for the chunks
            embeddings = generate_embeddings(json_chunks)
            
            # Define essential fields for comparison
            essential_fields = [
                "companyName", "companyOrigin", "companyCity", "companyOutput",
                "contributionAmount", "totalCapitalAmount", "shareholderCompanyPartnerName",
                "shareholderNationality", "valueOfEquityOrShares", "percentageOfEquityOrShares",
                "numberOfEquityOrShares", "totalInvestmentValue",
            ]
            
            # Use helper function to prepare search query
            _, query_text = prepare_search_query(sanitized_application_data, essential_fields)
            
            similar_applications = search_client.hybrid_search(
                query_text=query_text,
                embeddings=embeddings, 
                filter_expr=request.filter_expr, 
                top=request.top_similar,
                app_data=sanitized_application_data
            )
        except Exception as e:
            logger.error(f"Error finding similar applications: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error finding similar applications: {str(e)}"
            )

        # Start timing the model analysis and analyze application
        start_time = time.time()
        try:
            analysis_result = openai_client.analyze_application(sanitized_application_data, similar_applications)
        except Exception as e:
            logger.error(f"Error analyzing application: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error analyzing application: {str(e)}"
            )
            
        model_response_time = round(time.time() - start_time, 3)  # in seconds

        # Extract application ID or a unique identifier if available
        app_id = sanitized_application_data.get("id", "unknown")

        # Log the model response time to Azure Application Insights
        logger.info("Model response time", extra={
            'custom_dimensions': {
                'metric_name': 'model_response_time',
                'response_time_seconds': model_response_time,
                'application_id': app_id
            }
        })
        
        # Log the decision to App Insights if available
        if "Decision" in analysis_result:
            logger.info(
                f"Application decision: {analysis_result['Decision']}",
                extra={'custom_dimensions': {
                    'decision_type': analysis_result['Decision'],
                    'application_id': app_id
                }}
            )
        
        return {
            "analysis_result": analysis_result
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing request: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


if __name__ == "__main__":
    try:
        port = int(os.getenv("PORT", 8000))
        print(f"Starting server on port {port}...")
        print(f"REST endpoint: http://localhost:{port}/analyze-application")
        uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
    except ValueError as e:
        logger.error(f"Invalid PORT environment variable: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error starting server: {str(e)}")
        sys.exit(1)