"""
Main application for Azure data comparison integrating Azure AI Search, Azure OpenAI,
and providing REST interface for application license analysis.
"""

import os
import json
import logging
import requests
import math
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from prompts import get_investment_analysis_prompt

# Additional imports for JSON chunking and embedding
import tiktoken
from openai import AzureOpenAI
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Application License Analysis Service")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom JSON encoder to handle NaN and float values
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        # Handle numpy arrays
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        
        # Handle numpy numeric types
        if isinstance(obj, (np.float32, np.float64)):
            # Convert NaN or Inf to None
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        
        # Handle regular float NaN
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None
        
        return super().default(obj)

# Utility function to sanitize data for JSON serialization
def sanitize_data(data):
    """
    Recursively sanitize data to make it JSON-compliant.
    
    Args:
        data: Input data to sanitize
    
    Returns:
        Sanitized data
    """
    if isinstance(data, dict):
        return {k: sanitize_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_data(v) for v in data]
    elif isinstance(data, (np.ndarray, np.generic)):
        # Convert numpy types to Python types
        if np.isnan(data) or np.isinf(data):
            return None
        return data.item() if hasattr(data, 'item') else data.tolist()
    elif isinstance(data, float):
        # Handle float NaN and Inf
        if math.isnan(data) or math.isinf(data):
            return None
    return data

# Models for request validation
class ApplicationAnalysisRequest(BaseModel):
    application_data: dict
    filter_expr: str = None
    top_similar: int = 3

def chunk_json(json_data, max_tokens=500):
    """
    Chunk JSON data into smaller pieces while preserving structure.
    
    Args:
        json_data (dict): Input JSON data
        max_tokens (int): Maximum number of tokens per chunk
    
    Returns:
        list: List of JSON chunks
    """
    # Use tiktoken for token counting
    encoding = tiktoken.get_encoding("cl100k_base")
    
    # Serialize the entire JSON
    serialized_data = json.dumps(json_data, cls=CustomJSONEncoder)
    
    # Tokenize the serialized data
    tokens = encoding.encode(serialized_data)
    
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for token in tokens:
        current_chunk.append(token)
        current_tokens += 1
        
        # When chunk reaches max tokens, create a chunk
        if current_tokens >= max_tokens:
            decoded_chunk = encoding.decode(current_chunk)
            try:
                chunk = json.loads(decoded_chunk, parse_float=float)
                chunks.append(chunk)
                
                # Print chunk information
                print(f"Chunk {len(chunks)}:")
                print(json.dumps(chunk, indent=2, cls=CustomJSONEncoder))
                print(f"Tokens in chunk: {current_tokens}\n")
            except json.JSONDecodeError:
                # If decoding fails, try to keep it as a partial JSON
                logger.warning("Partial chunk might lose JSON structure")
                chunks.append(json.loads(decoded_chunk + "}", parse_float=float))
            
            # Reset for next chunk
            current_chunk = []
            current_tokens = 0
    
    # Add remaining tokens
    if current_chunk:
        decoded_chunk = encoding.decode(current_chunk)
        try:
            chunk = json.loads(decoded_chunk, parse_float=float)
            chunks.append(chunk)
            
            # Print chunk information
            print(f"Final Chunk:")
            print(json.dumps(chunk, indent=2, cls=CustomJSONEncoder))
            print(f"Tokens in final chunk: {len(current_chunk)}\n")
        except json.JSONDecodeError:
            logger.warning("Last chunk might be incomplete")
            chunks.append(json.loads(decoded_chunk + "}", parse_float=float))
    
    print(f"Total number of chunks: {len(chunks)}")
    return chunks

def generate_embeddings(chunks):
    """
    Generate embeddings for JSON chunks using Azure OpenAI.
    
    Args:
        chunks (list): List of JSON chunks
    
    Returns:
        list: List of embeddings
    """
    # Initialize Azure OpenAI client for embedding
    embedding_client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2025-01-01-preview"
    )
    
    embeddings = []
    for i, chunk in enumerate(chunks, 1):
        try:
            # Convert chunk back to string for embedding
            chunk_str = json.dumps(chunk, cls=CustomJSONEncoder)
            
            response = embedding_client.embeddings.create(
                input=chunk_str,
                model=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
            )
            
            # Extract the first embedding (assuming single chunk)
            embedding = response.data[0].embedding
            
            # Sanitize embedding
            sanitized_embedding = [
                float(val) if not (math.isnan(val) or math.isinf(val)) else 0.0 
                for val in embedding
            ]
            
            embeddings.append(sanitized_embedding)
            
            # Print embedding details
            print(f"Embedding for Chunk {i}:")
            print(f"Embedding dimensions: {len(sanitized_embedding)}")
            print(f"First 5 embedding values: {sanitized_embedding[:5]}")
            print(f"Embedding norm (magnitude): {np.linalg.norm(sanitized_embedding):.4f}\n")
        except Exception as e:
            logger.error(f"Embedding generation error for chunk {i}: {str(e)}")
            embeddings.append(None)
    
    print(f"Total number of embeddings: {len(embeddings)}")
    return embeddings

class AzureSearchClient:
    def __init__(self):
        self.endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
        self.index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")
        self.api_key = os.getenv("AZURE_SEARCH_API_KEY")
        self.api_version = "2020-06-30"
        
        # Comprehensive configuration logging
        logger.info("Azure Search Configuration:")
        logger.info(f"Endpoint: {self.endpoint}")
        logger.info(f"Index Name: {self.index_name}")
        logger.info(f"API Key Provided: {'Yes' if bool(self.api_key) else 'No'}")
        
        if not all([self.endpoint, self.index_name, self.api_key]):
            raise ValueError("Azure Search credentials are not properly configured in .env file")
            
    def _get_headers(self):
        """Return the headers needed for Azure Search API requests."""
        return {
            "Content-Type": "application/json",
            "api-key": self.api_key
        }
        
    def search_similar_applications(self, app_data, filter_expr=None, top=3):
        """
        Search for similar applications in the Azure AI Search index.
        
        Args:
            app_data (dict): The application data to compare against
            filter_expr (str, optional): OData filter expression
            top (int, optional): Number of results to return
            
        Returns:
            dict: The search results
        """
        url = f"{self.endpoint}/indexes/{self.index_name}/docs/search?api-version={self.api_version}"
        
        # Ensure search terms are properly formatted as strings
        search_terms = []
        if app_data.get("licenseType"):
            license_type = app_data['licenseType']
            # Handle if licenseType is a list/array
            if isinstance(license_type, list):
                search_terms.extend([str(lt) for lt in license_type])
            else:
                search_terms.append(str(license_type))
                
        if app_data.get("appType"):
            app_type = app_data['appType']
            # Handle if appType is a list/array
            if isinstance(app_type, list):
                search_terms.extend([str(at) for at in app_type])
            else:
                search_terms.append(str(app_type))
        
        # Create a fallback search if no specific terms
        search_query = " OR ".join(search_terms) if search_terms else "*"
        
        payload = {
            "search": search_query,
            "top": top,
            "queryType": "simple",
            "searchFields": "content",
            "select": "content"
        }
        
        if filter_expr:
            payload["filter"] = filter_expr
        
        # Log the payload for debugging
        logger.info(f"Search payload: {json.dumps(payload)}")
        
        try:
            response = requests.post(url, headers=self._get_headers(), json=payload)
            
            if response.status_code == 200:
                result = response.json()
                # Process the results to extract application data from content field
                if "value" in result:
                    processed_results = []
                    for item in result["value"]:
                        if "content" in item:
                            try:
                                # Parse the JSON string from the content field
                                app_data = json.loads(item["content"])
                                
                                # Process the details_form field if it's a string
                                if isinstance(app_data.get("details_form"), str):
                                    try:
                                        app_data["details_form"] = json.loads(app_data["details_form"])
                                    except json.JSONDecodeError:
                                        logger.warning(f"Failed to parse details_form for application {app_data.get('uuid')}")
                                
                                processed_results.append(app_data)
                            except json.JSONDecodeError:
                                logger.warning(f"Failed to parse content field in search result")
                    
                    return processed_results
                return []
            else:
                logger.error(f"Search request failed: {response.status_code} - {response.text}")
                return []
                
        except Exception as e:
            logger.error(f"Exception during search: {str(e)}")
            return []
    
    def search_similar_applications_with_embeddings(self, embeddings, app_data, filter_expr=None, top=3):
        """
        Enhanced search method using embeddings for similarity search.
        
        Args:
            embeddings (list): List of embeddings from the input data
            app_data (dict): Original application data
            filter_expr (str, optional): OData filter expression
            top (int, optional): Number of results to return
            
        Returns:
            dict: The search results
        """
        # Combine embedding search with traditional search logic
        similar_apps = self.search_similar_applications(app_data, filter_expr, top)
        
        # You could add more sophisticated embedding-based filtering here
        # For now, we'll use the existing search results
        return similar_apps

class AzureOpenAIClient:
    def __init__(self):
        # Detailed environment variable logging
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip('/')
        self.api_version = os.getenv("AZURE_OPENAI_VERSION","2025-01-01-preview")
        self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        
        # Comprehensive logging of configuration
        logger.info("Azure OpenAI Configuration:")
        logger.info(f"Endpoint: {self.endpoint}")
        logger.info(f"Deployment Name: {self.deployment_name}")
        logger.info(f"API Version: {self.api_version}")
        logger.info(f"API Key Provided: {'Yes' if bool(self.api_key) else 'No'}")
        
        # Validate configuration
        self._validate_configuration()
    
    def _validate_configuration(self):
        """
        Comprehensive configuration validation
        """
        errors = []
        
        if not self.api_key:
            errors.append("AZURE_OPENAI_API_KEY is missing")
        
        if not self.endpoint:
            errors.append("AZURE_OPENAI_ENDPOINT is missing")
        elif not self.endpoint.startswith("https://"):
            errors.append("AZURE_OPENAI_ENDPOINT must start with https://")
        
        if not self.deployment_name:
            errors.append("AZURE_OPENAI_DEPLOYMENT_NAME is missing")
        
        if not self.api_version:
            errors.append("AZURE_OPENAI_VERSION is missing")
        
        if errors:
            error_message = "OpenAI Configuration Errors:\n" + "\n".join(errors)
            logger.error(error_message)
            raise ValueError(error_message)
    
    def _get_headers(self):
        """
        Comprehensive headers with additional diagnostics
        """
        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key
        }
        return headers
    
    def get_completion(self, prompt, max_tokens=2000, temperature=0.0):
        """
        Enhanced completion method with multiple diagnostic checks
        """
        try:
            # Construct full URL with detailed logging
            full_url = f"{self.endpoint}/openai/deployments/{self.deployment_name}/chat/completions?api-version={self.api_version}"
            logger.info(f"Full OpenAI Request URL: {full_url}")
            
            # Detailed payload construction
            payload = {
                "messages": [
                    {"role": "system", "content": "You are an AI assistant analyzing investment license applications."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "n": 1
            }
            
            # Log payload details (be cautious with sensitive data)
            logger.info(f"Payload Structure: {json.dumps(list(payload.keys()), indent=2)}")
            
            # Make request with timeout and additional error handling
            try:
                response = requests.post(
                    full_url, 
                    headers=self._get_headers(), 
                    json=payload,
                    timeout=30  # Add timeout to prevent hanging
                )
            except requests.exceptions.RequestException as req_ex:
                logger.error(f"Request Exception: {str(req_ex)}")
                raise
            
            # Comprehensive response logging
            logger.info(f"Response Status Code: {response.status_code}")
            logger.info(f"Response Headers: {dict(response.headers)}")
            
            # Detailed error handling
            if response.status_code != 200:
                logger.error(f"Full Error Response: {response.text}")
                
                # Try to parse error details
                try:
                    error_details = response.json().get('error', {})
                    logger.error(f"Parsed Error Details: {error_details}")
                except ValueError:
                    logger.error("Could not parse error response as JSON")
                
                raise Exception(f"OpenAI request failed: {response.status_code} - {response.text}")
            
            # Process successful response
            response_json = response.json()
            return response_json['choices'][0]['message']['content'].strip()
        
        except Exception as e:
            logger.error(f"Comprehensive Error in get_completion: {str(e)}")
            raise
    
    def analyze_application(self, new_application, similar_applications):
        """
        Analyze the application against similar applications.
        
        Args:
            new_application (dict): The new application data
            similar_applications (list): List of similar applications
            
        Returns:
            dict: The analysis result in structured format
        """
        try:
            # Convert similar applications to a more compact indexed dataset
            indexed_dataset = {
                "total_applications": len(similar_applications),
                "applications": similar_applications
            }
            
            # Use the new get_investment_analysis_prompt
            prompt = get_investment_analysis_prompt(new_application, indexed_dataset)
            
            # Log the full prompt for debugging
            logger.info(f"Full Prompt: {prompt}")
            
            response_text = self.get_completion(prompt, max_tokens=2000)
            
            # Additional parsing to remove Markdown code block markers if present
            if response_text.startswith('```json'):
                response_text = response_text.replace('```json', '').replace('```', '').strip()
            
            try:
                return json.loads(response_text)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse response: {response_text}")
                return {
                    "Error": "Failed to parse analysis response",
                    "Raw response": response_text
                }
        
        except Exception as e:
            logger.error(f"Error in analyze_application: {str(e)}")
            raise

# Function to generate debug report
def generate_debug_report():
    """
    Generate a comprehensive debug report for Azure configurations
    """
    report = [
        "Azure Configuration Debug Report",
        "=" * 50,
        "OpenAI Configuration:",
        f"Endpoint: {os.getenv('AZURE_OPENAI_ENDPOINT', 'NOT SET')}",
        f"Deployment Name: {os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME', 'NOT SET')}",
        f"API Version: {os.getenv('AZURE_OPENAI_VERSION', 'NOT SET')}",
        f"API Key Provided: {'Yes' if os.getenv('AZURE_OPENAI_API_KEY') else 'No'}",
        "",
        "Azure Search Configuration:",
        f"Endpoint: {os.getenv('AZURE_SEARCH_ENDPOINT', 'NOT SET')}",
        f"Index Name: {os.getenv('AZURE_SEARCH_INDEX_NAME', 'NOT SET')}",
        f"API Key Provided: {'Yes' if os.getenv('AZURE_SEARCH_API_KEY') else 'No'}",
        "\nTroubleshooting Steps:",
        "1. Verify Azure resources are active",
        "2. Check deployment names match exactly in Azure Portal",
        "3. Confirm API key permissions",
        "4. Validate network/firewall settings"
    ]
    return "\n".join(report)

# Initialize clients
search_client = AzureSearchClient()
openai_client = AzureOpenAIClient()

def find_similar_applications(application_data, filter_expr=None, top=3):
    """
    Find similar applications from Azure AI Search.
    
    Args:
        application_data (dict): The application data
        filter_expr (str, optional): OData filter expression
        top (int, optional): Number of similar applications to retrieve
        
    Returns:
        list: The similar applications
    """
    # Chunk and embed the input data
    json_chunks = chunk_json(application_data)
    embeddings = generate_embeddings(json_chunks)
    
    return search_client.search_similar_applications_with_embeddings(
        embeddings, 
        application_data, 
        filter_expr, 
        top
    )

def analyze_application(new_application, similar_applications):
    """
    Analyze the application against similar applications.
    
    Args:
        new_application (dict): The new application data
        similar_applications (list): List of similar applications
        
    Returns:
        dict: The analysis result
    """
    return openai_client.analyze_application(new_application, similar_applications)


@app.get("/", status_code=200)
async def root():
    return {"message": "This app is called using HTTP Get Method"}


@app.post("/analyze-application")
async def analyze_application_endpoint(request: ApplicationAnalysisRequest):
    """
    REST API endpoint for application analysis
    """
    try:
        # Sanitize input data to remove any problematic values
        sanitized_application_data = sanitize_data(request.application_data)
        
        # Log the incoming application data for debugging
        logger.info(f"Received application data: {json.dumps(sanitized_application_data, cls=CustomJSONEncoder)}")
        
        # Find similar applications with explicit chunk and embedding logging
        print("\n--- JSON Chunking Process ---")
        json_chunks = chunk_json(sanitized_application_data)
        
        print("\n--- Embedding Generation Process ---")
        embeddings = generate_embeddings(json_chunks)
        
        # Find similar applications
        similar_applications = find_similar_applications(
            sanitized_application_data, 
            request.filter_expr, 
            request.top_similar
        )
        
        # Analyze application
        analysis_result = analyze_application(sanitized_application_data, similar_applications)
        
        # Sanitize final result before returning
        return {
            # "similar_applications": sanitize_data(similar_applications),
            "analysis_result": sanitize_data(analysis_result)
        }
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8765))
    print(f"Starting server on port {port}...")
    print(f"REST endpoint: http://localhost:{port}/analyze-application")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
