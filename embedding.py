import json
import math
import numpy as np
import tiktoken
import logging
from openai import AzureOpenAI
from config import Config

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        # Handle numpy arrays
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        
        # Handle numpy numeric types
        if isinstance(obj, (np.float32, np.float64)):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        
        # Handle regular float NaN
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None
        
        return super().default(obj)

def sanitize_data(data):
    """
    Recursively sanitize data to make it JSON-compliant.
    """
    if isinstance(data, dict):
        return {k: sanitize_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_data(v) for v in data]
    elif isinstance(data, (np.ndarray, np.generic)):
        if np.isnan(data) or np.isinf(data):
            return None
        return data.item() if hasattr(data, 'item') else data.tolist()
    elif isinstance(data, float):
        if math.isnan(data) or math.isinf(data):
            return None
    return data

def chunk_json(json_data, max_tokens=500):
    """
    Chunk JSON data into smaller pieces while preserving structure.
    """
    encoding = tiktoken.get_encoding("cl100k_base")
    
    serialized_data = json.dumps(json_data, cls=CustomJSONEncoder)
    tokens = encoding.encode(serialized_data)
    
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for token in tokens:
        current_chunk.append(token)
        current_tokens += 1
        
        if current_tokens >= max_tokens:
            decoded_chunk = encoding.decode(current_chunk)
            try:
                chunk = json.loads(decoded_chunk, parse_float=float)
                chunks.append(chunk)
            except json.JSONDecodeError:
                # Fallback for partial chunks
                chunks.append(json.loads(decoded_chunk + "}", parse_float=float))
            
            current_chunk = []
            current_tokens = 0
    
    # Add remaining tokens
    if current_chunk:
        decoded_chunk = encoding.decode(current_chunk)
        try:
            chunk = json.loads(decoded_chunk, parse_float=float)
            chunks.append(chunk)
        except json.JSONDecodeError:
            chunks.append(json.loads(decoded_chunk + "}", parse_float=float))
    
    return chunks


def generate_embeddings(chunks):
    """
    Generate embeddings for JSON chunks using Azure OpenAI.
    """
    # Initialize Azure OpenAI client for embedding
    embedding_client = AzureOpenAI(
        azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
        api_key=Config.AZURE_OPENAI_API_KEY,
        api_version=Config.AZURE_OPENAI_VERSION
    )
    
    logger = logging.getLogger(__name__)
    embeddings = []
    
    for i, chunk in enumerate(chunks, 1):
        try:
            # Convert chunk back to string for embedding
            chunk_str = json.dumps(chunk, cls=CustomJSONEncoder)
            
            response = embedding_client.embeddings.create(
                input=chunk_str,
                model=Config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT
            )
            
            # Extract the first embedding
            embedding = response.data[0].embedding
            
            # Sanitize embedding
            sanitized_embedding = [
                float(val) if not (np.isnan(val) or np.isinf(val)) else 0.0 
                for val in embedding
            ]
            
            embeddings.append(sanitized_embedding)
        except Exception as e:
            logger.error(f"Embedding generation error for chunk {i}: {str(e)}")
            embeddings.append(None)
    
    return embeddings
