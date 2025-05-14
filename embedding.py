import json
import math
import numpy as np
import tiktoken
import logging
from openai import AzureOpenAI
from config import Config
import concurrent.futures

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        # Handle numpy arrays
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        
        # Handle numpy numeric types and regular float NaN combined
        if isinstance(obj, (np.float32, np.float64, float)):
            if math.isnan(obj) or math.isinf(obj):
                return None
            return float(obj) if isinstance(obj, (np.float32, np.float64)) else obj
        
        return super().default(obj)

def clean_data(data):
    """
    Recursively clean data to make it JSON-compliant.
    """
    if isinstance(data, dict):
        return {k: clean_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_data(v) for v in data]
    elif isinstance(data, (np.ndarray, np.generic, float)):
        if np.isnan(data) or np.isinf(data) or (isinstance(data, float) and (math.isnan(data) or math.isinf(data))):
            return None
        
        # Extract this nested conditional into independent statements
        if hasattr(data, 'item'):
            return data.item()
        elif isinstance(data, (np.ndarray, np.generic)):
            return data.tolist()
        else:
            return data
    
    return data


def chunk_json(json_data, max_tokens=500):
    """
    Optimized chunking implementation that minimizes tokenization overhead
    """
    # Quick check - if data is small, don't bother chunking
    serialized = json.dumps(json_data)
    if len(serialized) < 4000:  # Rough estimate that this will be under max_tokens
        return [json_data]
    
    return _process_chunking(json_data, max_tokens)

def _process_chunking(json_data, max_tokens):
    """Helper function to handle the chunking logic and reduce cognitive complexity"""
    # Copy the data to avoid modifying the original
    json_data_copy = json_data.copy() if isinstance(json_data, dict) else json_data
    
    # Process large string fields
    _truncate_large_strings(json_data_copy)
    
    # Get encoder and check token size
    encoding = tiktoken.get_encoding("cl100k_base")
    serialized_data = json.dumps(json_data_copy, separators=(',', ':'))
    tokens = encoding.encode(serialized_data)
    
    if len(tokens) <= max_tokens * 1.5:  # Give some buffer
        return [json_data_copy]
    
    # Determine chunking strategy based on data type
    chunks = _apply_chunking_strategy(json_data_copy, encoding, max_tokens)
    
    # Fallback if no chunks were created
    return chunks if chunks else [json_data_copy]

def _apply_chunking_strategy(data, encoding, max_tokens):
    """Choose and apply appropriate chunking strategy based on data type"""
    if isinstance(data, list):
        return _chunk_list(data, encoding, max_tokens)
    elif isinstance(data, dict):
        return _chunk_dict(data, encoding, max_tokens)
    return [data]  # Default fallback

def _truncate_large_strings(data):
    """Helper function to truncate large string fields"""
    # Handle dictionary case
    if isinstance(data, dict):
        _truncate_dict_strings(data)
    # Handle list case
    elif isinstance(data, list):
        _truncate_list_strings(data)

def _truncate_dict_strings(data_dict):
    """Helper for truncating strings in dictionaries"""
    for key, value in data_dict.items():
        if isinstance(value, str) and len(value) > 5000:
            data_dict[key] = value[:5000] + "..."
        elif isinstance(value, dict):
            _truncate_dict_strings(value)
        elif isinstance(value, list):
            _truncate_list_strings(value)

def _truncate_list_strings(data_list):
    """Helper for truncating strings in lists"""
    for i, item in enumerate(data_list):
        if isinstance(item, str) and len(item) > 5000:
            data_list[i] = item[:5000] + "..."
        elif isinstance(item, dict):
            _truncate_dict_strings(item)
        elif isinstance(item, list):
            _truncate_list_strings(item)

def _chunk_list(data_list, encoding, max_tokens):
    """Helper function to chunk a list"""
    chunks = []
    current_chunk = []
    current_size = 0
    
    for item in data_list:
        item_json = json.dumps(item, separators=(',', ':'))
        item_tokens = len(encoding.encode(item_json))
        
        if current_size + item_tokens > max_tokens and current_chunk:
            chunks.append(current_chunk)
            current_chunk = [item]
            current_size = item_tokens
        else:
            current_chunk.append(item)
            current_size += item_tokens
    
    if current_chunk:
        chunks.append(current_chunk)
        
    return chunks if chunks else [data_list]

def _chunk_dict(data_dict, encoding, max_tokens):
    """Helper function to chunk a dictionary"""
    chunks = []
    current_chunk = {}
    current_size = 0
    
    # Sort keys by value size for more efficient chunking
    sorted_keys = sorted(
        data_dict.keys(), 
        key=lambda k: len(json.dumps(data_dict[k], separators=(',', ':')))
    )
    
    for key in sorted_keys:
        value = data_dict[key]
        pair_json = json.dumps({key: value}, separators=(',', ':'))
        pair_tokens = len(encoding.encode(pair_json))
        
        if current_size + pair_tokens > max_tokens and current_chunk:
            chunks.append(current_chunk)
            current_chunk = {key: value}
            current_size = pair_tokens
        else:
            current_chunk[key] = value
            current_size += pair_tokens
    
    if current_chunk:
        chunks.append(current_chunk)
        
    return chunks if chunks else [data_dict]

def fix_json_chunk(chunk_str):
    """Helper function to try to fix truncated JSON"""
    # Count opening and closing braces
    open_braces = chunk_str.count('{')
    close_braces = chunk_str.count('}')
    open_brackets = chunk_str.count('[')
    close_brackets = chunk_str.count(']')
    
    # Add missing closing braces and brackets
    result = chunk_str
    for _ in range(open_braces - close_braces):
        result += '}'
    for _ in range(open_brackets - close_brackets):
        result += ']'
        
    return result


# Create a thread pool executor for parallel processing
executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)

def generate_embeddings(chunks):
    """
    Generate embeddings for JSON chunks using Azure OpenAI with parallel processing.
    """
    logger = logging.getLogger(__name__)
    
    # Create a single client to be reused across threads
    embedding_client = AzureOpenAI(
        azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
        api_key=Config.AZURE_OPENAI_API_KEY,
        api_version=Config.AZURE_OPENAI_VERSION
    )
    
    def generate_single_embedding(chunk_info):
        """Process a single chunk and return its embedding"""
        i, chunk = chunk_info
        try:
            # Convert chunk to string for embedding
            chunk_str = json.dumps(chunk, cls=CustomJSONEncoder)
            
            # Check if chunk is too large
            if len(chunk_str) > 8000:
                logger.warning(f"Chunk {i} exceeds size limit. Truncating...")
                chunk_str = chunk_str[:8000]
            
            response = embedding_client.embeddings.create(
                input=chunk_str,
                model=Config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT
            )
            
            # Extract embedding and clean it
            embedding = response.data[0].embedding
            cleaned_embedding = [
                float(val) if not (np.isnan(val) or np.isinf(val)) else 0.0 
                for val in embedding
            ]
            
            return i, cleaned_embedding
        except Exception as e:
            logger.error(f"Embedding error for chunk {i}: {str(e)}")
            # Return zero vector as fallback
            return i, [0.0] * 1536
    
    # Submit all chunks for parallel processing
    futures = [executor.submit(generate_single_embedding, (i, chunk)) 
               for i, chunk in enumerate(chunks)]
    
    # Collect results in the correct order
    result_map = {}
    for future in concurrent.futures.as_completed(futures):
        idx, embedding = future.result()
        result_map[idx] = embedding
    
    # Return embeddings in original chunk order
    return [result_map[i] for i in range(len(chunks))]