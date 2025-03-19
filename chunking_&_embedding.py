
import os
import csv
import json
import pandas as pd
import numpy as np
import uuid
from datetime import datetime
from openai import AzureOpenAI
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Azure OpenAI credentials from .env file
api_key = os.getenv("AZURE_OPENAI_API_KEY")
api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
deployment_name = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

# Debug: Print environment variables
print("OpenAI API Key found:", "Yes" if api_key else "No")
print("OpenAI API Version found:", "Yes" if api_version else "No")
print("OpenAI Endpoint found:", "Yes" if azure_endpoint else "No")
print("OpenAI Deployment found:", "Yes" if deployment_name else "No")

# Check for missing required variables
if not api_key or not api_version or not azure_endpoint or not deployment_name:
    raise ValueError("Azure OpenAI credentials not found in environment variables")

# Initialize Azure OpenAI client
openai_client = AzureOpenAI(
    api_key=api_key,
    api_version=api_version,
    azure_endpoint=azure_endpoint
)

def load_csv_data(file_path):
    """Load CSV data into a pandas DataFrame."""
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded CSV file with {len(df)} rows and {len(df.columns)} columns.")
        print(f"Columns: {', '.join(df.columns)}")
        return df
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None

def chunk_single_columns(df):
    """
    Create a chunk for each individual column in the dataframe.
    
    Args:
        df: Input DataFrame
    
    Returns:
        List of chunks, each containing a single column
    """
    all_columns = list(df.columns)
    chunks = []
    
    # Create a chunk for each column
    for column in all_columns:
        # Create chunk with just this column
        chunk_df = df[[column]].copy()
        
        # Convert to string for embedding
        chunk_text = chunk_df.to_csv(index=False)
        
        # Get sample values (first 5 values or fewer)
        sample_values = chunk_df[column].head(5).tolist()
        sample_values = [str(val) for val in sample_values if pd.notna(val)]
        
        # Generate a description of the column based on its content
        column_description = f"Column '{column}' containing data like: {', '.join(sample_values[:3])}"
        
        # Store info about this chunk
        chunk_info = {
            "id": str(uuid.uuid4()),
            "chunk_type": "column",
            "column": column,
            "text": chunk_text,
            "sample_values": sample_values,
            "description": column_description,
            "data": chunk_df,
            "created_at": datetime.now().isoformat()
        }
        chunks.append(chunk_info)
    
    print(f"Created {len(chunks)} column chunks")
    return chunks

def chunk_individual_rows(df, max_rows=None):
    """
    Create a separate chunk for each individual row in the dataframe.
    
    Args:
        df: Input DataFrame
        max_rows: Maximum number of rows to process (default: None = all rows)
    
    Returns:
        List of chunks, each containing a single row
    """
    chunks = []
    total_rows = len(df)
    
    # Limit the number of rows if specified
    if max_rows is not None and max_rows < total_rows:
        total_rows = max_rows
        print(f"Processing only the first {max_rows} rows as requested")
    
    # Create a chunk for each individual row
    for i in range(total_rows):
        # Get the single row as a DataFrame (not a Series)
        row_df = df.iloc[[i]].copy()
        
        # Convert to string for embedding (dictionary format is often more readable for single rows)
        row_dict = row_df.iloc[0].to_dict()
        row_text = json.dumps(row_dict, default=str)
        
        # Create a summary of the row data
        row_summary = []
        for col, val in row_dict.items():
            if pd.notna(val):
                row_summary.append(f"{col}: {val}")
        
        # Generate a description of the row
        row_description = f"Row {i} data: " + "; ".join(row_summary[:5])
        if len(row_summary) > 5:
            row_description += f"; and {len(row_summary) - 5} more fields"
        
        # Store info about this chunk
        chunk_info = {
            "id": str(uuid.uuid4()),
            "chunk_type": "single_row",
            "row_index": i,
            "text": row_text,
            "description": row_description,
            "data": row_df,
            "created_at": datetime.now().isoformat()
        }
        chunks.append(chunk_info)
    
    print(f"Created {len(chunks)} individual row chunks")
    return chunks

def get_embeddings(chunks, batch_size=100):
    """
    Get embeddings for each chunk using Azure OpenAI.
    Processes chunks in batches to avoid rate limits.
    
    Args:
        chunks: List of chunks to embed
        batch_size: Number of chunks to process in each batch (default: 100)
    
    Returns:
        List of chunks with embeddings
    """
    embeddings = []
    
    print(f"Generating embeddings using model: {deployment_name}")
    
    # Process chunks in batches
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1} of {(len(chunks) + batch_size - 1)//batch_size}")
        
        for chunk in tqdm(batch, desc=f"Batch {i//batch_size + 1}"):
            try:
                response = openai_client.embeddings.create(
                    input=chunk["text"],
                    model=deployment_name
                )
                
                # Extract embedding from response
                embedding_vector = response.data[0].embedding
                
                # Add embedding to chunk info
                chunk_with_embedding = chunk.copy()
                chunk_with_embedding["embedding"] = embedding_vector
                chunk_with_embedding["vector_dimensions"] = len(embedding_vector)
                embeddings.append(chunk_with_embedding)
                
            except Exception as e:
                print(f"Error generating embedding for chunk {chunk.get('id')}: {e}")
                # Add empty embedding to maintain data structure
                chunk_with_embedding = chunk.copy()
                chunk_with_embedding["embedding"] = []
                chunk_with_embedding["vector_dimensions"] = 0
                embeddings.append(chunk_with_embedding)
    
    print(f"Generated embeddings for {len(embeddings)} chunks")
    return embeddings

def save_as_csv(embedded_chunks, output_file):
    """Save embedded chunks to CSV file."""
    try:
        # Create a list to store chunk data
        chunk_data = []
        
        for chunk in embedded_chunks:
            # Convert embedding to string
            embedding_str = json.dumps(chunk["embedding"])
            
            # Create a base record for this chunk
            chunk_record = {
                "id": chunk["id"],
                "chunk_type": chunk["chunk_type"],
                "description": chunk["description"],
                "embedding": embedding_str,
                "created_at": chunk["created_at"]
            }
            
            # Add type-specific fields
            if chunk["chunk_type"] == "column":
                chunk_record["column_name"] = chunk["column"]
                chunk_record["sample_values"] = str(chunk["sample_values"])
            elif chunk["chunk_type"] == "single_row":
                chunk_record["row_index"] = chunk["row_index"]
            
            chunk_data.append(chunk_record)
        
        # Convert to DataFrame and save
        output_df = pd.DataFrame(chunk_data)
        output_df.to_csv(output_file, index=False)
        print(f"Successfully saved chunk embeddings to CSV: {output_file}")
        
    except Exception as e:
        print(f"Error saving to CSV: {e}")

def save_as_json(embedded_chunks, output_file):
    """Save embedded chunks to JSON file."""
    try:
        # Create serializable data structure
        json_data = []
        
        for chunk in embedded_chunks:
            # Create a copy without the DataFrame
            chunk_dict = {k: v for k, v in chunk.items() if k != 'data'}
            json_data.append(chunk_dict)
        
        # Save to JSON file
        with open(output_file, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"Successfully saved chunk embeddings to JSON: {output_file}")
        
    except Exception as e:
        print(f"Error saving to JSON: {e}")

def main():
    # File paths
    input_file = r"C:\Users\Dipesh.charthad\OneDrive - iLink Systems Inc\Desktop\Desktop\KDIPA\Code\new_output_final.csv"  # Change to your input file path
    column_chunks_csv = "column_617.csv"
    column_chunks_json = "column_617.json"
    row_chunks_csv = "row_73.csv"
    row_chunks_json = "row_73.json"
    
    # Load data
    df = load_csv_data(input_file)
    if df is None:
        return
    
    # Choose chunking mode
    chunk_mode = input("Choose chunking mode (1 for column-wise, 2 for row-wise, 3 for both): ")
    
    if chunk_mode in ["1", "3"]:
        # Column-wise chunking
        print("\nPerforming column-wise chunking...")
        column_chunks = chunk_single_columns(df)
        column_chunks_with_embeddings = get_embeddings(column_chunks)
        save_as_csv(column_chunks_with_embeddings, column_chunks_csv)
        save_as_json(column_chunks_with_embeddings, column_chunks_json)
        print("Column chunking completed.")
    
    if chunk_mode in ["2", "3"]:
        # Row-wise chunking - one chunk per row
        print("\nPerforming individual row chunking...")
        
        # Ask if user wants to limit the number of rows
        try:
            max_rows_input = input("Enter maximum number of rows to process (press Enter for all rows): ")
            max_rows = int(max_rows_input) if max_rows_input else None
        except ValueError:
            print("Invalid input. Processing all rows.")
            max_rows = None
            
        row_chunks = chunk_individual_rows(df, max_rows)
        
        # For large datasets, confirm before proceeding with embedding
        if len(row_chunks) > 1000:
            confirm = input(f"You are about to generate embeddings for {len(row_chunks)} rows. This may be expensive and time-consuming. Proceed? (y/n): ")
            if confirm.lower() != 'y':
                print("Row chunking aborted.")
                return
        
        # Process in batches to avoid rate limits
        batch_size = 100
        row_chunks_with_embeddings = get_embeddings(row_chunks, batch_size)
        save_as_csv(row_chunks_with_embeddings, row_chunks_csv)
        save_as_json(row_chunks_with_embeddings, row_chunks_json)
        print("Row chunking completed.")
    
    print("Process completed successfully!")

if __name__ == "__main__":
    main()