# import os
# import json
# from azure.core.credentials import AzureKeyCredential
# from azure.search.documents import SearchClient

# # Load environment variables
# endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
# api_key = os.getenv("AZURE_SEARCH_API_KEY")
# index_name = "aiml-row"

# # Initialize the client
# search_client = SearchClient(endpoint=endpoint,
#                              index_name=index_name,
#                              credential=AzureKeyCredential(api_key))

# # Load data from JSON file
# with open(r'C:\Users\Dipesh.charthad\OneDrive - iLink Systems Inc\Desktop\Desktop\KDIPA\Code\imp\row.json', 'r') as file:
#     raw_data = json.load(file)

# # Prepare data for upload
# documents = []
# for item in raw_data:
#     document = {
#         "id": item.get("id"),
#         "metadata_spo_item_name": item.get("description"),
#         "metadata_spo_item_last_modified": item.get("created_at"),
#         "content": item.get("text"),
#         "vectorcontent": json.dumps(item.get("embedding"))  # Convert embedding list to JSON string
#     }
#     documents.append(document)

# # Upload documents to Azure AI Search
# try:
#     result = search_client.upload_documents(documents=documents)
#     print(f"Upload result: {result}")
# except Exception as e:
#     print(f"Error uploading documents: {e}")


# import os
# import json
# from datetime import datetime
# from azure.core.credentials import AzureKeyCredential
# from azure.search.documents import SearchClient

# # Load environment variables
# endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
# api_key = os.getenv("AZURE_SEARCH_API_KEY")
# index_name = "aiml-row-test"

# # Initialize the client
# search_client = SearchClient(endpoint=endpoint,
#                              index_name=index_name,
#                              credential=AzureKeyCredential(api_key))

# # Function to format datetime to ISO 8601 with UTC
# def format_datetime(dt_str):
#     try:
#         dt = datetime.fromisoformat(dt_str)
#         return dt.isoformat() + 'Z'  # Add 'Z' to indicate UTC
#     except Exception as e:
#         print(f"Invalid datetime format: {dt_str} - Error: {e}")
#         return None

# # Load data from JSON file
# with open(r'C:\Users\Dipesh.charthad\OneDrive - iLink Systems Inc\Desktop\Desktop\KDIPA\Code\imp\row.json', 'r') as file:
#     raw_data = json.load(file)

# # Prepare data for upload
# documents = []
# for item in raw_data:
#     document = {
#         "id": item.get("id"),
#         "metadata_spo_item_name": item.get("description"),
#         "metadata_spo_item_last_modified": format_datetime(item.get("created_at")),
#         "content": item.get("text"),
#         "vectorcontent": json.dumps(item.get("embedding"))  # Convert embedding list to JSON string
#     }
#     documents.append(document)

# # Filter out documents with invalid datetime
# documents = [doc for doc in documents if doc["metadata_spo_item_last_modified"] is not None]

# # Upload documents to Azure AI Search
# try:
#     result = search_client.upload_documents(documents=documents)
#     print(f"Upload result: {result}")
# except Exception as e:
#     print(f"Error uploading documents: {e}")

import os
import json
from datetime import datetime
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient

# Load environment variables
endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
api_key = os.getenv("AZURE_SEARCH_API_KEY")
index_name = "aiml-index"

# Initialize the client
search_client = SearchClient(endpoint=endpoint,
                             index_name=index_name,
                             credential=AzureKeyCredential(api_key))

# Function to format datetime to ISO 8601 with UTC
def format_datetime(dt_str):
    try:
        dt = datetime.fromisoformat(dt_str)
        return dt.isoformat() + 'Z'  # Add 'Z' to indicate UTC
    except Exception as e:
        print(f"Invalid datetime format: {dt_str} - Error: {e}")
        return None

# Load data from JSON file
with open(r'C:\Users\Dipesh.charthad\OneDrive - iLink Systems Inc\Desktop\Desktop\KDIPA\Code\imp\row_73.json', 'r') as file:
    raw_data = json.load(file)

# Check if raw_data is a dictionary or a list
if isinstance(raw_data, dict):
    # If it's a dictionary, convert it to a list with a single item
    raw_data = [raw_data]

# Prepare data for upload
documents = []
for item in raw_data:
    # Make sure embedding is a list/array, not a string
    embedding = item.get("embedding")
    if isinstance(embedding, str):
        try:
            # If embedding is a string, try to parse it as JSON
            embedding = json.loads(embedding)
        except json.JSONDecodeError:
            print(f"Error: Embedding for item {item.get('id')} is not a valid JSON string")
            continue
    
    document = {
        "id": item.get("id"),
        "metadata_spo_item_name": item.get("description"),
        "metadata_spo_item_last_modified": format_datetime(item.get("created_at")),
        "content": item.get("text"),
        "vectorcontent": embedding  # Pass the embedding as an array, not as a JSON string
    }
    documents.append(document)

# Filter out documents with invalid datetime
documents = [doc for doc in documents if doc["metadata_spo_item_last_modified"] is not None]

# Debug: Print the first document to check format
if documents:
    print("First document format:")
    print(documents[0])

# Upload documents to Azure AI Search
try:
    result = search_client.upload_documents(documents=documents)
    print(f"Upload result: {result}")
except Exception as e:
    print(f"Error uploading documents: {e}")
    # Print more detailed error information
    import traceback
    traceback.print_exc()