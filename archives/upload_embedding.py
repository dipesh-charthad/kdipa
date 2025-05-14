import os
import json
import uuid
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Azure AI Search Configurations
SEARCH_SERVICE_NAME = os.getenv("AZURE_SEARCH_ENDPOINT")
SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")
SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")

# Local Path for JSON Files
LOCAL_JSON_FOLDER = r"C:\Users\Dipesh.charthad\Downloads\processed_pdfs\processed_pdfs"

# Initialize Search Client
search_client = SearchClient(
    endpoint=f"https://{SEARCH_SERVICE_NAME}.search.windows.net",
    index_name=SEARCH_INDEX_NAME,
    credential=AzureKeyCredential(SEARCH_API_KEY)
)

# Function to read and process JSON chunk files
def process_and_upload_json():
    documents = []
    
    for filename in os.listdir(LOCAL_JSON_FOLDER):
        if filename.endswith(".json"):  # Process only JSON files
            file_path = os.path.join(LOCAL_JSON_FOLDER, filename)

            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)

            document_name = data.get("document_name", "Unknown")
            total_chunks = data.get("total_chunks", 0)
            chunks = data.get("chunks", [])

            for i, chunk in enumerate(chunks):
                chunk_text = chunk.get("text", "")
                embedding = chunk.get("embedding", [])

                document = {
                    "id": str(uuid.uuid4()),  # Unique ID for each chunk
                    "metadata_spo_item_name": document_name,
                    "metadata_spo_item_path": file_path,
                    "metadata_spo_item_content_type": "application/pdf",
                    "metadata_spo_item_last_modified": "2025-03-05T12:00:00Z",
                    "metadata_spo_item_size": len(chunk_text),
                    "metadata_spo_item_weburi": f"file://{file_path}",
                    "content": chunk_text,
                    "vectorcontent": json.dumps(embedding)  # Store as JSON string
                }
                documents.append(document)

    # Upload to Azure AI Search
    if documents:
        search_client.upload_documents(documents=documents)
        print(f"Uploaded {len(documents)} chunks to Azure AI Search.")

# Execute the function
process_and_upload_json()
