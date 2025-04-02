import os
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from openai import AzureOpenAI

# Load environment variables from .env file
load_dotenv()

def test_azure_openai_connection():
    """Test connection to Azure OpenAI"""
    try:
        # Set up the Azure OpenAI client
        client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        
        # Test by listing available models
        models = client.models.list()
        model_names = [model.id for model in models]
        
        print("\n✅ Azure OpenAI Connection Successful!")
        print(f"Available models: {', '.join(model_names)}")
        return True
    
    except Exception as e:
        print(f"\n❌ Azure OpenAI Connection Failed: {str(e)}")
        print("Please check your API key, endpoint, and API version.")
        return False

def test_azure_search_connection():
    """Test connection to Azure Cognitive Search"""
    try:
        # Set up the Azure Search client
        search_client = SearchClient(
            endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
            index_name=os.getenv("AZURE_SEARCH_INDEX_NAME"),
            credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_API_KEY"))
        )
        
        # Test by getting index statistics
        stats = search_client.get_document_count()
        
        print("\n✅ Azure Search Connection Successful!")
        print(f"Documents in index '{os.getenv('AZURE_SEARCH_INDEX_NAME')}': {stats}")
        return True
    
    except Exception as e:
        print(f"\n❌ Azure Search Connection Failed: {str(e)}")
        print("Please check your Search service endpoint, API key, and index name.")
        return False

def main():
    """Main function to test all connections"""
    print("Testing Azure connections...\n")
    
    openai_success = test_azure_openai_connection()
    search_success = test_azure_search_connection()
    
    if openai_success and search_success:
        print("\n🎉 All Azure components connected successfully!")
    else:
        print("\n⚠️ Some connections failed. Please check the details above.")

if __name__ == "__main__":
    main()

