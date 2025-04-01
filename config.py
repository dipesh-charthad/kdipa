import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    # Azure OpenAI Configuration
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
    AZURE_OPENAI_VERSION = os.getenv("AZURE_OPENAI_VERSION", "2025-01-01-preview")
    AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "")
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "")

    # Azure Search Configuration
    AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT", "")
    AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME", "")
    AZURE_SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY", "")

    @classmethod
    def validate_config(cls):
        """
        Validate critical configuration parameters
        """
        errors = []
        
        if not cls.AZURE_OPENAI_ENDPOINT:
            errors.append("AZURE_OPENAI_ENDPOINT is missing")
        if not cls.AZURE_OPENAI_API_KEY:
            errors.append("AZURE_OPENAI_API_KEY is missing")
        if not cls.AZURE_SEARCH_ENDPOINT:
            errors.append("AZURE_SEARCH_ENDPOINT is missing")
        if not cls.AZURE_SEARCH_API_KEY:
            errors.append("AZURE_SEARCH_API_KEY is missing")
        
        if errors:
            raise ValueError("\n".join(errors))

    @classmethod
    def get_openai_config(cls):
        """
        Get OpenAI configuration dictionary
        """
        return {
            "azure_endpoint": cls.AZURE_OPENAI_ENDPOINT,
            "api_key": cls.AZURE_OPENAI_API_KEY,
            "api_version": cls.AZURE_OPENAI_VERSION
        }
