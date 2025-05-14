import os
from dotenv import load_dotenv
from azure_keyvault_service import get_key_vault_client

# Load environment variables
load_dotenv()

class Config:
    # Determine if we should use Key Vault
    USE_KEY_VAULT = os.getenv("USE_KEY_VAULT", "false").lower() == "true"
    
    # Azure OpenAI Configuration
    AZURE_OPENAI_ENDPOINT = None
    AZURE_OPENAI_API_KEY = None
    AZURE_OPENAI_VERSION = None
    AZURE_OPENAI_DEPLOYMENT_NAME = None
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT = None

    # Azure Search Configuration
    AZURE_SEARCH_ENDPOINT = None
    AZURE_SEARCH_INDEX_NAME = None
    AZURE_SEARCH_API_KEY = None
    
    @classmethod
    def load_config(cls):
        """
        Load configuration either from environment variables or Key Vault
        """
        if cls.USE_KEY_VAULT:
            # Initialize Key Vault client
            key_vault = get_key_vault_client()
            
            # Load OpenAI configuration from Key Vault
            cls.AZURE_OPENAI_ENDPOINT = key_vault.get_secret("openaiendpoint")
            cls.AZURE_OPENAI_API_KEY = key_vault.get_secret("openaikey")
            cls.AZURE_OPENAI_VERSION = os.getenv("AZURE_OPENAI_VERSION", "2025-01-01-preview")
            cls.AZURE_OPENAI_DEPLOYMENT_NAME = key_vault.get_secret("openaideployment")
            cls.AZURE_OPENAI_EMBEDDING_DEPLOYMENT = key_vault.get_secret("openaiembedding")
            cls.APPLICATION_INSIGHTS_INSTRUMENTATION_KEY = key_vault.get_secret("appinsightkey")
            
            # Load Azure Search configuration from Key Vault
            cls.AZURE_SEARCH_ENDPOINT = key_vault.get_secret("searchendpoint")
            cls.AZURE_SEARCH_INDEX_NAME = key_vault.get_secret("searchindexname")
            cls.AZURE_SEARCH_API_KEY = key_vault.get_secret("AIsearchKey")
        else:
            # Load from environment variables
            cls.AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
            cls.AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
            cls.AZURE_OPENAI_VERSION = os.getenv("AZURE_OPENAI_VERSION", "2025-01-01-preview")
            cls.AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "")
            cls.AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "")
            cls.APPLICATION_INSIGHTS_INSTRUMENTATION_KEY = os.getenv("APPLICATION_INSIGHTS_INSTRUMENTATION_KEY", "")
            
            cls.AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT", "")
            cls.AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME", "")
            cls.AZURE_SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY", "")

    @classmethod
    def validate_config(cls):
        """
        Validate critical configuration parameters.
        
        Raises:
            ValueError: If any required configuration parameter is missing
        """
        # Load configuration first
        cls.load_config()
        
        errors = []
        
        # Check for required OpenAI parameters
        if not cls.AZURE_OPENAI_ENDPOINT:
            errors.append("AZURE_OPENAI_ENDPOINT is missing")
        elif not isinstance(cls.AZURE_OPENAI_ENDPOINT, str):
            errors.append("AZURE_OPENAI_ENDPOINT must be a string")
            
        if not cls.AZURE_OPENAI_API_KEY:
            errors.append("AZURE_OPENAI_API_KEY is missing")
        elif not isinstance(cls.AZURE_OPENAI_API_KEY, str):
            errors.append("AZURE_OPENAI_API_KEY must be a string")
        
        # Check for required Search parameters
        if not cls.AZURE_SEARCH_ENDPOINT:
            errors.append("AZURE_SEARCH_ENDPOINT is missing")
        elif not isinstance(cls.AZURE_SEARCH_ENDPOINT, str):
            errors.append("AZURE_SEARCH_ENDPOINT must be a string")
            
        if not cls.AZURE_SEARCH_API_KEY:
            errors.append("AZURE_SEARCH_API_KEY is missing")
        elif not isinstance(cls.AZURE_SEARCH_API_KEY, str):
            errors.append("AZURE_SEARCH_API_KEY must be a string")
        
        # Check for deployment names
        if not cls.AZURE_OPENAI_DEPLOYMENT_NAME:
            errors.append("AZURE_OPENAI_DEPLOYMENT_NAME is missing")
        
        if not cls.AZURE_OPENAI_EMBEDDING_DEPLOYMENT:
            errors.append("AZURE_OPENAI_EMBEDDING_DEPLOYMENT is missing")
        
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

# Load configuration at module import time
Config.load_config()