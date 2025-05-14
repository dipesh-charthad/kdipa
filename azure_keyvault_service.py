from azure.identity import ClientSecretCredential
from azure.keyvault.secrets import SecretClient
import os
import logging
from dotenv import load_dotenv

# Load environment variables for Key Vault authentication
load_dotenv()

class KeyVaultClient:
    def __init__(self):
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        
        # Load Key Vault authentication details
        self.tenant_id = os.getenv("AZURE_TENANT_ID", "")
        self.client_id = os.getenv("AZURE_CLIENT_ID", "")
        self.client_secret = os.getenv("AZURE_CLIENT_SECRET", "")
        self.key_vault_url = os.getenv("AZURE_KEYVAULT_URL", "")
        
        # Validate configuration
        self._validate_config()
        
        # Initialize the credential and client
        try:
            self.credential = ClientSecretCredential(
                tenant_id=self.tenant_id,
                client_id=self.client_id,
                client_secret=self.client_secret
            )
            
            self.secret_client = SecretClient(
                vault_url=self.key_vault_url, 
                credential=self.credential
            )
            
            self.logger.info("Key Vault client initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Key Vault client: {str(e)}")
            raise
    
    def _validate_config(self):
        """
        Validate Key Vault configuration parameters
        """
        errors = []
        
        if not self.tenant_id:
            errors.append("AZURE_TENANT_ID is missing")
        if not self.client_id:
            errors.append("AZURE_CLIENT_ID is missing")
        if not self.client_secret:
            errors.append("AZURE_CLIENT_SECRET is missing")
        if not self.key_vault_url:
            errors.append("AZURE_KEY_VAULT_URL is missing")
        
        if errors:
            self.logger.error("Key Vault configuration validation failed")
            raise ValueError("\n".join(errors))
    
    def get_secret(self, secret_name):
        """
        Retrieve a secret from Azure Key Vault
        
        Args:
            secret_name (str): The name of the secret to retrieve
            
        Returns:
            str: The secret value
        """
        try:
            secret = self.secret_client.get_secret(secret_name)
            return secret.value
        except Exception as e:
            self.logger.error(f"Failed to retrieve secret '{secret_name}': {str(e)}")
            return None

# Singleton pattern for KeyVaultClient
_key_vault_client = None

def get_key_vault_client():
    """
    Get or create a KeyVaultClient instance (singleton pattern).
    
    Returns:
        KeyVaultClient: The global KeyVault client instance
        
    Raises:
        ValueError: If configuration validation fails
    """
    global _key_vault_client
    try:
        if _key_vault_client is None:
            _key_vault_client = KeyVaultClient()
        return _key_vault_client
    except ValueError as e:
        logging.error(f"Failed to initialize KeyVaultClient: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error in get_key_vault_client: {str(e)}")
        raise