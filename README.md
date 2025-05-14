# Environment Setup

## Local Development Configuration

This application uses environment variables for configuration, particularly for Azure Key Vault integration. Before running the application locally, you'll need to create a `.env` file in the root directory with the following variables:

```
# Azure Key Vault Configuration
USE_KEY_VAULT=true
AZURE_TENANT_ID=your_tenant_id
AZURE_CLIENT_ID=your_client_id
AZURE_CLIENT_SECRET=your_client_secret
AZURE_KEYVAULT_URL=your_keyvault_url
AZURE_SUBSCRIPTION_ID=your_subscription_id
AZURE_KEYVAULT_NAME=your_keyvault_name
RESOURCE_GROUP=your_resource_group
AZURE_OBJECT_ID=your_object_id
```

### Required Variables

- `USE_KEY_VAULT`: Set to `true` to enable Azure Key Vault integration
- `AZURE_TENANT_ID`: Your Azure AD tenant ID
- `AZURE_CLIENT_ID`: The client (application) ID registered in Azure AD
- `AZURE_CLIENT_SECRET`: The client secret for your registered application
- `AZURE_KEYVAULT_URL`: The URL of your Azure Key Vault instance
- `AZURE_SUBSCRIPTION_ID`: Your Azure subscription ID
- `AZURE_KEYVAULT_NAME`: The name of your Azure Key Vault
- `RESOURCE_GROUP`: The resource group containing your Key Vault
- `AZURE_OBJECT_ID`: The object ID for your application or service principal

## Setting Up Azure Key Vault Access

To get the necessary credentials:

1. Create an Azure Key Vault in your Azure portal
2. Register an application in Azure Active Directory
3. Grant the application appropriate permissions to access the Key Vault
4. Note down all the required IDs and secrets

## Security Notes

- Never commit your `.env` file to version control
- This repository includes `.env` in the `.gitignore` file
- For production deployment, use environment variables configured in your hosting environment
