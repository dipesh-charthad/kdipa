import os
import json
import logging
import requests
from config import Config

class AzureSearchClient:
    def __init__(self):
        self.endpoint = Config.AZURE_SEARCH_ENDPOINT
        self.index_name = Config.AZURE_SEARCH_INDEX_NAME
        self.api_key = Config.AZURE_SEARCH_API_KEY
        self.api_version = "2020-06-30"
        
        # Logging configuration
        self.logger = logging.getLogger(__name__)
        self.logger.info("Azure Search Configuration Initialized")
        
    def _get_headers(self):
        """Return the headers needed for Azure Search API requests."""
        return {
            "Content-Type": "application/json",
            "api-key": self.api_key
        }
        
    def search_similar_applications(self, app_data, filter_expr=None, top=3):
        """
        Search for similar applications in the Azure AI Search index.
        """
        url = f"{self.endpoint}/indexes/{self.index_name}/docs/search?api-version={self.api_version}"
        
        # Prepare search terms
        search_terms = []
        for key in ['licenseType', 'appType']:
            value = app_data.get(key)
            if value:
                # Handle both list and string types
                terms = value if isinstance(value, list) else [value]
                search_terms.extend([str(v) for v in terms])
        
        search_query = " OR ".join(search_terms) if search_terms else "*"
        
        payload = {
            "search": search_query,
            "top": top,
            "queryType": "simple",
            "searchFields": "content",
            "select": "content"
        }
        
        if filter_expr:
            payload["filter"] = filter_expr
        
        try:
            response = requests.post(url, headers=self._get_headers(), json=payload)
            
            if response.status_code == 200:
                result = response.json()
                if "value" in result:
                    processed_results = []
                    for item in result["value"]:
                        if "content" in item:
                            try:
                                app_data = json.loads(item["content"])
                                
                                # Process details_form if it's a string
                                if isinstance(app_data.get("details_form"), str):
                                    try:
                                        app_data["details_form"] = json.loads(app_data["details_form"])
                                    except json.JSONDecodeError:
                                        self.logger.warning(f"Failed to parse details_form for application {app_data.get('uuid')}")
                                
                                processed_results.append(app_data)
                            except json.JSONDecodeError:
                                self.logger.warning("Failed to parse content field in search result")
                    
                    return processed_results
                return []
            else:
                self.logger.error(f"Search request failed: {response.status_code} - {response.text}")
                return []
                
        except Exception as e:
            self.logger.error(f"Exception during search: {str(e)}")
            return []
    
    def search_similar_applications_with_embeddings(self, embeddings, app_data, filter_expr=None, top=3):
        """
        Enhanced search method using embeddings for similarity search.
        """
        return self.search_similar_applications(app_data, filter_expr, top)