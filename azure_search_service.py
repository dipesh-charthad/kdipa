import json
import logging
import requests
import numpy as np
from config import Config
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

class AzureSearchClient:
    def __init__(self):
        self.endpoint = Config.AZURE_SEARCH_ENDPOINT
        self.index_name = Config.AZURE_SEARCH_INDEX_NAME
        self.api_key = Config.AZURE_SEARCH_API_KEY
        self.api_version = "2023-07-01-Preview"  # API version for vector and semantic search
        
        # Logging configuration
        self.logger = logging.getLogger(__name__)
        
        # Create a session with connection pooling and retry logic
        self.session = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[500, 502, 503, 504]
        )
        self.session.mount('https://', HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=10))
        
        self.logger.info("Azure Search Configuration Initialized")
        
    def _get_headers(self):
        """Return the headers needed for Azure Search API requests."""
        return {
            "Content-Type": "application/json",
            "api-key": self.api_key
        }
    
    def _build_search_query(self, app_data, essential_fields):
        """Extract search terms from application data and build a search query."""
        search_terms = []
        
        for field in essential_fields:
            value = app_data.get(field)
            if value:
                # Handle both list and string types
                terms = value if isinstance(value, list) else [value]
                search_terms.extend([str(v) for v in terms])
        
        return " OR ".join(search_terms) if search_terms else "*"
    
    def _create_search_payload(self, search_query, top, filter_expr=None):
        """Create the payload for a basic search request."""
        payload = {
            "search": search_query,
            "top": top,
            "queryType": "simple",
            "searchFields": "content",
            "select": "id,metadata_spo_item_name,metadata_spo_item_last_modified,content"
        }
        
        if filter_expr:
            payload["filter"] = filter_expr
            
        return payload
    
    def _process_search_result(self, item):
        """Process a single search result item and return application data."""
        if "content" not in item:
            return None
            
        try:
            app_data = json.loads(item["content"])
            
            # Add metadata from search result
            app_data["id"] = item.get("id")
            app_data["item_name"] = item.get("metadata_spo_item_name")
            app_data["last_modified"] = item.get("metadata_spo_item_last_modified")
            
            # Process details_form if it's a string
            if isinstance(app_data.get("details_form"), str):
                try:
                    app_data["details_form"] = json.loads(app_data["details_form"])
                except json.JSONDecodeError:
                    self.logger.warning(f"Failed to parse details_form for application {app_data.get('uuid')}")
            
            return app_data
        except json.JSONDecodeError:
            self.logger.warning("Failed to parse content field in search result")
            return None
    
    def _process_search_results(self, result):
        """Process all search results and return processed applications."""
        processed_results = []
        
        if "value" not in result:
            return processed_results
            
        for item in result["value"]:
            app_data = self._process_search_result(item)
            if app_data:
                processed_results.append(app_data)
                
        return processed_results
    
    def _execute_search_request(self, url, payload):
        """Execute a search request and return the processed results."""
        try:
            response = self.session.post(url, headers=self._get_headers(), json=payload)
            
            if response.status_code == 200:
                return self._process_search_results(response.json())
            else:
                self.logger.error(f"Search request failed: {response.status_code} - {response.text}")
                return []
                
        except Exception as e:
            self.logger.error(f"Exception during search: {str(e)}")
            return []
    
    def search_similar_applications(self, app_data, filter_expr=None, top=3):
        """
        Search for similar applications in the Azure AI Search index using keyword-based search.
        This method serves as a fallback for the hybrid search.
        """
        url = f"{self.endpoint}/indexes/{self.index_name}/docs/search?api-version={self.api_version}"
        
        # Prepare search terms using essential fields
        essential_fields = [
            "companyName", "companyOrigin", "companyCity", "companyOutput",
            "contributionAmount", "totalCapitalAmount", "shareholderCompanyPartnerName",
            "shareholderNationality", "valueOfEquityOrShares", "percentageOfEquityOrShares",
            "numberOfEquityOrShares", "totalInvestmentValue",
        ]
        
        # Build search query
        search_query = self._build_search_query(app_data, essential_fields)
        
        # Create search payload
        payload = self._create_search_payload(search_query, top, filter_expr)
        
        # Execute search and return results
        return self._execute_search_request(url, payload)
    
    def _validate_hybrid_search_params(self, query_text, embeddings, filter_expr, top, app_data):
        """Validate parameters for hybrid search."""
        if query_text is not None and not isinstance(query_text, str):
            self.logger.error("query_text must be a string")
            raise TypeError("query_text must be a string")
            
        if embeddings is not None and not isinstance(embeddings, list):
            self.logger.error("embeddings must be a list")
            raise TypeError("embeddings must be a list")
            
        if filter_expr is not None and not isinstance(filter_expr, str):
            self.logger.error("filter_expr must be a string")
            raise TypeError("filter_expr must be a string")
            
        if not isinstance(top, int) or top <= 0:
            self.logger.error("top must be a positive integer")
            raise ValueError("top must be a positive integer")
            
        if app_data is not None and not isinstance(app_data, dict):
            self.logger.error("app_data must be a dictionary")
            raise TypeError("app_data must be a dictionary")
    
    def _create_hybrid_search_payload(self, query_text, top):
        """Create base payload for hybrid search."""
        # Truncate long query text
        search_text = query_text
        if search_text and len(search_text) > 5000:
            self.logger.warning("Query text too large, truncating to 5000 characters")
            search_text = search_text[:5000]
        
        return {
            "search": search_text or "*",
            "top": top,
            "queryType": "semantic",
            "semanticConfiguration": "aiml-semanticconfig",
            "select": "id,metadata_spo_item_name,metadata_spo_item_last_modified,content",
            "queryLanguage": "en-us",
            "captions": "extractive",
            "answers": "extractive",
            "searchFields": "content"
        }
    
    def _add_vector_search(self, payload, embeddings, top):
        """Add vector search to payload if valid embeddings are provided."""
        valid_embeddings = []
        if embeddings:
            valid_embeddings = [e for e in embeddings if e is not None and len(e) > 0]
            
        if not valid_embeddings:
            return
            
        try:
            # Use numpy for efficient array operations
            embedding_arrays = np.array(valid_embeddings)
            avg_embedding = np.mean(embedding_arrays, axis=0).tolist()
            
            # Add vector search to payload
            payload["vectors"] = [
                {
                    "value": avg_embedding,
                    "fields": "vectorcontent",
                    "k": top
                }
            ]
            
            # Use proper string instead of f-string without parameters
            self.logger.info("Performing hybrid search with vector and semantic components")
        except Exception as e:
            error_msg = "Error processing embeddings: {}. Falling back to semantic-only search.".format(str(e))
            self.logger.error(error_msg)
    
    def _fallback_to_basic_search(self, filter_expr, top, app_data):
        """Fall back to basic search if hybrid search fails."""
        if app_data:
            self.logger.info("Falling back to regular search")
            return self.search_similar_applications(app_data, filter_expr, top)
        return []
    
    def hybrid_search(self, query_text, embeddings=None, filter_expr=None, top=3, app_data=None):
        """
        Comprehensive search that combines semantic search and vector search.
        Falls back to basic search if hybrid search fails.
        
        Args:
            query_text: Natural language query text
            embeddings: Optional embeddings for vector search
            filter_expr: Optional filter expression
            top: Number of results to return
            app_data: Optional application data dictionary for fallback search
            
        Returns:
            List of matching applications with relevance scores

        Raises:
            TypeError: If parameters are of incorrect type
            ValueError: If parameters have invalid values
        """
        # Validate parameters
        self._validate_hybrid_search_params(query_text, embeddings, filter_expr, top, app_data)
        
        # Prepare search URL
        url = f"{self.endpoint}/indexes/{self.index_name}/docs/search?api-version={self.api_version}"
        
        # Create base payload
        payload = self._create_hybrid_search_payload(query_text, top)
        
        # Add vector search components if embeddings are provided
        self._add_vector_search(payload, embeddings, top)
        
        # Add filter if provided
        if filter_expr:
            payload["filter"] = filter_expr
        
        try:
            # Execute search
            response = self.session.post(url, headers=self._get_headers(), json=payload)
            
            if response.status_code == 200:
                result = response.json()
                processed_results = self._process_search_results(result)
                self.logger.info("Hybrid search returned {} results".format(len(processed_results)))
                return processed_results
            else:
                self.logger.error(f"Hybrid search request failed: {response.status_code} - {response.text}")
                return self._fallback_to_basic_search(filter_expr, top, app_data)
                
        except Exception as e:
            self.logger.error(f"Exception during hybrid search: {str(e)}")
            return self._fallback_to_basic_search(filter_expr, top, app_data)