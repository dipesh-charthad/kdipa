import json
import logging
import hashlib
from functools import lru_cache
from openai import AzureOpenAI
from config import Config
from prompts import get_investment_analysis_prompt

class AzureOpenAIClient:
    def __init__(self):
        """Initialize Azure OpenAI client with configuration from Config."""
        try:
            Config.validate_config()
            
            # Initialize logging
            self.logger = logging.getLogger(__name__)
            
            # Initialize OpenAI client
            self.client = AzureOpenAI(
                azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
                api_key=Config.AZURE_OPENAI_API_KEY,
                api_version=Config.AZURE_OPENAI_VERSION
            )
            self.logger.info("Azure OpenAI client initialized successfully")
        except ValueError as e:
            self.logger.error(f"Configuration error: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Failed to initialize Azure OpenAI client: {str(e)}")
            raise

    def get_completion(self, prompt, max_tokens=2000, temperature=0.0):
        """
        Get completion from Azure OpenAI
        """
        try:
            response = self.client.chat.completions.create(
                model=Config.AZURE_OPENAI_DEPLOYMENT_NAME,
                messages=[
                    {"role": "system", "content": "You are an AI assistant analyzing investment license applications."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
           
            return response.choices[0].message.content.strip()
       
        except Exception as e:
            self.logger.error(f"OpenAI request error: {str(e)}")
            raise
    
    def analyze_application(self, new_application, similar_applications):
        """Analyze the application against similar applications"""
        try:
            # Convert similar applications to a more compact indexed dataset
            indexed_dataset = {
                "total_applications": len(similar_applications),
                "applications": similar_applications
            }
            
            prompt = get_investment_analysis_prompt(new_application, indexed_dataset)
            
            response_text = self.get_completion(prompt, max_tokens=2000)
            
            # Remove Markdown code block markers if present
            if response_text.startswith('```json'):
                response_text = response_text.replace('```json', '').replace('```', '').strip()
            
            try:
                return json.loads(response_text)
            except json.JSONDecodeError:
                self.logger.error(f"Failed to parse response: {response_text}")
                return {
                    "Error": "Failed to parse analysis response",
                    "Raw response": response_text
                }
        
        except Exception as e:
            self.logger.error(f"Error in analyze_application: {str(e)}")
            raise