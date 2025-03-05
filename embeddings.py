import os
import re
import json
import logging
from dotenv import load_dotenv

# NLTK Download First
import nltk
nltk.download('punkt', quiet=True)

# PDF Parsing Libraries
import PyPDF2
import pdfplumber

# Text Preprocessing Libraries
from nltk.tokenize import sent_tokenize

# Azure OpenAI Embedding
from openai import AzureOpenAI

class BulkPDFProcessor:
    def __init__(self, 
                 chunk_size: int = 200, 
                 chunk_overlap: int = 50,
                 azure_deployment: str = "text-embedding-3-large"):
        """
        Initialize BulkPDFProcessor for processing multiple PDF documents with Azure OpenAI embeddings.
        
        :param chunk_size: Number of tokens/words per chunk
        :param chunk_overlap: Number of tokens/words to overlap between chunks
        :param azure_deployment: Azure deployment name for embedding model
        """
        # Load environment variables
        load_dotenv()
        
        # Configure logging
        logging.basicConfig(level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Ensure NLTK resources are downloaded
        try:
            sent_tokenize("Test sentence.")
        except LookupError:
            print("Downloading NLTK resources...")
            nltk.download('punkt', download_dir=nltk.data.path[0])
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Retrieve Azure OpenAI credentials from environment variables
        azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
        azure_key = os.getenv('AZURE_OPENAI_API_KEY')
        
        if not azure_endpoint or not azure_key:
            raise ValueError("Azure OpenAI endpoint and key must be set in environment variables")
        
        # Initialize Azure OpenAI client
        self.azure_client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=azure_key,
            api_version="2024-02-01"  # Use the latest API version
        )
        self.azure_deployment = azure_deployment
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from PDF using multiple methods for robustness.
        
        :param pdf_path: Path to the PDF file
        :return: Extracted text from the PDF
        """
        extracted_text = ""
        
        # Try pdfplumber first (better for complex layouts)
        try:
            with pdfplumber.open(pdf_path) as pdf:
                extracted_text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
        except Exception as e:
            self.logger.warning(f"pdfplumber extraction failed: {e}")
            
            # Fallback to PyPDF2 if pdfplumber fails
            try:
                with open(pdf_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    extracted_text = "\n".join(
                        page.extract_text() for page in reader.pages if page.extract_text()
                    )
            except Exception as e:
                self.logger.error(f"PyPDF2 extraction failed: {e}")
                raise ValueError(f"Could not extract text from PDF: {pdf_path}")
        
        return extracted_text
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text by cleaning and normalizing.
        
        :param text: Input text
        :return: Cleaned text
        """
        # Remove extra whitespaces and newlines
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove headers, footers, page numbers (basic approach)
        text = re.sub(r'\n*\d+\n*', '', text)
        
        # Optional: Convert to lowercase (comment out if case sensitivity matters)
        text = text.lower()
        
        return text
    
    def chunk_text(self, text: str) -> list:
        """
        Chunk text into smaller segments with overlapping.
        
        :param text: Input text to chunk
        :return: List of text chunks
        """
        # Preprocess text first
        text = self.preprocess_text(text)
        
        # Tokenize into sentences
        try:
            sentences = sent_tokenize(text)
        except LookupError:
            # Fallback method if tokenization fails
            sentences = text.split('. ')
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            # Check if adding this sentence would exceed chunk size
            if current_length + len(sentence.split()) > self.chunk_size:
                # If chunk is not empty, add it
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                
                # Start new chunk with overlap
                current_chunk = current_chunk[-self.chunk_overlap:]
            
            current_chunk.append(sentence)
            current_length = len(' '.join(current_chunk).split())
        
        # Add last chunk if not empty
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def generate_embeddings(self, chunks: list) -> list:
        """
        Generate embeddings for text chunks using Azure OpenAI.
        
        :param chunks: List of text chunks
        :return: List of embeddings
        """
        embeddings = []
        for chunk in chunks:
            try:
                response = self.azure_client.embeddings.create(
                    input=chunk, 
                    model=self.azure_deployment
                )
                embeddings.append(response.data[0].embedding)
            except Exception as e:
                self.logger.error(f"Embedding generation failed for chunk: {e}")
                embeddings.append(None)
        
        return embeddings
    
    def process_pdf_folder(self, folder_path: str, output_folder: str = None):
        """
        Process all PDF files in a specified folder.
        
        :param folder_path: Path to folder containing PDF files
        :param output_folder: Path to save processed JSON files (optional)
        """
        # Validate input folder
        if not os.path.isdir(folder_path):
            raise ValueError(f"Invalid folder path: {folder_path}")
        
        # Set default output folder if not specified
        if output_folder is None:
            output_folder = os.path.join(folder_path, 'processed_pdfs')
        
        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
        # Find all PDF files
        pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
        
        # Process each PDF
        for pdf_filename in pdf_files:
            try:
                # Full path to PDF
                pdf_path = os.path.join(folder_path, pdf_filename)
                
                # Extract text
                text = self.extract_text_from_pdf(pdf_path)
                
                # Chunk text
                chunks = self.chunk_text(text)
                
                # Generate embeddings
                embeddings = self.generate_embeddings(chunks)
                
                # Prepare processed chunks
                processed_chunks = []
                for chunk, embedding in zip(chunks, embeddings):
                    if embedding is not None:
                        processed_chunks.append({
                            'text': chunk,
                            'embedding': embedding
                        })
                
                # Prepare output data
                output_data = {
                    'document_name': pdf_filename,
                    'total_chunks': len(processed_chunks),
                    'chunks': processed_chunks
                }
                
                # Output JSON filename
                output_filename = f"{os.path.splitext(pdf_filename)[0]}_chunks.json"
                output_path = os.path.join(output_folder, output_filename)
                
                # Save to JSON
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)
                
                self.logger.info(f"Processed: {pdf_filename} - {len(processed_chunks)} chunks")
            
            except Exception as e:
                self.logger.error(f"Error processing {pdf_filename}: {e}")

# Main execution
if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # Get PDF folder path from environment variable or use default
    PDF_FOLDER_PATH = os.getenv('PDF_FOLDER_PATH', r"C:\Projects\KDIPA\KDIPA Documents")
    
    # Initialize processor
    processor = BulkPDFProcessor(
        chunk_size=200,   # Adjust chunk size as needed
        chunk_overlap=50,  # Adjust overlap as needed
        azure_deployment="text-embedding-3-large"
    )
    
    # Process all PDFs in the folder
    processor.process_pdf_folder(PDF_FOLDER_PATH)