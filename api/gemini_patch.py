from typing import Sequence, List
from copy import deepcopy
from tqdm import tqdm
import logging
import adalflow as adal
from adalflow.core.types import Document
from adalflow.core.component import DataComponent
import google.generativeai as genai

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GeminiDocumentProcessor(DataComponent):
    """
    Process documents for Gemini embeddings by processing one document at a time.
    This is needed because adalflow's GoogleGenAIClient doesn't support the EMBEDDER model type.
    """
    def __init__(self, api_key: str, model: str = "gemini-embedding-exp-03-07", dimensions: int = 256) -> None:
        super().__init__()
        self.model = model
        self.dimensions = dimensions
        genai.configure(api_key=api_key)
        
    def __call__(self, documents: Sequence[Document]) -> Sequence[Document]:
        output = deepcopy(documents)
        logger.info(f"Processing {len(output)} documents for Gemini embeddings")
        
        batch_size = 10  # Adjust based on API limits
        for i in range(0, len(output), batch_size):
            batch = output[i:i+batch_size]
            self._process_batch(batch, start_idx=i)
            
        return output
    
    def _process_batch(self, documents: List[Document], start_idx: int) -> None:
        """Process a batch of documents to get embeddings."""
        for j, doc in enumerate(tqdm(documents, desc=f"Embedding documents batch {start_idx//10 + 1}")):
            try:
                embedding_result = genai.embed_content(
                    model=self.model,
                    content=doc.text,
                    task_type="SEMANTIC_SIMILARITY",
                    dimensions=self.dimensions
                )
                
                if embedding_result and hasattr(embedding_result, 'embedding'):
                    documents[j].vector = embedding_result.embedding
                else:
                    logger.warning(f"Failed to get embedding for document {start_idx + j}")
            except Exception as e:
                logger.error(f"Error processing document {start_idx + j}: {e}")
