# semantic_search.py

import os
import torch
from sentence_transformers import SentenceTransformer, util
from loguru import logger

from util import file_util
from util.qdrantdb_util import QdrantDbUtil

class SemanticSearchModule:

    # Load a pre-trained model for asymmetric search
    MODEL =  "msmarco-distilbert-base-v4"
    MODEL_1 = "BAAI/bge-large-zh-v1.5"
    MODEL_2 = "paraphrase-MiniLM-L6-v2"
    embedder = SentenceTransformer(MODEL_1)

    '''
    *   Function that takes in the query string as input.
    *   retrieves search results 
    '''
    def semantic_search_results(self, query_str : str) -> list:

        # Create the query embedding
        query_embeddings = self.embedder.encode(query_str, convert_to_tensor=True)

        try:
            # Initialize the QDrantDbUtil class 
            qdrant_db_util = QdrantDbUtil()

            # Compare the query with search corpus using cosine similarities method
            search_results = qdrant_db_util.retrieve_semantic_results(query_embeddings)
            
        except Exception as e:
            logger.error(f"exception occured in SemanticSearchModule.semantic_search_results : {e}")

        #process the search results and return as list of dict.
        semantic_results = []

        for index, result in enumerate(search_results, start=1):
            # get the score 
            score = result.score
            # get the chunk id
            chunk_id = result.id
            # get the chunk text
            document_text = result.payload.get('text', 'No text available')
            # get the document url
            document_url = result.payload.get('doc_url', 'No URL available')

            semantic_results.append({
                "rank" : index, 
                "score": score,
                "chunk_id" : chunk_id, 
                "chunk_text": document_text,  
                "doc_url": document_url
            })

        return semantic_results
