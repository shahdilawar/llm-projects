# combined_search.py

import pandas as pd
from loguru import logger

from sentence_transformers import SentenceTransformer, CrossEncoder

from util.qdrantdb_util import QdrantDbUtil

class CombinedSearchModule:

    MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
    # Load a pre-trained model for asymmetric search
    MODEL_2 = "BAAI/bge-large-zh-v1.5"
    embedder = SentenceTransformer(MODEL_2)
    
    '''
    * The below function combines the semantic and keyword search 
    * results and re-ranks them using cross encoder.
    '''
    def combined_search_results(self, query : str) -> list:
        # Initialize QdrantDbClient
        qdrant_db_util = QdrantDbUtil()
        logger.info(f"qdrant_db_util is : {qdrant_db_util}")
        query_vector = self.embedder.encode(query, convert_to_tensor=True)

        # retrieve the combined search results. 
        try:
            combined_search_results = qdrant_db_util.combined_hybrid_search(query,
                                                                             query_vector)
            logger.info(f"combined_search_results is : {combined_search_results}")
        except Exception as e:
            logger.error(f"Exceptio occured : {e}")
        
        # list of dictionary containing cgunk text and scores are retrieved.
        final_re_ranked_list = self.rerank_results(query, combined_search_results)[:5]

        return final_re_ranked_list
    '''
    * Below function uses a cross encoder model.
    * It compares the query with the sorted chunk text and returns 
    * re-ranked results.
    '''
    def rerank_results(self, query: str, candidate_docs: list) -> list:
        # Initialize the cross encoder
        cross_encoder = CrossEncoder(self.MODEL)
        
        # Initialize a tuple containing the pairing of query to doc
        pairs = [(query, doc['text']) for doc in candidate_docs]  # Use doc['text'] to get the text for reranking

        # Predict the scores using the cross encoder model
        scores = cross_encoder.predict(pairs)

        # Now sort them by scores and build a tuple of score to doc
        reranked_results = sorted(zip(candidate_docs, scores), key=lambda x: x[1], reverse=True)

        # Create a final list using list comprehension with URL and ID included
        final_results = [
            {
                "rank" : i, 
                "chunk_id": doc['id'],
                "score": score, 
                "chunk_text": doc['text'], 
                "doc_url": doc['doc_url']
            }
            for i, (doc, score) in enumerate(reranked_results, start=1)
        ]

        return final_results

'''
Below function for unit testing
'''
def test_classes(): 
    csm = CombinedSearchModule()
    csm.combined_search_results("civil war")

if __name__ == "__main__":
    test_classes()