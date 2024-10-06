# qdrantdb_util.py

import torch
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance, Filter, Range
from sentence_transformers import SentenceTransformer
from typing import List, Dict
from bm25s import BM25
from loguru import logger

class QdrantDbUtil:
    
    # Initialize QdrantClient
    client = QdrantClient(host="localhost", port=6333)
    # collection where the data to be stored in qdrant
    COLLECTION_NAME = "dilawar_hybrid_search_collection"

    '''
    * constructor method. Collection gets created if it does not exist.
    '''
    def __init__(self) -> None:
        try:
            # check if the collection exists
            existing_collection = self.client.collection_exists(self.COLLECTION_NAME)
            logger.info(f"existing_collection is : {existing_collection}")
            # create a collection if it does not exist.
            if not existing_collection:
                self.client.create_collection(
                    collection_name=self.COLLECTION_NAME, 
                    vectors_config=VectorParams(size=1024, 
                                                distance=Distance.COSINE)
                )
        except Exception as e:
            logger.error(f"Excption occured : {e}")

    '''
    * function that will store the vectors, associated chunks, key word indexes
    * and document list.
    * we are batching so that we dont perform lot of individual calls that
    * will impact resource utilization.
    '''
    def load_data_to_qdrantdb(self, vector_embeddings : List[torch.Tensor], chunk_list : list,
                               bm25_indexes : List[Dict], document_urls : list):
        # Batch size 
        BATCH_SIZE = 500
        vector_embeddings_size = len(vector_embeddings)
        logger.info(f"vector_embeddings_size  is : {vector_embeddings_size}")

        if vector_embeddings_size < BATCH_SIZE:
            BATCH_SIZE = vector_embeddings_size

        try:
            for i in range(0, len(vector_embeddings), BATCH_SIZE):
                # Batch the embeddings and indexes.
                logger.info(f"inside batch : {i}")
                batch_embeddings = vector_embeddings[i:i + BATCH_SIZE]
                batch_indexes = bm25_indexes[i:i + BATCH_SIZE]
                # This list will contain the collection of rows
                batch_payload = []
                logger.info("moving near 2nd loop")
                # Loop over and build the payload
                for j, (embedding, index) in enumerate(zip(batch_embeddings, batch_indexes)):

                    batch_payload.append(PointStruct(
                        id = i + j,
                        vector = embedding.tolist(), # convert tensor to list
                        payload = {
                            "text" : chunk_list [i+j],
                            "bm25_index" : index,
                            "doc_url" : document_urls[i+j]
                        }
                    )
                    )
                logger.info("after batching")
    
                # Upsert the batch to Qdrant 
                self.client.upsert(
                    collection_name=self.COLLECTION_NAME,
                    points=batch_payload
                )
                logger.info(f"Batch {i} has been processed and loaded to qdrant db")
        except Exception as e:
            logger.error(f"Exception occured in load to dqdrant db : {e}")

    '''
    * Function to retrieve the semantic results from the collection. returns dict in below format.
        {
        "id": 12345,
        "score": 0.95,
         "payload": {
             "text": "This is a sample text from the document.",
             "url": "http://example.com/document1",
             "bm25_index": {"idf": 1.5, "tf": 2}
             }
        }
    '''    
    def retrieve_semantic_results(self, query_embedding : torch.Tensor, top_k = 5) -> list:
        # retrieve the results
        try:
                search_results = self.client.search(
                collection_name=self.COLLECTION_NAME,
                query_vector=query_embedding.tolist(),
                limit=5
            )
        except Exception as e:            
            logger.error( f"error occured : {e}" )
        
        logger.info(f"semantic point struct is : {search_results}")
        return search_results
        
    '''
    * Search function that matches the query against the built keyword index.
    '''
    def keyword_search(self, query: str, min_bm25_score=0):
        # No vector embedding needed for pure keyword search
        embedder = SentenceTransformer("BAAI/bge-large-zh-v1.5")
        vector_embed = embedder.encode(query, convert_to_tensor=True)
    
         # Tokenize the query for keyword search
        query_terms = query.split()
        logger.info(f"query_terms results is : {query_terms}")

        # Step 1: Perform the  vector search in Qdrant
        try:
            search_results = self.client.search(
                collection_name=self.COLLECTION_NAME,
                query_vector=vector_embed.tolist(),  # Convert tensor to list for Qdrant
                limit=10  # Get top 5 results based on semantic relevance
            )
        except Exception as e:
            logger.error(f"exception in keyword search : {e}")

        logger.info(f"Result is: {search_results}")
        return search_results

    """
    * Perform a combined search based on semantic (vector) search and keyword (BM25) search.
    * Returns a list of ranked results combining semantic and keyword scores.
    """   
    def combined_hybrid_search(self, query: str, 
                        vector_query: torch.Tensor,
                        min_bm25_score = 0):

        # Tokenize the query for keyword search
        query_terms = query.split()
        logger.info(f"query_terms results is : {query_terms}")

        # Step 1: Perform the semantic vector search in Qdrant
        try:
            semantic_results = self.client.search(
                collection_name=self.COLLECTION_NAME,
                query_vector=vector_query.tolist(),  # Convert tensor to list for Qdrant
                limit=5  # Get top 5 results based on semantic relevance
            )
        except Exception as e:
            logger.error(f"exception in chs : {e}")

        # Process and combine semantic and keyword search results
        combined_results = []
        for result in semantic_results:            
            # Use dot notation to access properties
            chunk_id = result.id
            document_url = result.payload.get('doc_url', 'No URL available')           
            document_text = result.payload.get('text', 'No text available')
            
            # Extract the BM25 index from the result's payload
            bm25_index = result.payload.get('bm25_index', {}).get('terms', {})

            # Calculate the BM25 score for the document based on query terms
            bm25_score = sum(bm25_index.get(term, {}).get('tf', 0) * bm25_index.get(term, {}).get('idf', 0)
                            for term in query_terms)
            
            # Get the semantic vector score
            vector_score = result.score  # Use dot notation for score
            
            # Step 4: Combine the two scores (adjust weights as needed)
            combined_score = 0.7 * vector_score + 0.3 * bm25_score  # Recommended weights
            combined_results.append({
                "id": chunk_id,
                "doc_url": document_url,
                "text": document_text,
                "semantic_score": vector_score,
                "bm25_score": bm25_score,
                "combined_score": combined_score
            })

        # Sort the combined results by combined score
        combined_results = sorted(combined_results, key=lambda x: x['combined_score'], reverse=True)

        return combined_results

    
