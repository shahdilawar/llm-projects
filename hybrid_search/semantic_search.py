import os
import torch
from sentence_transformers import SentenceTransformer, util
from loguru import logger

from util import file_util

class SemanticSearchModule:

    # Load a pre-trained model for asymmetric search
    MODEL =  "msmarco-distilbert-base-v4"
    MODEL_1 = "BAAI/bge-large-zh-v1.5"
    MODEL_2 = "paraphrase-MiniLM-L6-v2"
    embedder = SentenceTransformer(MODEL_1)

    #output directory paths
    output_dir = "./llm-projects/hybrid_search"

    '''
    *   Function that takes in the query string as input.
    *   retrieves search results 
    '''
    def semantic_search_results(self, query_str : str) -> list:
        # Create data directory if it doesn't exist
        os.makedirs(os.path.join(self.output_dir, "pickle_data"), exist_ok=True)
        data_file = os.path.join(self.output_dir, "pickle_data", "embeddings.pkl")

        embeddings_list = file_util.read_from_pickle(data_file)
        corpus_embeddings = torch.tensor(embeddings_list)
        
        logger.debug( f"{type(corpus_embeddings[0])}" )

        # Create the query embedding
        query_embeddings = self.embedder.encode(query_str, convert_to_tensor=True)

        # Compare the query with search corpus using cosine similarities method
        search_results = util.semantic_search( query_embeddings,
                                            corpus_embeddings, top_k=5)
        return search_results
