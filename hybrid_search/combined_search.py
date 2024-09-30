# combined_search.py

import pandas as pd
from loguru import logger

from sentence_transformers import CrossEncoder

from semantic_search import SemanticSearchModule
from keyword_search_module import KeywordSearchModule
from util import sqlite_util

class CombinedSearchModule:

    MODEl = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
    
    '''
    * The below function combines the semantic and keyword search 
    * results and re-ranks them using cross encoder.
    '''
    def combined_search_results(self, query : str) -> pd.DataFrame:

        # Initialize Semantic Search and Keyword Search Module
        search_mod = SemanticSearchModule()
        keyword_search_mod = KeywordSearchModule()

        # retrieve the semantic search results. 
        search_results = search_mod.semantic_search_results(query)
        # get the chunk_ids and scores
        semantic_indices = [result['corpus_id'] for result in search_results[0]]
        semantic_scores = [result['score'] for result in search_results[0]]

        # retrieve the keyword search results
        results, scores = keyword_search_mod.search_keyword_results(query)

        kw_chunk_ids = []
        kw_scores = []
        for i in range(results.shape[1]):
            doc, score = results[0, i], scores[0, i]
            kw_chunk_ids.append(doc['id'])
            kw_scores.append(score)

        # retrieve sorted corpus_ids
        sorted_corpus_ids = self.combine_indices(semantic_scores, 
                                                 semantic_indices,
                                                 kw_scores, 
                                                 kw_chunk_ids
                                                 )
        
        logger.info(f"sorted indices is : {sorted_corpus_ids}")

        # retrieve the dataframe from db containing the 
        # search results.
        combined_df = self.get_corpus_details(sorted_corpus_ids)
        # logger.info(f"combined df is : {combined_df['chunk_txt']}")
        
        # list of tuples containing doc and scores are retrieved.
        re_ranked_list = self.rerank_results(query, combined_df['chunk_txt'])[:5]

        # lists to contain the chunks, rank and scores
        doc_list = []
        rank_list = []
        score_list = []
        for rank, (doc, score) in enumerate(re_ranked_list, start=1):
            # print(f"Rank {rank}: {doc} (Score: {score:.4f})")
            doc_list.append(doc)
            rank_list.append(rank)
            score_list.append(score)
        
        # Re-arranging the DataFrame rows based on the new order of 'Name' values in Chun_txt
        df_reordered = combined_df.set_index('chunk_txt').loc[doc_list].reset_index()
        df_reordered['rank'] = pd.Series(rank_list)
        df_reordered['score'] = pd.Series(score_list)        
        logger.info("-------------------------------")
        logger.info(f"re ordered docs list : {df_reordered}")

        return df_reordered
    '''
    * Below function uses a cross encoder model.
    * It compares the query with the sorted chunk text and returns 
    * re-ranked results.
    '''
    def rerank_results(self, query : str, candidate_docs : list) -> list:

        # initialize the cross encoder
        cross_encoder = CrossEncoder(self.MODEl)
        # Initialize a tuple containing the pairing of query to doc
        pairs = [(query, doc) for doc in candidate_docs]

        #predict the scores using cross encoder model.
        scores = cross_encoder.predict(pairs)
        logger.info(f"re ranked scores is : {scores}")

        # now sort them by scores and build a tuple of score to doc
        # by zipping them and limiting to top 5 results.
        reranked_results = sorted(zip(candidate_docs, scores),
                                   key = lambda x : x[1], reverse=True)

        return reranked_results

    def combine_indices(self, semantic_scores, semantic_indices,
                         keyword_scores, keyword_indices, top_k=10):
        scores = {}
    
        for idx, score in zip(semantic_indices, semantic_scores):
            normalized_score = score
            scores[idx] = scores.get(idx, 0) + normalized_score
        
        for idx, score in zip(keyword_indices, keyword_scores):
            normalized_score = self.normalize_keyword_score(score)
            scores[idx] = scores.get(idx, 0) + normalized_score
        
        sorted_indices = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return sorted_indices[:top_k]

    # Define normalization function
    def normalize_keyword_score(self, score, min_score=1, max_score=10):
        # Normalize keyword score to range [0, 1].
        normalized_score =  (score - min_score) / (max_score - min_score)
        return normalized_score
    '''
    * Utility function to retreive dataframe from db by passing
    * chunk ids
    '''
    def get_corpus_details(self, chunk_ids : list) -> pd.DataFrame:
        logger.info(f"sorted chunk idds: {chunk_ids}")
        # Filter the records by the chunk IDs
        try:
            conn = sqlite_util.connect_to_db()
            filtered_df = sqlite_util.load_data_to_dataframe(conn,
                                                        chunk_ids)
        except ConnectionError as ce:
            logger.error( f"Conection error : {ce}" )
        except Exception as e:
            logger.error( f"Exception is : {e}" )
        
        return filtered_df


def test_classes(): 
    csm = CombinedSearchModule()
    csm.combined_search_results("civil war")

if __name__ == "__main__":
    test_classes()