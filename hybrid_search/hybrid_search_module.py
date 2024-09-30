# hybrid_search_module.py

import pandas as pd
import sqlite3
from loguru import logger


from util import file_util as file_util
from util import sqlite_util as sqlite_util
from keyword_search_module import KeywordSearchModule
from semantic_search import SemanticSearchModule
from combined_search import CombinedSearchModule

class HybridSearchModule:

    '''
    *   Function that takes in the query string as input.
    *   retrieves search corpus embeddings 
    '''
    def display_hybrid_search_results(self, query_str : str) -> pd.DataFrame:
 
        cs = CombinedSearchModule()
        # Compare the query with search corpus using cosine similarities method
        search_results = cs.combined_search_results(query_str)

        return search_results


    '''
    *   Function that takes in the query string as input.
    *   retrieves search corpus embeddings 
    '''
    def display_semantic_search_results(self, query_str : str) -> pd.DataFrame:
 
        ssm = SemanticSearchModule()
        # Compare the query with search corpus using cosine similarities method
        search_results = ssm.semantic_search_results(query_str)

        # Retrieve search results and populate rank, score, and chunk_id lists
        rank_list = []
        score_list = []
        chunk_id_list = []

        for index, result in enumerate(search_results[0]):
            rank_list.append(index + 1)
            score_list.append(result['score'])
            chunk_id_list.append(result['corpus_id'])
        
        logger.debug( f"{chunk_id_list}" )

        # Filter the records by the chunk IDs
         # Filter the records by the chunk IDs
        filtered_df = self.get_corpus_details(chunk_id_list)  
        
        filtered_df['rank'] = pd.Series(rank_list,
                                        index=filtered_df.index)
        filtered_df['score'] = pd.Series(score_list,
                                        index=filtered_df.index)

        # Reset the index to avoid issues with missing chunk_ids
        filtered_df.reset_index(drop=True, inplace=True)

        return filtered_df

    '''
    *   Function that takes in the query string as input.
    *   retrieves search corpus embeddings 
    '''
    def display_keyword_search_results(self, query_str : str) -> pd.DataFrame:
        
        # Initialize the keyword search module
        ksm = KeywordSearchModule()

        # call the keyword search results
        new_results, new_scores = ksm.search_keyword_results(query_str)

        # Retrieve search results and populate rank, score, and chunk_id lists
        rank_list = []
        score_list = []
        chunk_id_list = []

        for i in range(new_results.shape[1]):
            doc, score = new_results[0, i], new_scores[0, i]
            rank_list.append(i + 1)
            score_list.append(score)
            chunk_id_list.append(doc['id'])
        
        logger.debug( f"{chunk_id_list}" )

        # Filter the records by the chunk IDs
        filtered_df = self.get_corpus_details(chunk_id_list)  

        filtered_df['rank'] = pd.Series(rank_list,
                                        index=filtered_df.index)
        filtered_df['score'] = pd.Series(score_list,
                                        index=filtered_df.index)

        # Reset the index to avoid issues with missing chunk_ids
        filtered_df.reset_index(drop=True, inplace=True)

        return filtered_df
    
    '''
    * Function to display combined search results and re-rank them.
    '''
    '''

    * Utility function to retreive dataframe from db by passing
    * chunk ids
    '''
    def get_corpus_details(self, chunk_ids : list) -> pd.DataFrame:
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


'''
Below function for unit testing
'''
if __name__ == "__main__":

    hsm = HybridSearchModule()
    df = hsm.display_hybrid_search_results("loss functions in machine learning")
    logger.debug( f"{df.head()}" )
