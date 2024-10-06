# hybrid_search_module.py

import pandas as pd
from loguru import logger


from util import file_util as file_util
from keyword_search_module import KeywordSearchModule
from semantic_search import SemanticSearchModule
from combined_search import CombinedSearchModule

class HybridSearchModule:

    '''
    *   Function that takes in the query string as input.
    *   retrieves re-ranked search results using cross encoder.
    '''
    def display_hybrid_search_results(self, query_str : str) -> pd.DataFrame:
 
        cs = CombinedSearchModule()
        # Compare the query against the search candidate documents of 
        # semantic search and keyword search. Cross encoder is used to
        # re-rank the results.
        try:
            search_results = cs.combined_search_results(query_str)
            # dataframe that holds the search results.
            combined_df = pd.DataFrame(search_results)
        except Exception as e:
                logger.error(f"Exception occured in HybridSearchModule.display_hybrid_search_results {e}")

        return combined_df

    '''
    *   Function that takes in the query string as input.
    *   retrieves semantic search results. 
    '''
    def display_semantic_search_results(self, query_str : str) -> pd.DataFrame:
 
        ssm = SemanticSearchModule()

        try:
            # Compare the query with search corpus using cosine similarities method
            search_results = ssm.semantic_search_results(query_str)
            semantic_df = pd.DataFrame(search_results)
        except Exception as e:
            logger.error(f" Exception occured in HybridSearchModule.display_semantic_search_results : {e}")

        return semantic_df

    '''
    *   Function that takes in the query string as input.
    *   retrieves keyword search results. 
    '''
    def display_keyword_search_results(self, query_str : str) -> pd.DataFrame:
        
        # Initialize the keyword search module
        ksm = KeywordSearchModule()

        # call the keyword search results
        try:
            search_results = ksm.search_keyword_results(query_str)
            # pass the search results and retrieves a dataframe 
            keyword_results_df = pd.DataFrame(search_results)
            logger.info(f"keyword results column: {keyword_results_df.columns}")
        except Exception as e:
            logger.error(f"Exception occured in HybridSearchModule.display_keyword_search_results : {e}")

        return keyword_results_df
    

'''
Below function for unit testing
'''

if __name__ == "__main__":

    hsm = HybridSearchModule()
    df = hsm.display_keyword_search_results("loss function in machine learning")
    logger.debug( f"df columns : {df.head()}" )

    hsm1 = HybridSearchModule()
    hyb_df = hsm1.display_hybrid_search_results("friend of man")
    logger.debug( f"{hyb_df.head()}" )
