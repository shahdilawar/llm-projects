import bm25s
# from rank_bm25 import BM25Okapi
import pandas as pd
import Stemmer  # optional: for stemming
from loguru import logger

from util import file_util

class KeywordSearchModule:
    # Create your corpus here
    corpus = []
    stemmer = Stemmer.Stemmer("english")

    # optional: create a stemmer

    def retrieve_corpus(self, chunks : list):
        # # Iterate through folder structure and build the brute force chunks.
        # file_list = file_util.iterate_folder()
        # for file_obj in file_list:
        #     #retrieve brute force chunks.
        #     self.corpus.extend(file_util.convert_to_chunks(file_obj))
        self.corpus = chunks
    
    def build_index(self) -> bm25s.BM25:

        # Tokenize the corpus and only keep the ids (faster and saves memory)
        corpus_tokens = bm25s.tokenize(self.corpus, stopwords="en",
                                        stemmer=self.stemmer)
        
        # Create the BM25 model and index the corpus
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens)
        retriever.save("./llm-projects/pickle_data/keyword_search_index.pkl",
                                                             corpus=self.corpus)
        return retriever

    def search_keyword_results(self, query : str) -> tuple:

        #load the index
        import bm25s
        reloaded_retriever = bm25s.BM25.load("./llm-projects/pickle_data/keyword_search_index.pkl",
                                              load_corpus=True)
        # Query the corpus
        query_tokens = bm25s.tokenize(query, stemmer=self.stemmer)

        # Get top-k results as a tuple of (doc ids, scores). 
        # Both are arrays of shape (n_queries, k)
        new_results, new_scores = reloaded_retriever.retrieve(query_tokens, k=5)
        
        return new_results, new_scores

'''
Below function is for unit testing.
'''
def test_classes():
    ksm = KeywordSearchModule()
    
    #query 1
    query_1  = "machine learning loss function"
    logger.info (f" Query 1 is : {query_1}")
    new_results, new_scores = ksm.search_keyword_results(query_1)
    logger.info( "\nNew query results:" )
    for i in range(new_results.shape[1]):
        doc, score = new_results[0, i], new_scores[0, i]
        logger.info( f"Rank {i+1} (score: {score:.2f}): {doc['id']}" )

if __name__ == "__main__":
    test_classes()