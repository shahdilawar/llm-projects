import pandas as pd
from rank_bm25 import BM25Okapi
from collections import Counter
from nltk.stem import PorterStemmer
from loguru import logger

from util.qdrantdb_util import QdrantDbUtil

class KeywordSearchModule:
    def __init__(self):
        self.corpus = []
        self.stemmer = PorterStemmer()
        self.idf_dict = {}

    # Retrieve and set the corpus (chunks of text)
    def retrieve_corpus(self, chunks):
        self.corpus = chunks
    
    # Build the BM25 index using tokenized corpus
    def build_index(self):
        # Tokenize the corpus
        tokenized_corpus = [self.tokenize(doc) for doc in self.corpus]
        
        # Create the BM25 model and index the corpus
        retriever = BM25Okapi(tokenized_corpus)

        # Create the IDF dictionary directly from the tokenized corpus
        unique_terms = set(token for doc in tokenized_corpus for token in doc)
        self.idf_dict = {term: retriever.idf[term] for term in unique_terms}

        logger.info(f"IDF values: {self.idf_dict}")

        return retriever

    # Tokenize each document
    def tokenize(self, document):
        tokens = document.split()
        tokens = [token.lower() for token in tokens]  # Convert to lowercase
        tokens = [self.stemmer.stem(token) for token in tokens]  # Apply stemming
        return tokens

    # Prepare BM25 data with term frequency and IDF values
    def prepare_bm25_data(self, retriever):
        bm25_data_list = []
        
        for idx, doc in enumerate(self.corpus):
            tokenized_doc = self.tokenize(doc)
            term_freqs = Counter(tokenized_doc)  # Term frequency using Counter

            doc_bm25_data = {
                "doc_length": float(len(tokenized_doc)),
                "terms": {
                    token: {
                        "tf": float(term_freqs[token]),  # Term frequency
                        "idf": float(self.idf_dict.get(token, 1.0))  # IDF from the dict, default to 1.0 if not found
                    }
                    for token in tokenized_doc
                    if term_freqs[token] > 0
                }
            }
            bm25_data_list.append(doc_bm25_data)

        return bm25_data_list


    # Function to return the keyword search results based on the query
    def search_keyword_results(self, query: str) -> list:
        # Tokenize the query into individual terms
        query_terms = self.tokenize(query)  # Tokenizing the query
        
        # Initialize QdrantDbUtil
        qdrant_db_util = QdrantDbUtil()

        try:
            # Search in Qdrant DB
            search_results = qdrant_db_util.keyword_search(query)
        except Exception as e:
            logger.error(f"Exception occurred in search_keyword_results: {e}")
            return []

        # Process results and calculate BM25 scores using query terms
        keyword_results = []
        for index, result in enumerate(search_results, start=1):
            # Get the chunk id and document data
            chunk_id = result.id
            document_text = result.payload.get('text', 'No text available')
            document_url = result.payload.get('doc_url', 'No URL available')
            bm25_index = result.payload.get('bm25_index', {}).get('terms', {})  # BM25 terms

            # Calculate BM25 score for the document based on query terms
            bm25_score = sum(bm25_index.get(term, {}).get('tf', 0) * bm25_index.get(term, {}).get('idf', 0)
                             for term in query_terms)

            keyword_results.append({
                "rank": index, 
                "chunk_id": chunk_id, 
                "score": bm25_score,
                "chunk_text": document_text,  
                "doc_url": document_url
            })

        # Sort the combined results by combined score
        keyword_results = sorted(keyword_results, key=lambda x: x['score'], reverse=True)[:5]


        return keyword_results


# Example usage
if __name__ == "__main__":    
    ksm = KeywordSearchModule()
    results = ksm.search_keyword_results("friend of man")
    
