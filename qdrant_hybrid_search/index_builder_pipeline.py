# index_builder_pipeline.py

from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import json
from rank_bm25 import BM25Okapi
from typing import List, Dict
from loguru import logger

from util import file_util
from util.semantic_chunk_parser import SemanticChunkParser
from util.qdrantdb_util import QdrantDbUtil
from keyword_search_module import KeywordSearchModule


class IndexBuilderPipeline:

    # Variable containing the document list to load data.
    book_list = [ "https://www.gutenberg.org/cache/epub/1513/pg1513.txt", 
                 "https://www.gutenberg.org/cache/epub/1342/pg1342.txt", 
                 "https://www.gutenberg.org/cache/epub/100/pg100.txt", 
                 "https://www.gutenberg.org/cache/epub/34463/pg34463.txt", 
                 "https://www.gutenberg.org/cache/epub/4361/pg4361.txt", 
                 "https://www.gutenberg.org/cache/epub/4367/pg4367.txt", 
                 "https://www.gutenberg.org/files/18993/18993-pdf.pdf",
                 "https://scholar.harvard.edu/files/shleifer/files/stock_market_and_investment.pdf", 
                 "https://unipub.lib.uni-corvinus.hu/3842/1/pfi-briefings.pdf", 
                 "https://web.cs.dal.ca/~tt/CSCI415009b/CSCI415009.pdf"
                     ]
    
    #list to hold the semantic chunks.
    sem_chunks = []
    # list to hold the document URL's
    doc_title_list = []
    # Best model for semantic classification
    MODEL_1 = "BAAI/bge-large-zh-v1.5"
    #initialize the model.
    embedder = SentenceTransformer(MODEL_1)

    '''
    * Below function is the index builder that builds semantic and 
    * key word indexes
    * Below function builds the semantic dense vector embeddings index 
    * and saves to sql lite vector and pickle file.
    * It also builds the keyword index and saves to pickle storage.
    '''
    def build_cumulative_indexes(self):
        # Call the semantic index builder. This provides a tuple of chunks
        # and document titles.
        try:
            self.sem_chunks, self.doc_title_list = self.build_semantic_index()
            # Call the keyword index builder
            keyword_index_data = self.build_keyword_index()

            # Call the build datastore that indexes dense and sparse embeddings
            # used for hybrid search.
            self.build_data_store(self.sem_chunks, self.doc_title_list,
                                keyword_index_data)
        except Exception as e:
            logger.error(f"exception in build cumulative indexes: {e}")

    '''
    * Below function builds the semantic dense vector embeddings index 
    * and saves to sql lite vector and pickle file.
    '''
    def build_semantic_index(self) -> tuple:
        #initialize semantic chunker object
        scp = SemanticChunkParser()

        #initialze url_list
        url_list = []

        try:
            # download the books to local folder structure.
            file_util.download_files(self.book_list)
            # Iterate through folder structure and build the brute force chunks.
            file_list = file_util.iterate_folder()
            # retrieve through list of downloaded files, convert to
            # semantic chunks 
            for index, file_obj in enumerate(file_list):
                # retrieve brute force chunks.
                brute_force_chunk = file_util.convert_to_chunks(file_obj)
                # Pass the brute force chunks and build the semantic chunks.
                sem_chunk_processed = scp.chunk_data_in_semantic_pattern(brute_force_chunk)
                
                # extend to sem chunks list so that we have a single list.
                # here we can do the extend of the collections.
                # in build data store method, we encode and send the embeddings.
                self.sem_chunks.extend(sem_chunk_processed)

                # 
                url_list = [self.book_list[index]] * len(sem_chunk_processed)
                self.doc_title_list.extend(url_list)

            logger.info(f"len of semantic chunks are : {len(self.sem_chunks)}")
            logger.info(f"len of doc title is are : {len(self.doc_title_list)}")

            return (self.sem_chunks, self.doc_title_list)

        except IOError as ie:
            logger.error( f"IO Exception occured : {ie}" )
        except ConnectionError as ce:
            logger.error( f"Database connection error : {ce}" )
        except Exception as e:
            logger.error( f"Exception occured as : {e}" )

    '''
    * function to build the keyword index.
    * retrieves the sem chunks and pass it on to the KeywordSearchModule
    * class to build the index.
    '''
    def build_keyword_index(self):
        logger.info(f" inside build keyword index - sem chunks length is : {len(self.sem_chunks)}")

        ksm = KeywordSearchModule()
        ksm.retrieve_corpus(self.sem_chunks)
        retriever = ksm.build_index()  
        logger.info(f"index has been built : {type(retriever)}")              
        # Prepare BM25 data
        bm25_data = ksm.prepare_bm25_data(retriever)

        return bm25_data

    
    """
    * function to build the local datastore that contains chunk_id,
    * chunt_txt, embeddings, document_file
    """
    def build_data_store(self, sem_chunks_list : list,
                          doc_title_list : list, 
                          bm25_index : List[Dict]):
        
        logger.info( f" sem chunks list : {len(sem_chunks_list)}")

        try:
            # Initialize QdrantDbUtil
            qdrant_db_util = QdrantDbUtil()

            # Prepare the embeddings for the paragraph list
            corpus_embeddings = self.embedder.encode(sem_chunks_list,
                                            convert_to_tensor=True)

            size = corpus_embeddings.size(0)
            logger.info(f"Length of embeddings : {size}")

            # call the qdrant db
            qdrant_db_util.load_data_to_qdrantdb(corpus_embeddings, sem_chunks_list, 
                                                bm25_index, doc_title_list)
        except Exception as e:
            logger.error(f"Exception occured in IndexBuilderPipeline.build_data_store : {e}")


def testclasses():
    ibp = IndexBuilderPipeline()
    ibp.build_cumulative_indexes()

if __name__ == "__main__":
    testclasses()


