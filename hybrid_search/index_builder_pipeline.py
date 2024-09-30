from util import file_util
from util import sqlite_util
from util.semantic_chunk_parser import SemanticChunkParser
from keyword_search_module import KeywordSearchModule

from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import json
from loguru import logger


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
    # Best model for semantic classification
    MODEL_1 = "BAAI/bge-large-zh-v1.5"
    #initialize the model.
    embedder = SentenceTransformer(MODEL_1)

    '''
    * Below function is the index builder that builds semantic and 
    * key word indexes
    '''
    def build_cumulative_indexes(self):
        #call the semantic index builder
        self.build_semantic_index()
        #call the keyword index builder
        self.build_keyword_index()

    '''
    * Below function is the index builder that builds semantic and 
    * key word indexes
    * Below function builds the semantic dense vector embeddings index 
    * and saves to sql lite vector and pickle file.
    * It also builds the keyword index and saves to pickle storage..
    '''
    def build_semantic_index(self):
        #initialize semantic chunker object
        scp = SemanticChunkParser()

        try:
            # download the books to local folder structure.
            file_util.download_files(self.book_list)
            # Iterate through folder structure and build the brute force chunks.
            file_list = file_util.iterate_folder()
            for file_obj in file_list:
                #retrieve brute force chunks.
                brute_force_chunk = file_util.convert_to_chunks(file_obj)
                # Pass the brute force chunks and build the semantic chunks.
                sem_chunk_coll = scp.chunk_data_in_semantic_pattern(brute_force_chunk)
                #append to sem chunks list
                self.sem_chunks.append(sem_chunk_coll)

            # save the vector embeddings to pickle and data store to store the index.
            self.build_data_store(self.sem_chunks)

        except IOError as ie:
            logger.error( f"IO Exception occured : {ie}" )
        except ConnectionError as ce:
            logger.error( f"Database connection error : {ce}" )
        except Exception as e:
            logger.error( f"Exception occured as : {e}" )
    
    """
    * function to build the local datastore that contains chunk_id,
    * chunt_txt, embeddings, document_id and document_file
    """
    def build_data_store(self, sem_chunks_list : list):
        
        logger.info( f" sem chunks list : {len(sem_chunks_list)}")


        # Initialize Data Dictionary
        chunk_data_dict = {}

        # Create individual lists to contain the data
        chunk_id_list = []
        chunk_list = []
        vector_embedding_list = []
        document_id_list = []
        document_title_list = []
        chunk_counter = 0
        final_embeddings_list = []

        '''
        * Iterate through the list that contains the semantic chunks and build
        * the vector embeddings, chunk text, chunk id and document titk
        '''
        for index, chunk_data_list in enumerate(sem_chunks_list):
            # retrieve book title
            doc_title = self.book_list[index]

            # Prepare the embeddings for the paragraph list
            corpus_embeddings = self.embedder.encode(chunk_data_list,
                                            convert_to_tensor=True)
            corpus_embeddings_cpu = corpus_embeddings.cpu().numpy()
            final_embeddings_list.append(corpus_embeddings_cpu)
            size = corpus_embeddings.size(0)


            # Process each chunk and its embedding
            for i in range(size):
                # Populate chunk id's
                chunk_id_list.append(chunk_counter)

                #populate the chunk texts.
                chunk_list.append(chunk_data_list[i])
                
                # Retrieve individual vector embeddings and store as a list
                # tensor_obj = corpus_embeddings[i].cpu()
                # numpy_arr = tensor_obj.numpy()       
                # Convert to list for CSV compatibility
                vector_embedding_list.append(corpus_embeddings_cpu[i].tolist())  

                # associate the book title
                document_title_list.append(doc_title)
                #associate the document id
                document_id_list.append(index)

                # increment the chunk id
                chunk_counter += 1

        # Populate the chunk data dictionary
        chunk_data_dict['chunk_id'] = chunk_id_list
        chunk_data_dict['chunk_txt'] = chunk_list
        chunk_data_dict['vector_embeddings'] = vector_embedding_list
        chunk_data_dict['document_id'] = document_id_list
        chunk_data_dict['document_file'] = document_title_list

        # Add the final embeddings to np single array for pickle load.
        final_embeddings_list = np.vstack(final_embeddings_list)

        logger.info( "before pickle" )

        '''
        save vector embeddings to pickle data store, which will serve
        as local cache of vector embeddings.
        '''
        try:
            file_obj = file_util.save_to_pickle(final_embeddings_list)
        except IOError as ioe:
            logger.error(  f"IOError occured : {ioe}" )
        except Exception as e:
            logger.error(  f"Exception occured : {e}" )

        logger.info(  "after pickle" )

        # Create a dataframe object with chunk data dictionary
        logger.info(  "before dataframe" )

        df = pd.DataFrame(chunk_data_dict)
        #convert the vector embeddings to json before storing in DB
        df['vector_embeddings'] = df['vector_embeddings'].apply(json.dumps)
        logger.debug( "after dataframe" )
        logger.info( f"DataFrame object: {df.tail()}" )
        logger.info( f"embeddings length : {len(final_embeddings_list)}" ) 

        # write it to database
        # create db connection
        try:
            connection = sqlite_util.connect_to_db()
            # pass the contents to db
            sqlite_util.write_to_db_from_dataframe(connection, df)
        except ConnectionError as ce:
            logger.error( f"Connection error : {ce}" )
        except Exception as e:
            logger.error( f"Exception occured : {e}" )
    
    '''
    * function to build the keyword index.
    * retrieves the sem chunks and pass it on to the KeywordSearchModule
    * class to build the index.
    '''
    def build_keyword_index(self):
        ksm = KeywordSearchModule()
        collated_list = []
        logger.info(f" sem chunks length is : {len(self.sem_chunks)}")

        [collated_list.extend(sem_chunk) for sem_chunk in self.sem_chunks]
        logger.info(f"Collated sem chunks length is : {len(collated_list)}")

        ksm.retrieve_corpus(collated_list)
        built_index = ksm.build_index()
        logger.info("index has been built")

def testclasses():
    ibp = IndexBuilderPipeline()
    ibp.build_cumulative_indexes()

if __name__ == "__main__":
    testclasses()


