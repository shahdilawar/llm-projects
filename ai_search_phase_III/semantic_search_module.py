from sentence_transformers import SentenceTransformer, util
import pandas as pd
import numpy as np
import torch
import ast
import os
import json
import sqlite3

from util import file_util as file_util
from util import semantic_chunk_parser as sc
from util import sqlite_util as sqlite_util

# Load a pre-trained model for asymmetric search
MODEL =  "msmarco-distilbert-base-v4"
MODEL_1 = "BAAI/bge-large-zh-v1.5"
MODEL_2 = "paraphrase-MiniLM-L6-v2"
embedder = SentenceTransformer(MODEL_1)

#URL of the book from project gutenberg
url = "https://www.gutenberg.org/cache/epub/100/pg100.txt"
data_dir = "./ai_search_phase_III"

"""
* function to build the local datastore that contains chunk_id,
* chunt_txt, embeddings, document_id and document_file
"""
def build_sqlite_data_store(url : str):
    
    file_obj = file_util.download_file(url, "text")

    # Convert to chunks into list of paragraphs
    chunk_list = sc.chunk_data_in_semantic_pattern(url,
                                                     file_obj)
    print(f"{len(chunk_list)}")


    # Prepare the embeddings for the paragraph list
    corpus_embeddings = embedder.encode(chunk_list,
                                         convert_to_tensor=True)
    size = corpus_embeddings.size(0)

    # Initialize Data Dictionary
    chunk_data_dict = {}

    # Create individual lists to contain the data
    chunk_id_list = []
    vector_embedding_list = []
    
    # Populate the document_id and document_title based on chunk list length
    document_id_list = [1] * size
    document_title_list = [url] * size

    # Process each chunk and its embedding
    for index in range(size):
        # Populate chunk id's
        chunk_id_list.append(index)
        
        # Retrieve individual vector embeddings and store as a list
        tensor_obj = corpus_embeddings[index].cpu()
        numpy_arr = tensor_obj.numpy()       
        # # Convert to list for CSV compatibility
        vector_embedding_list.append(numpy_arr.tolist())  

    # Populate the chunk data dictionary
    chunk_data_dict['chunk_id'] = chunk_id_list
    chunk_data_dict['chunk_txt'] = chunk_list
    chunk_data_dict['vector_embeddings'] = vector_embedding_list
    chunk_data_dict['document_id'] = document_id_list
    chunk_data_dict['document_file'] = document_title_list

    '''
    save vector embeddings to pickle data store, which will serve
    as local cache of vector embeddings.
    '''
    file_obj = file_util.save_to_pickle(corpus_embeddings)

    # Create a dataframe object with chunk data dictionary
    df = pd.DataFrame(chunk_data_dict)
    #convert the vector embeddings to json before storing in DB
    df['vector_embeddings'] = df['vector_embeddings'].apply(json.dumps)

    print(f"DataFrame object: {df.head()}")

    # write it to database
    # create db connection
    connection = sqlite_util.connect_to_db()
    # pass the contents to db
    sqlite_util.write_to_db_from_dataframe(connection, df)

'''
*   Function that takes in the query string as input.
*   retrieves search corpus embeddings 
'''
def display_search_results(query_str : str) -> pd.DataFrame:
    '''
    Access the DataFrame object that contains vector embeddings 
    from pickle datastore.
    '''
    # Create data directory if it doesn't exist
    os.makedirs(os.path.join(data_dir, "data"), exist_ok=True)
    data_file = os.path.join(data_dir, "data", "embeddings.pkl")

    embeddings_list = file_util.read_from_pickle(data_file)
    print(f"{type(embeddings_list[0])}")

    # Create the query embedding
    query_embeddings = embedder.encode(query_str, convert_to_tensor=True)

    # Compare the query with search corpus using cosine similarities method
    search_results = util.semantic_search( query_embeddings,
                                           embeddings_list, top_k=3)

    # Retrieve search results and populate rank, score, and chunk_id lists
    rank_list = []
    score_list = []
    chunk_id_list = []

    for index, result in enumerate(search_results[0]):
        rank_list.append(index + 1)
        score_list.append(result['score'])
        chunk_id_list.append(result['corpus_id'])
    
    print(f"{chunk_id_list}")

    # Filter the records by the chunk IDs
    conn = sqlite_util.connect_to_db()
    filtered_df = sqlite_util.load_data_to_dataframe(conn,
                                                      chunk_id_list)
    
    filtered_df['rank'] = pd.Series(rank_list,
                                     index=filtered_df.index)
    filtered_df['score'] = pd.Series(score_list,
                                      index=filtered_df.index)

    # Reset the index to avoid issues with missing chunk_ids
    filtered_df.reset_index(drop=True, inplace=True)

    return filtered_df

if __name__ == "__main__":
    build_sqlite_data_store(url)
    df = display_search_results("backpropogation")
    print(f"{df.head()}")
