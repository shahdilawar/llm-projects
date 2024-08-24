from sentence_transformers import SentenceTransformer, util
import pandas as pd
import numpy as np
import torch
import ast
import os
import util.fille_util as file_util

# Load a pre-trained model for asymmetric search
MODEL = "msmarco-distilbert-base-v4"
embedder = SentenceTransformer(MODEL)

#URL of the book from project gutenberg
url = "https://www.gutenberg.org/cache/epub/100/pg100.txt"

"""
* function to build the local datastore that contains chunk_id,
* chunt_txt, embeddings, document_id and document_file
"""
def build_csv_data_store(url : str):
    
    print("1")
    file_obj = file_util.download_file(url)
    print("2")
    # Convert to chunks into list of paragraphs
    chunk_list = file_util.convert_to_chunks(file_obj)
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
        # Convert to list for CSV compatibility
        vector_embedding_list.append(numpy_arr.tolist())  

    # Populate the chunk data dictionary
    chunk_data_dict['chunk_id'] = chunk_id_list
    chunk_data_dict['chunk_txt'] = chunk_list
    chunk_data_dict['vector_embeddings'] = vector_embedding_list
    chunk_data_dict['document_id'] = document_id_list
    chunk_data_dict['document_file'] = document_title_list

    # Create a dataframe object with chunk data dictionary
    df = pd.DataFrame(chunk_data_dict)
    print(f"DataFrame object: {df.head()}")

    output_dir = "./ai_search_phase_II"
    # Create data directory if it doesn't exist
    os.makedirs(os.path.join(output_dir, "data"), exist_ok=True)
    data_file = os.path.join(output_dir, "data", "csv_data_store.csv")

    # Create the file and write the DataFrame to CSV
    with open(data_file, mode='w', encoding='utf-8') as file:
        df.to_csv(file, index=False)


'''
* utility function to load the datastore
'''
def load_data_store() -> pd.DataFrame:

    #access the file
    output_dir = "./ai_search_phase_II"
    data_file = os.path.join(output_dir, "data", "csv_data_store.csv")

    with open(data_file, 'r') as file:
        dataframe_from_data_store = pd.read_csv(file)

    return dataframe_from_data_store

def display_search_results(query_str : str) -> pd.DataFrame:
    # Access the DataFrame object that represents local datastore
    data_store_df = load_data_store()

    # Create the query embedding
    query_embeddings = embedder.encode(query_str, convert_to_tensor=True)

    # Convert the vector embeddings from string format to tensors
    data_store_df['vector_embeddings'] = data_store_df['vector_embeddings'].apply(
        lambda x: torch.tensor(np.array(ast.literal_eval(x)), dtype=torch.float32)
    )

    # Ensure the 'vector_embeddings' column is of type list of tensors
    tensor_list = data_store_df['vector_embeddings'].tolist()

    # Check if the conversion to tensor was successful
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensor_list):
        raise ValueError("Not all elements in 'vector_embeddings' are tensors.")

    # Stack the tensor list
    corpus_embeddings = torch.stack(tensor_list)

    # Compare the query with search corpus using cosine similarities method
    search_results = util.semantic_search(query_embeddings, corpus_embeddings, top_k=3)

    # Retrieve search results and populate rank, score, and chunk_id lists
    rank_list = []
    score_list = []
    chunk_id_list = []

    for index, result in enumerate(search_results[0]):
        rank_list.append(index + 1)
        score_list.append(result['score'])
        chunk_id_list.append(result['corpus_id'])

    # Filter the records by the chunk IDs
    filtered_df = data_store_df[data_store_df['chunk_id'].isin(chunk_id_list)].copy()
    filtered_df['rank'] = pd.Series(rank_list, index=filtered_df.index)
    filtered_df['score'] = pd.Series(score_list, index=filtered_df.index)

    # Reset the index to avoid issues with missing chunk_ids
    filtered_df.reset_index(drop=True, inplace=True)

    return filtered_df

if __name__ == "__main__":
    build_csv_data_store(url)
    df = display_search_results("sonnet")
