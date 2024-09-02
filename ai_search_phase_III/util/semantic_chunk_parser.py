from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np
from util import file_util as file_util

#initialize the embeddings model.
MODEL = "all-MiniLM-L6-v2"
MODEL_2 = "msmarco-distilbert-cos-v5"
MODEL_3 = "msmarco-distilbert-base-v4"
MODEL_4 = "distiluse-base-multilingual-cased-v1"

# Best model for semantic classification
MODEL_1 = "BAAI/bge-large-zh-v1.5"
# Lightweight for CPU
MODEL_5 = "paraphrase-MiniLM-L6-v2" 
similarity_embeddings = SentenceTransformer(MODEL_1)

# initialize similarity score to be 0.6
BENCHMARK_SIMILARITY_SCORE =  0.6

#batch size
BATCH_SIZE = 500

'''
Utility method for batch encodings.
'''
def batch_encode(paragraphs : list,
                  model : SentenceTransformer) -> list:
    # list to contain the embeddings.
    embeddings = []
    
    paragraphs_size = len(paragraphs)
    print(f"paragraphs_size in batch method : {paragraphs_size}")

    if paragraphs_size < BATCH_SIZE:
        batch_embeddings = model.encode(paragraphs, 
                                        convert_to_numpy=True)
        embeddings.append(batch_embeddings)
    else:
        for i in range(0, paragraphs_size, BATCH_SIZE):
            # retrieve chunks in batches by index slicing.
            print(f"i is : {i}")
            batch = paragraphs[i : (i + BATCH_SIZE)]
            #embed them in batches
            batch_embeddings = model.encode(batch, convert_to_numpy=True)
            embeddings.append(batch_embeddings)

    return np.vstack(embeddings)


'''
* Function that takes in list of paragraphs as input.
* Does a paragraph to paragraph cosine similarity comparison and chunk them
* together.
* we will take a 0.5 score as benchmark.
'''
def chunk_data_in_semantic_pattern(url : str,
                                    file_name : str) -> list:

    file_obj = file_util.download_file(url, file_name)
    print(f"{file_obj}")
    paragraphs = file_util.convert_to_chunks(file_obj)

    print(f"{len(paragraphs)}")

    # Embed each paragraph and add it numpy array into 2D array using list
    # comprehension for performance reasons.
    # paragraph_embeddings = [ np.array(similarity_embeddings.encode(para)).reshape(1, -1)
    #                                  for para in paragraphs ]
    paragraph_embeddings = batch_encode(paragraphs,
                                         similarity_embeddings)
    
    # Initialize the list that will contain the semantic chunks
    semantic_chunks = []

    # Compute cosine similarity between all paragraphs at once
    similarity_matrix = cosine_similarity(paragraph_embeddings)

    '''
    *   Iterate through the paragraphs list. 
        * Compare the previous para embedding's with current embedding.
        * If they have a similarity score > 0.4 then chunk it together.
    '''
    for i in range( len(paragraphs) ):
        # Check for first element.
        if i == 0:
            semantic_chunks.append( [paragraphs[i]] )
        else:
            #retrieve the similarity score between two chunks
            similarity_score = similarity_matrix[i-1][i]
            # compare against benchmark score
            if similarity_score > BENCHMARK_SIMILARITY_SCORE:
                #append the current chunk to the previous para
                #contained in list
                semantic_chunks[-1].append( paragraphs[i] )
            else:
                semantic_chunks.append( [paragraphs[i]] ) 

    # Join paragraphs within each chunk and then flatten the list of chunks
    flattened_chunks = [' '.join(chunk) for chunk in semantic_chunks]


    print(f"{type(flattened_chunks)}")
    print(f"{flattened_chunks}")

    return flattened_chunks

def test_classes():
    url = "https://www.gutenberg.org/cache/epub/100/pg100.txt"
    file_name = "complete_works_of_william_shakespeare.txt"

    chunk = chunk_data_in_semantic_pattern(url, file_name)

    print(f"chunk size is : {len(chunk)}")

if __name__ == "__main__":
    test_classes()
            

