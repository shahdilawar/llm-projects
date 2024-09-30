# semantic_chunk_parser.py

from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np
from torch import Tensor
from util import file_util as file_util
from loguru import logger

class SemanticChunkParser:
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
    * Function that takes in list of paragraphs as input.
    * Does a paragraph to paragraph cosine similarity comparison and chunk them
    * together.
    * we will take a 0.5 score as benchmark.
    '''
    def chunk_data_in_semantic_pattern(self, input_paragraphs : list) -> list:

        logger.debug( f"numner of input paragraphs{len(input_paragraphs)}" )

        #Embed the input tokens into vectors.
        paragraph_embeddings = self.similarity_embeddings.encode(
                                            input_paragraphs,
                                            convert_to_tensor = True)
        
        # Convert tensor to numpy array for cosine_similarity
        paragraph_embeddings_np = paragraph_embeddings.cpu().numpy()
        
        # Initialize the list that will contain the semantic chunks
        semantic_chunks = []

        # Compute cosine similarity between all paragraphs at once
        similarity_matrix = cosine_similarity(paragraph_embeddings_np)

        '''
        *   Iterate through the paragraphs list. 
            * Compare the previous para embedding's with current embedding.
            * If they have a similarity score > 0.4 then chunk it together.
        '''
        for i in range( len(input_paragraphs) ):
            # Check for first element.
            if i == 0:
                semantic_chunks.append( [input_paragraphs[i]] )
            else:
                #retrieve the similarity score between two chunks
                similarity_score = similarity_matrix[i-1][i]
                # compare against benchmark score
                if similarity_score > self.BENCHMARK_SIMILARITY_SCORE:
                    #append the current chunk to the previous para
                    #contained in list
                    semantic_chunks[-1].append( input_paragraphs[i] )
                else:
                    semantic_chunks.append( [input_paragraphs[i]] ) 

        # Join paragraphs within each chunk and then flatten the list of chunks
        flattened_chunks = [' '.join(chunk) for chunk in semantic_chunks]

        return flattened_chunks

def test_classes():

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

    #retrieve files from folder
    file_list = file_util.iterate_folder()
    input_para = []
    sem_chunked = []
    data_dict = {}

    scp = SemanticChunkParser()

    for index, file_obj in enumerate(file_list):
        input_para.append(file_util.convert_to_chunks(file_obj))
        sem_chunked.append(scp.chunk_data_in_semantic_pattern(input_para[index]))

        #populate the data_dict object with title.
        data_dict[f"{book_list[index]}"] = sem_chunked[index]  
    
    logger.debug( f"chunk size is : {len(sem_chunked)}")
    logger.debug( f"data dict is : {data_dict}" )

if __name__ == "__main__":
    test_classes()
            

