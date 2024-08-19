'''
This module will serve as the search provider tying UI and search 
corpus.
'''
from sentence_transformers import SentenceTransformer, util
import search_corpus_module as scm

# Load a pre-trained model for asymmetric search
MODEL = "msmarco-distilbert-base-v4"
embedder = SentenceTransformer(MODEL)

#retrieve search corpus list.
sentence_list = scm.return_list()

#embed the search corpus
corpus_embeddings = embedder.encode(sentence_list, convert_to_tensor = True)
print(corpus_embeddings.shape)

'''
* Function that takes in input query as string
    * It builds query vector embedding
    * Compare it against search corpus vector embeddings.
    * Uses the util module semantic search function of Sentence transformers.
'''
def search_lists(query_str : str) -> list:
    query_embeddings = embedder.encode(query_str,
                                        convert_to_tensor = True)

    #compare the query with search corpus using cosine similarities method
    search_results = util.semantic_search(query_embeddings,
                                           corpus_embeddings, top_k = 3)
    
    return search_results


