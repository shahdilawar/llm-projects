# file_util.py

import requests
import os
import re
import pickle
import gzip
import tika
from tika import parser 
from pathlib import Path
import time
from loguru import logger


#output directory paths
output_dir = "./llm-projects/qdrant_hybrid_search"

'''
* This method takes in list of document files and saves them to
* local file path.
'''
def download_files(document_urls : list):

    # create the data directory if it does not exist
    os.makedirs( os.path.join(output_dir, 'data'), exist_ok=True )

    # Use Tika Client only, without starting the server
    parser.TikaClientOnly = True
    logger.debug("Tika Client initialized")
    

    #using tika download the files.
    for file_url in document_urls:
        # Use the Tika parser class to download the file.
        try:
            parsed = parser.from_file( file_url )  
            #content is extracted
            file_content = parsed["content"]     
            #extract metadata of the file.
            meta_data = parsed['metadata']
            logger.debug(f"metadata is : {meta_data}")
            #retrieve the file title.
            #book_title = meta_data.get( "meta:Title", "Title not found" )
            book_title = meta_data.get("dc:title") or meta_data.get("title", "Title not found")
            # # If title not found in metadata, extract from content
            if book_title == "Title not found":
                # Look for title at the beginning of the content
                title_search = re.search(r'(?<=Title: ).+', file_content)
                if title_search:
                    book_title = title_search.group(0)
                else:
                    # Fallback to using a default name if title extraction fails
                    timestamp_in_millis = int(time.time() * 1000)

                    book_title = f"{timestamp_in_millis}"

            data_file = os.path.join( output_dir, "data", book_title + ".txt" )
            logger.debug(f"file is : {data_file}")


            # print(f"file content is : {file_content}")
            with open(data_file, "w", encoding='utf-8') as file:
                file.write(file_content)
                logger.debug( f"{book_title} downloaded successfully!" )
            
            #call the utility methods to cleanup the file 
            #read between markers:
            cleaned_up_content = read_between_markers(data_file)

            if len(cleaned_up_content) != 0:
                #write between markers and save to the local file system
                write_between_markers(data_file, cleaned_up_content)

        except FileNotFoundError:
            logger.error( f"File not found error : {file_url}" )
        except Exception as e:
            logger.error( f"exception occured : {e}" )

'''
* Function to read the file from project gutenburg within the markers.
* This way we get just the actual content and not licensing stuff.
# Gutenburg doc starts with ***start and ends with ***end
'''
def read_between_markers(file_path : str) -> list:
    # List containing the content between the markers.
    content = []

    #open the file
    with open (file_path, 'r') as file:
        start_reading = False
        #iterate through the lines.
        for line in file:
            if "*** START" in line:
                start_reading = True
                continue
            if "*** END" in line:
                start_reading = False
                break
            if start_reading:
                content.append(line.strip())
        return content
    
'''
Function that takes in the content and writes between the markers.
'''
def write_between_markers(file_path : str, content : list):
    with open(file_path, 'w', encoding='utf-8') as file:
        for line in content:
            file.write(line + "\n")
    
'''
* File that converts the file into chunks of paragraph.
'''
def convert_to_chunks(file_obj : str) -> list:       

    with open(file_obj, 'r') as file:
        text = file.read()

    # print(f"document text is : \n {text}")
    # Tokenize paragraphs using regular expressions
    paragraphs = re.split(r'\n\s*\n', text)

    return paragraphs

'''
* Utility method to iterate over folder and retun list of files 
* in the folder.
'''
def iterate_folder() -> list:

    try:
        folder_path = os.path.join(output_dir, 'data')
        logger.debug( f"folder path is : {folder_path}")
        #instantiate Path object.
        path_obj = Path(folder_path)
        #initialize list obj that contains the files list.
        file_list = []
        #iterate the files list and add it to the list.
        for file_path in path_obj.iterdir():
            file_list.append(file_path)
    except IOError as ie:
        logger.error( f"IO Error : {ie}")
    except Exception as e:
        logger.error( f"Exception occured  : {e}")

    return file_list


'''
*   This method is used to write vector embeddings to a pickle file.
*   This will save as a cache and avoid computing multiple times.
'''
def save_to_pickle(vector_embeddings):
    # Save embeddings
    # Create data directory if it doesn't exist
    os.makedirs(os.path.join(output_dir, "pickle_data"), exist_ok=True)
    data_file = os.path.join(output_dir, "pickle_data", "embeddings.pkl")
    
    #save the pickle file in data    
    with gzip.open(data_file, 'wb') as f:
        pickle.dump(vector_embeddings, f)

'''
* This function will serve as a cache
'''
def read_from_pickle(file_obj : str) -> list:

    with gzip.open(file_obj, 'rb') as f:
        vector_embeddings = pickle.load(f)

    return  vector_embeddings

if __name__ == "__main__":
    #instantiate books list
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


    download_files(book_list)
    file_list = iterate_folder() 
    res_list = []
    for file_obj in file_list:
        logger.debug( f"file obj is : {file_obj}" )
        res_list.extend( convert_to_chunks(file_obj) )
    logger.debug( f"{len(res_list)}" )
 
