import requests
import os
import re
import pickle


#URL of the book from project gutenberg
url = "https://www.gutenberg.org/cache/epub/100/pg100.txt"
output_dir = "./ai_search_phase_III"

def download_file(url : str, file_name : str) -> str:
    #send a get request
    response = requests.get(url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Save the content to a .txt file
        
        # Create data directory if it doesn't exist
        os.makedirs(os.path.join(output_dir, "data"), exist_ok=True)
        data_file = os.path.join(output_dir, "data", "text")
        
        with open(data_file, "w", encoding='utf-8') as file:
            file.write(response.text)
            print("Book downloaded successfully!")
    else:
        print(f"Failed to download the book. Status code: {response.status_code}")

    #call the utility methods to cleanup the file 
    #read between markers:
    content = read_between_markers(data_file)

    #write between markers
    file_obj = write_between_markers(data_file, content)

    return file_obj

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
def write_between_markers(file_path : str, content : list) -> str:
    with open(file_path, 'w', encoding='utf-8') as file:
        for line in content:
            file.write(line + "\n")
    
    return file_path

'''
* File that converts the file into chunks of paragraph.
'''
def convert_to_chunks(file_obj : str) -> list:       

    # Open and read the file
    with open(file_obj, 'r') as file:
        text = file.read()

    print(f"document text is : \n {text}")
    # Tokenize paragraphs using regular expressions
    paragraphs = re.split(r'\n\s*\n', text)

    return paragraphs

'''
*   This method is used to write vector embeddings to a pickle file.
*   This will save as a cache and avoid computing multiple times.
'''
def save_to_pickle(vector_embeddings):
    # Save embeddings
    # Create data directory if it doesn't exist
    os.makedirs(os.path.join(output_dir, "data"), exist_ok=True)
    data_file = os.path.join(output_dir, "data", "embeddings.pkl")
    
    #save the pickle file in data    
    with open(data_file, 'wb') as f:
        pickle.dump(vector_embeddings, f)

'''
* This function will serve as a cache
'''
def read_from_pickle(file_obj : str) -> list:

    with open(file_obj, 'rb') as f:
        vector_embeddings = pickle.load(f)

    return  vector_embeddings



if __name__ == "__main__":
    file_obj = download_file(url, "complete_works_of_william_shakespeare.txt")
    res_list = convert_to_chunks(file_obj)

    print(f"{len(res_list)}")