import requests
import os
import re

#URL of the book from project gutenberg
url = "https://www.gutenberg.org/cache/epub/100/pg100.txt"
output_dir = "./ai_search_phase_II"

def download_file(url : str) -> str:
    #send a get request
    response = requests.get(url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Save the content to a .txt file
        
        # Create data directory if it doesn't exist
        os.makedirs(os.path.join(output_dir, "data"), exist_ok=True)
        data_file = os.path.join(output_dir, "data", "complete_works_of_william_shakespeare.txt")
        
        with open(data_file, "w", encoding='utf-8') as file:
            file.write(response.text)
            print("Book downloaded successfully!")
    else:
        print(f"Failed to download the book. Status code: {response.status_code}")

    return data_file

def convert_to_chunks(file_obj : str) -> list:       

    # Open and read the file
    with open(file_obj, 'r') as file:
        text = file.read()

    # Tokenize paragraphs using regular expressions
    paragraphs = re.split(r'\n\s*\n', text)

    return paragraphs


if __name__ == "__main__":
    file_obj = download_file(url)
    res_list = convert_to_chunks(file_obj)

    print(f"{len(res_list)}")