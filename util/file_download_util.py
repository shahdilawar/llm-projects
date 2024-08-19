import requests
#URL of the book from project gutenberg
url = "https://www.gutenberg.org/cache/epub/100/pg100.txt"

#send a get request
response = requests.get(url)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Save the content to a .txt file
    with open("complete_works_of_william_shakespeare.txt", "w", encoding='utf-8') as file:
        file.write(response.text)
        print("Book downloaded successfully!")
else:
    print(f"Failed to download the book. Status code: {response.status_code}")