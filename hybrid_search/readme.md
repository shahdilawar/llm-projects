# AI Hybrid Search

AI Hybrid Search is a Python-based project that combines semantic search and keyword-based search to provide efficient and accurate document retrieval. This project uses various natural language processing techniques and machine learning models to process, index, and search through a collection of documents.

## Table of Contents

1. [Features](#features)
2. [Installation](#installation)
3. [Project Structure](#project-structure)
4. [Usage](#usage)
5. [Components](#components)
6. [Dependencies](#dependencies)
7. [Contributing](#contributing)
8. [License](#license)

## Features

- Hybrid search combining semantic and keyword-based approaches
- Semantic chunking of documents for better context preservation
- Vector embeddings for efficient semantic search
- BM25 algorithm for keyword-based search
- SQLite database for storing document information and embeddings
- Streamlit-based web interface for easy interaction
- Cross-encoder re-ranking for improved search results

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/ai-hybrid-search.git
   cd ai-hybrid-search
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Download the necessary pre-trained models and data files (if any).

## Project Structure

![Project Architecture](hybrid_search/images/proj_arch.png)


## Usage

1. Build the search index:
   ```
   python index_builder_pipeline.py
   ```

2. Run the Streamlit web interface:
   ```
   streamlit run home.py
   ```

3. Open your web browser and navigate to the provided local URL (usually `http://localhost:8501`).

4. Enter your search query in the text input field and view the hybrid, semantic, and keyword search results.

## Components

### file_util.py
Utility functions for file operations, including downloading files, chunking text, and handling pickle files.

### sqlite_util.py
Functions for interacting with the SQLite database, including creating tables and reading/writing data.

### semantic_chunk_parser.py
Implements semantic chunking of documents to preserve context and improve search quality.

### combined_search.py
Combines and re-ranks results from semantic and keyword searches using a cross-encoder model.

### home.py
Streamlit-based web interface for the search application.

### hybrid_search_module.py
Main module that orchestrates the hybrid search process, combining semantic and keyword search results.

### index_builder_pipeline.py
Builds the search indexes for both semantic and keyword-based search.

### keyword_search_module.py
Implements keyword-based search using the BM25 algorithm.

### semantic_search.py
Implements semantic search using sentence transformers and vector embeddings.

### search_corpus_module.py
Contains sample text data for testing and development purposes.

## Dependencies

- Python 3.7+
- Streamlit
- PyTorch
- Sentence Transformers
- SQLite
- Pandas
- NumPy
- Loguru
- Tika
- bm25s

For a complete list of dependencies, please refer to the `requirements.txt` file.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.


