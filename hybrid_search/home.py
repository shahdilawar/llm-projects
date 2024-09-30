# home.py

import streamlit as st
from hybrid_search_module import HybridSearchModule
import pandas as pd

st.title(":blue[Welcome to AI Hybrid Search] :sunglasses:")
search_text = st.text_input("Enter your search query:")

#pass the search text to our search module
if (len(search_text) != 0): 

    #Initialize the HybrideSearchModule class
    hsm = HybridSearchModule()

    # Call the hybrid search function and get the results
    result_data_frame = hsm.display_hybrid_search_results(search_text)
    st.header("Hybrid search results")

    # Check if the DataFrame is empty
    if not result_data_frame.empty:
        # Select specific columns to display
        columns_to_display = ['rank', 'score', 'chunk_id', 'chunk_txt', 'document_file']
        subset_df = result_data_frame[columns_to_display]
            
        # Convert 'document_file' column to clickable URLs
        subset_df['document_file'] = subset_df['document_file'].apply(
            lambda x: f'<a href="{x}" target="_blank">{x}</a>'  
        )
        
    # Convert subset dataframe to html to display document file as
    # link
    st.columns([0.25, 0.25, 0.25, 16, 0.25])
    st.markdown(subset_df.to_html(escape=False), unsafe_allow_html=True)


    # Call the semantic search function and get the results
    result_data_frame = hsm.display_semantic_search_results(search_text)
    st.header("Semantic search results")

    # Check if the DataFrame is empty
    if not result_data_frame.empty:
        # Select specific columns to display
        columns_to_display = ['rank', 'score', 'chunk_id', 'chunk_txt', 'document_file']
        subset_df = result_data_frame[columns_to_display]
            
        # Convert 'document_file' column to clickable URLs
        subset_df['document_file'] = subset_df['document_file'].apply(
            lambda x: f'<a href="{x}" target="_blank">{x}</a>'  
        )
        
    # Convert subset dataframe to html to display document file as
    # link
    st.columns([0.25, 0.25, 0.25, 16, 0.25])
    st.markdown(subset_df.to_html(escape=False), unsafe_allow_html=True)

    # Section for keyword search
    # Initialize the HybridSearchModule class
    st.header("Keyword search results")

    # Call the key word search function and get the results
    result_data_frame_1 = hsm.display_keyword_search_results(search_text)

    # Check if the DataFrame is empty
    if not result_data_frame_1.empty:
        # Select specific columns to display
        columns_to_display = ['rank', 'score', 'chunk_id', 'chunk_txt', 'document_file']
        subset_df = result_data_frame_1[columns_to_display]
            
        # Convert 'document_file' column to clickable URLs
        subset_df['document_file'] = subset_df['document_file'].apply(
            lambda x: f'<a href="{x}" target="_blank">{x}</a>'  
        )
        
    # Convert subset dataframe to html to display document file as
    # link
    st.columns([0.25, 0.25, 0.25, 16, 0.25])
    st.markdown(subset_df.to_html(escape=False), unsafe_allow_html=True)

