import streamlit as st
import ai_search_phase_III.semantic_search_module as asm
import pandas as pd

st.title(":blue[Welcome to AI Search - phase III] :sunglasses:")
search_text = st.text_input("Enter your search query:")

#pass the search text to our search module
if (len(search_text) != 0): 
    # Call the search function and get the results
    result_data_frame = asm.display_search_results(search_text)
        
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
        st.columns([1, 1, 1, 4, 0.25])
        st.markdown(subset_df.to_html(escape=False), unsafe_allow_html=True)


