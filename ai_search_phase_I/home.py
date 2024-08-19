import streamlit as st
import search_corpus_module as sc
import search_module as sm

st.title(":blue[Welcome! to AI search] :sunglasses:")
search_text = st.text_input("")

#pass the search text to our search module
if (len(search_text) != 0): 
    search_data_list = sm.search_lists(search_text)
    sentence_list = sc.return_list()

    #retrieve through search list and print rank, score and sentence
    for index, result in enumerate(search_data_list[0]):
        st.write(f'Rank is {index} : and score is : {result["score"]} ' )
        st.write(f"{sentence_list[result['corpus_id']]}")

