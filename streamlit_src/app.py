import os
import streamlit as st
import pandas as pd
from Inference_folder import get_theme_recommandations, get_permutations, get_theme_recommandations_no_tech
from utils import load_model_and_tokenizer

# Cache the model loading to avoid reloading it every time
@st.cache_resource
def cached_model_loading(model_name, num_labels, use_fast):
    return load_model_and_tokenizer(model_name, num_labels, use_fast)

def load_model_huggingface(model_name):
    """Return the model path based on the model name."""
    model_paths = {
        'finetuned__deberta_3': 'Mayasahraoui/deberta_finetuned_reranker_3',
        'pretrained_deberta': 'Mayasahraoui/deberta_pretrained_reranker',
        'finetuned_deberta_1': 'Mayasahraoui/deberta_finetuned_reranker_1',
        'finetuned__deberta_2': 'Mayasahraoui/deberta_finetuned_reranker_2',
        'finetuned_bge_1': 'Mayasahraoui/bge_finetuned_reranker_1',
        'finetuned_bge_2': 'Mayasahraoui/bge_finetuned_reranker_2'
    }
    return model_paths.get(model_name, "")

# Streamlit app UI
st.title("Model Selection and Semantic Search")

model_options = ['finetuned__deberta_3', 'pretrained_deberta', 'finetuned_deberta_1', 'finetuned__deberta_2', 'finetuned_bge_1', 'finetuned_bge_2']

selected_model = st.selectbox("Choose a model", model_options)

# Initialize model and tokenizer in session state if not already done
if 'model' not in st.session_state:
    model_path = load_model_huggingface(selected_model)
    if model_path:
        st.write("Loading model, please wait...")
        with st.spinner("Model is loading..."):
            model, tokenizer, device = cached_model_loading(model_path, num_labels=1, use_fast=True)
            st.session_state.model = model
            st.session_state.tokenizer = tokenizer
            st.session_state.device = device
            st.success(f"{selected_model} has been loaded and is ready to use!")
    else:
        st.error("Invalid model path. Please select a valid model.")

# Search input field
search_query = st.text_input("Enter words to search")
n_rank = st.number_input("Number of reranked companies", value=100)
tech_or_no_tech = st.toggle("Do you want to use the technology key words for your search ? ", value=True)

# Initialize state variables
if 'stop_flag' not in st.session_state:
    st.session_state.stop_flag = False
if 'search_in_progress' not in st.session_state:
    st.session_state.search_in_progress = False

# Search button
if st.button("Search"):
    if not search_query:
        st.error("Please enter a search query.")
    else:
        # Reset stop flag and search progress flag
        st.session_state.stop_flag = False
        st.session_state.search_in_progress = True
        
        # Show a spinner while the search results are being fetched
        with st.spinner("Searching for results..."):
            try:
                # Use cached model
                model = st.session_state.model
                tokenizer = st.session_state.tokenizer
                

                #ES_enriched = add_company_info_columns(ES_list_comp)

                if st.session_state.stop_flag:
                    st.warning("Search was stopped.")
                    st.session_state.search_in_progress = False
                else:
                    # Elastic search
                    
                    #using tech keywords
                    if tech_or_no_tech : 
                        items, df_es = get_theme_recommandations(search_query)
    
                        if isinstance(df_es, pd.DataFrame):
                            st.subheader("Elastic search output")
                            
                            st.dataframe(df_es)
                            
                        else:
                            st.write("The items are not in a DataFrame format.")
                        
                        
                        
                        #Reranking
                        df_reranked = get_permutations(items, model, tokenizer, df_es, n_rank)
                        
                        
                        if isinstance(df_reranked, pd.DataFrame):
                            st.subheader("Reranked Items")
                            st.dataframe(df_reranked)
                        else:
                            st.write("The reranked items are not in a DataFrame format.")
                        st.session_state.search_in_progress = False
                        
                    else : 
                        items, df_es = get_theme_recommandations_no_tech(search_query)
    
                        if isinstance(df_es, pd.DataFrame):
                            st.subheader("Elastic search output")
                            
                            st.dataframe(df_es)
                            
                        else:
                            st.write("The items are not in a DataFrame format.")
                        
                        
                        
                        #Reranking
                        df_reranked = get_permutations(items, model, tokenizer, df_es, n_rank)
                        
                        
                        if isinstance(df_reranked, pd.DataFrame):
                            st.subheader("Reranked Items")
                            st.dataframe(df_reranked)
                        else:
                            st.write("The reranked items are not in a DataFrame format.")
                        st.session_state.search_in_progress = False

            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.session_state.search_in_progress = False

# Stop button
if st.session_state.search_in_progress:
    if st.button("Stop"):
        st.session_state.stop_flag = True
        st.warning("Stopping the current process...")