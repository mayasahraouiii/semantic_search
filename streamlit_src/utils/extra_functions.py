import pandas as pd
import torch

from transformers import AutoModelForSequenceClassification, AutoConfig, AutoTokenizer, AdamW

def extract_theme_and_company_data(data):
    """
    Extracts themes from 'query' and company information (name and alpha_id) from 'hits'
    for each entry in the provided data and returns a DataFrame containing two columns: 
    'Themes', 'Company Names', and 'Alpha IDs'.
    
    Parameters:
    data (list): A list of dictionaries containing 'query' and 'hits' with 'content'.
    
    Returns:
    pd.DataFrame: A DataFrame containing extracted 'Themes', 'Company Names', and 'Alpha IDs'.
    """

    # Helper function to extract the main focus from the 'query' string
    def extract_main_focus(text):
        # Split the text by the question mark (?)
        parts_before_question_mark = text.split('?')
        if len(parts_before_question_mark) > 0:
            # Focus on the first part before the question mark
            part_before_question = parts_before_question_mark[0]
            # Split by 'to ' to find the main focus (assuming 'to ' splits question and focus)
            parts_after_period = part_before_question.split('to ')
            # Return the last part and remove extra whitespace
            return parts_after_period[-1].strip()
        return ''
    
    # Initialize a list to store the results for each entry
    result = []

    # Iterate through each entry in the input data
    for entry in data:
        # Extract the theme from the query if available
        theme = extract_main_focus(entry['query']) if 'query' in entry else None
        
        # Initialize lists for company names and alpha_ids for each entry
        company_names = []
        alpha_ids = []
        
        # Iterate through the 'hits' in the entry
        for hit in entry.get('hits', []):
            # Extract the content
            content = hit.get('content', '')
            
            # Extract the company name using split
            if '"name":' in content:
                name = content.split('"name":')[1].split(",")[0].strip()
                company_names.append(name)  # Add the extracted name to the list
            
            # Extract the alpha_id using split
            if '"alpha_id":' in content:
                alpha_id = content.split('"alpha_id":')[1].split(",")[0].strip()
                alpha_ids.append(alpha_id)  # Add the extracted alpha_id to the list

        # Append the extracted data (theme, company names, and alpha_ids) to the result
        result.append({
            "Theme": theme, 
            "Company Names": company_names, 
            "Alpha IDs": alpha_ids
        })

    # Convert the result list to a pandas DataFrame
    df = pd.DataFrame(result)
    return df

def load_model_and_tokenizer(model_name: str, num_labels: int = 1, use_fast: bool = True):
    """
    Loads a pre-trained model, tokenizer, and configuration from the specified directory.

    Args:
        model_name (str): The directory or model name to load the model and tokenizer from.
        num_labels (int): The number of labels for the model (default is 1).

    Returns:
        model: The loaded model.
        tokenizer: The loaded tokenizer.
    """
    # Load the model configuration and set the number of labels
    config = AutoConfig.from_pretrained(model_name)
    config.num_labels = num_labels

    # Load the model and move it to the appropriate device (GPU if available, otherwise CPU)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=use_fast)

    # Load the tokenizer
    return model.to(device), tokenizer, device
