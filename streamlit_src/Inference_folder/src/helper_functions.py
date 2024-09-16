import os
import re
import pandas as pd
import numpy as np
import torch
import json
from typing import List
from copy import deepcopy

from elasticsearch import Elasticsearch
from elasticsearch.client import IndicesClient
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from operator import itemgetter
from .settings import configurations


FINANCING_STATUS_LABELS = {
    "vc_backed": "VC Backed",
    "bootstrapped": "Bootstrapped",
    "pe_backed": "PE Backed",
    "others": "Others",
    "unknown": "Unknown",
}

credentials = {
    "demo-search": {
        "hostname": "https://demo-search-dev.es.westeurope.azure.elastic-cloud.com",
        "cloud id": "demo-search-dev:d2VzdGV1cm9wZS5henVyZS5lbGFzdGljLWNsb3VkLmNvbTo0NDMkNTg0ODkzODVlMzA2NDg3ZjhiZTEwY2VkNDI4NDU5ZjAkNWRiMGY2Mjg3M2RlNDk1OWE1ODc2MzIwNjBlYTYxZjI=",
        "Authentification user": "elastic",
        "Authentification password": "9X4ZOql31DjAMEDcFvnBPxHP",
    },
    "analytics": {
        "hostname": "https://analytics.es.westeurope.azure.elastic-cloud.com",
        "cloud id": "analytics:d2VzdGV1cm9wZS5henVyZS5lbGFzdGljLWNsb3VkLmNvbTo0NDMkYWRiNTFlM2M3NTM0NGU4MjljOWJhN2ZhYjQxZjMxNWIkYjAxNjcwMTgxMjRiNGYxNzllOTA2MmE5OTZkZTY3ZDY=",
        "Authentification user": "elastic",
        "Authentification password": "UjaAiXIMyHdCemSEGVrUdR09",
    },
} 

PORT = "443"
hostname = credentials["demo-search"]["hostname"]
cloud_id = credentials["demo-search"]["cloud id"]
auth_user = credentials["demo-search"]["Authentification user"]
auth_pass = credentials["demo-search"]["Authentification password"]

# Set up the connection
es_client = Elasticsearch(
    cloud_id=cloud_id,
    basic_auth=(auth_user, auth_pass),
    timeout=60
)
indexclient = IndicesClient(es_client)



def format_company_card(company: dict):
    return {
        "alpha_id": company["_source"].get("alpha_id", None),
        "name": company["_source"].get("name", None),
        "website": company["_source"].get("homepage_url", None),
        "description": company["_source"].get("description", None),
        "short_description": company["_source"].get("short_description", None),
        "founded_year": company["_source"].get("founded_year", None),
        "employee_count_range": company["_source"].get("employee_count_range", None),
    }



def get_companies(query):
    


    companies = es_client.search(
        index='search-orgs_v6.0.3_dev',
        query= query,
        size=10000,
        _source=[
            "alpha_id",
            "name",
            "description",
            "short_description",
            "founded_year",
            "homepage_url",
            "employee_count_range"
        ]
            )

    # Format Output into card.
    companies_cards = [
        format_company_card(company)
        for company in companies["hits"]["hits"]
    ]
    



    return {
        "companies": companies_cards
    }

os.environ['AZURE_OPENAI_API_KEY']  ='3f2a477724c54782826777a79558d213'
os.environ['AZURE_OPENAI_ENDPOINT']  = 'https://alpha10x-open-ai.openai.azure.com/'
os.environ['OPENAI_API_VERSION']  = '2023-07-01-preview'

gpt_4_o = AzureChatOpenAI( azure_deployment="gpt-4o",
            model_name="gpt-4o",
            temperature=0,
            seed=42 )
# Prompts
prompt = PromptTemplate.from_template(
"""We want to verify if a company is relevant to the {theme} theme.
          The company about the company:     {company_information} Return 1 only if you can validate that the company is relevant to the theme, else return 0. Remember, Answer with just 1 token, either "0" or "1".
""")

insight_chain = (
    RunnableParallel(
        company_information=itemgetter("company_information"),
        theme=itemgetter("theme"),
    )
    | prompt
    | gpt_4_o
    | StrOutputParser()
    | RunnableLambda(lambda x: bool(int(x)))
)

# @chain


def remove_outer_quotes(query):
    # Regular expression to match quotes at the beginning and end of the string
    pattern = r'^"([^"]*)"$'
    match = re.match(pattern, query)
    if match:
        return match.group(1)
    return query

market_segmentation_prompt = PromptTemplate.from_template("""You are tasked with generating queries to extract related companies for a given theme or industry. Your goal is to identify relevant scientific sub-technologies and domains of application within the theme, and to create ElasticSearch query strings that will enable searching for companies related to the theme, sub-technologies, and domains of application.

The theme/industry you will be analyzing is provided below:

<theme>
{theme}
</theme>

First, carefully analyze the given theme/industry and identify the key sub-technologies and domains of application that fall within its scope. Consider the various aspects and components that make up the theme/industry.

<scratchpad>
Analyze the theme and identify relevant sub-technologies and domains of application. Explain your reasoning for each identification and how they relate to the overall theme/industry. Consider the following:
1. What are the core technologies that enable this theme/industry?
2. What are the emerging or innovative technologies in this field?
3. In which industries or sectors is this theme/industry being applied?
4. What are the potential future applications of this theme/industry?
</scratchpad>

Next, generate an ElasticSearch query string that encompasses the overall theme/industry. This query string should be designed to search for companies whose LinkedIn descriptions and website content are relevant to the theme.

<es_query_theme>
Write your query string for the overall theme here. Ensure it is a single string, not a JSON object or dictionary.
</es_query_theme>

Now, list the sub-technologies you identified within the theme/industry, along with their corresponding ElasticSearch query strings. Each sub-technology should have its own query string that specifically targets companies related to that sub-technology.

Write the list of 4-7 sub-technologies and their ElasticSearch query strings here in JSON Format, with sub-technologies as keys and query strings as values, using the following format:

<json_technology>
{{"sub-technology": Query string for technology,..}}
</json_technology>

Finally, list 4-7 domains of application you identified for the theme/industry, along with their corresponding ElasticSearch query strings. Each domain of application should have its own query string that specifically targets companies operating in that domain.

Write the list of domains of application and their ElasticSearch query strings here in JSON Format, with domains of applications as keys and query strings as values, using the following format:

<json_industries>
{{"domain of application": Query string for domain of application,..}}
</json_industries>

IMPORTANT REMARKS:
- Query strings should be strings, not JSON objects or dictionaries.
- Query string making instructions:
  - Keywords inside query strings are better in singular form and as one token.
  - Account for abbreviations and alternative semantic keywords to cover as much ground as possible, while certainly staying inside the specified theme.
  - Make the queries as relaxed as possible to retrieve the maximum number of related companies.
- It is mandatory to respect the <Tags> in the output format.

Remember to provide your analysis and reasoning in the <scratchpad> section before presenting the final outputs. The ElasticSearch query strings should be designed to effectively search for companies based on their LinkedIn descriptions and website content.""")

def remove_outer_quotes(query):
    # Regular expression to match quotes at the beginning and end of the string
    pattern = r'^"([^"]*)"$'
    match = re.match(pattern, query)
    if match:
        return match.group(1)
    return query

def extract_content_from_tags(
    text: str, tags: List[str], labels: List[str], json_output: List[bool]
) -> dict:
    from json_repair import json_repair
 
    """Extract JSONs from a string.
    text : text to extract jsons from
    tags : list of tags incorporating the JSONs
    labels : list of labels for the JSONs in the output
    """
    output_jsons = {}
    for label, tag, json_boolean in zip(labels, tags, json_output):
        pattern = re.compile(f"<{tag}>(.*?)</{tag}>", re.DOTALL)
        search = pattern.search(text)
        if search:
            if json_boolean:
                output_jsons[label] = json_repair.loads(
                    search.group(1).strip()
                )
                for key, value in output_jsons[label].items():
                    if value.count('"') > 3:
                        # Check if the value starts and ends with a quote
                        if not value.startswith('"') and "(" not in value:
                            value = '"' + value
                        if not value.endswith('"') and "(" not in value:
                            value = value + '"'
 
                    # value = quote_keywords(value)
                    output_jsons[label][key] = value
 
            else:
                output_jsons[label] = remove_outer_quotes(
                    search.group(1).strip()
                )
    return output_jsons

market_keywords_chain = (
    (
        market_segmentation_prompt
        |gpt_4_o
        |StrOutputParser()
        |RunnableLambda(
            lambda x: extract_content_from_tags(
                text=x,
                tags=["json_technology", "json_industries", "es_query_theme"],
                labels=[
                    "technology_queries",
                    "industries_queries",
                    "theme_query",
                ],
                json_output=[True, True, False],
            )
        )
    )
    .with_config({"run_name": "Market Segmentation"})
)

def get_theme_recommandations(theme):
    
    keywords = market_keywords_chain.invoke({"theme": theme})
    theme_query = keywords["theme_query"]
    tech_query = keywords["technology_queries"]
    t0 = theme_query
    n = ""
    for k in keywords['technology_queries'].items():

        n =  k[1] + " OR "+ k[0] + " OR " + n
    
    technology_query_string = n[-3]
    
    query = {
        "bool": {
            "must": [
                {
                    "query_string": {
                        "query": theme_query,
                        "fields": [
                            "website_content",
                            "website_title",
                            "website_description",
                            "website_keywords",
                            "description",
                            "short_description",
                            "tech_industries",
                            "specialties",
                            #"patents.title",
                            #"patents.top_terms",
                        ],
                    }
                },
                {
                    "query_string": {
                        "query": technology_query_string,
                        "fields": [
                            "website_content",
                            "website_title",
                            "website_description",
                            "website_keywords",
                            "description",
                            "short_description",
                            "tech_industries",
                            "specialties",
                            "patents.title",
                            "patents.top_terms",
                        ],
                    }
                },
            ]
        }
    }
    
    companiess =  get_companies(query)
    items = {
    'query': f'How much is this company related to {theme}, it has to be the main focus of the company ?',
    'hits': []
    }
    for i in range(0,len(companiess['companies']) ):
        item_i = {'content': f'name : {companiess["companies"][i]["name"]}, description : {companiess["companies"][i]["description"]}, alpha_id:{companiess["companies"][i]["alpha_id"]}'}
        items['hits'].append(item_i)
    return items, pd.DataFrame(companiess["companies"])


def get_theme_recommandations_no_tech(theme):
    
    keywords = market_keywords_chain.invoke({"theme": theme})
    theme_query = keywords["theme_query"]
    
    query = {
        "bool": {
            "must": [
                {
                    "query_string": {
                        "query": theme_query,
                        "fields": [
                            "website_content",
                            "website_title",
                            "website_description",
                            "website_keywords",
                            "description",
                            "short_description",
                            "tech_industries",
                            "specialties",
                            #"patents.title",
                            #"patents.top_terms",
                        ],
                    }
                }
            ]
        }
    }

    
    companiess =  get_companies(query)
    items = {
    'query': f'How much is this company related to {theme}, it has to be the main focus of the company ?',
    'hits': []
    }
    for i in range(0,len(companiess['companies']) ):
        item_i = {'content': f'name : {companiess["companies"][i]["name"]}, description : {companiess["companies"][i]["description"]}, alpha_id:{companiess["companies"][i]["alpha_id"]}'}
        items['hits'].append(item_i)
    return items, pd.DataFrame(companiess["companies"])

def get_permutations(item, model, tokenizer, df_es, x=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reranked_data = []
    permutations = []
    model = model.to(device)
    
    q = item['query']
    passages = [psg['content'] for i, psg in enumerate(item['hits'])][:x]

    if len(passages) == 0:
        reranked_data.append(item)
        return reranked_data, permutations
    
    # Tokenize on CPU
    features = tokenizer(
        [q] * len(passages),
        passages,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=512  # Adjust as necessary
    )

    # Move tokenized inputs to GPU
    features = {key: val.to(device) for key, val in features.items()}
    
    with torch.no_grad():
        scores = model(**features).logits
        normalized_scores = [float(score[0]) for score in scores.cpu()]  # Move scores back to CPU for processing
    
    
    ranked = np.argsort(normalized_scores)[::-1]
    response = ' > '.join([str(ss + 1) for ss in ranked])
    
    new_order = [int(x)-1 for x in response.split(">")]

    # Reorganize DataFrame columns based on the new order
    new_df_es = df_es.head(x)
    df_reordered = new_df_es.iloc[new_order]
    
    
    
    return df_reordered.reset_index(drop=True)


    
    

def clean_response(response: str):
    new_response = ''
    for c in response:
        if not c.isdigit():
            new_response += ' '
        else:
            new_response += c
    new_response = new_response.strip()
    return new_response


def remove_duplicate(response):
    new_response = []
    for c in response:
        if c not in new_response:
            new_response.append(c)
    return new_response


def receive_permutation(item, permutation, rank_start=0, rank_end=100):
    response = clean_response(permutation)
    response = [int(x) - 1 for x in response.split()]
    response = remove_duplicate(response)
    cut_range = deepcopy(item['hits'][rank_start: rank_end])
    original_rank = [tt for tt in range(len(cut_range))]
    response = [ss for ss in response if ss in original_rank]
    response = response + [tt for tt in original_rank if tt not in response]
    for j, x in enumerate(response):
        item['hits'][j + rank_start] = deepcopy(cut_range[x])
        if 'rank' in item['hits'][j + rank_start]:
            item['hits'][j + rank_start]['rank'] = cut_range[j]['rank']
        if 'score' in item['hits'][j + rank_start]:
            item['hits'][j + rank_start]['score'] = cut_range[j]['score']
    return item



