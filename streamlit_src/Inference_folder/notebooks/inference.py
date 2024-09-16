# Databricks notebook source
pip install -r ../requirements_market_seg.txt

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
import sys

notebook_path =  '/Workspace' + os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())
sys.path.append(notebook_path)
sys.path.append(os.path.abspath('../src'))
sys.path.append(os.path.abspath('../configs'))
sys.path.append(os.path.abspath('../../utils'))

# COMMAND ----------

import torch
import mlflow
from settings import *
from rank_gpt import *
from rank_loss import *
from helper_functions import *
from extra_functions import *

# COMMAND ----------

pretrained_model_path = '/dbfs/FileStore/deberta-10k-rank_net/deberta-10k-rank_net'
finetune_model_version_1 = '/dbfs/FileStore/finetuned_deberta-10k-rank_net/checkpoint_epoch_2'
finetuned_model_version_2 = '/dbfs/FileStore/finetuned_2_deberta-50-rank_net/checkpoint_epoch_3'
finetune_model_version_3 = '/dbfs/FileStore/finetuned_3_deberta-50-rank_net/checkpoint_epoch_3'


# COMMAND ----------

dbutils.widgets.text("deployment", "demo-search", label = "Deployment")
dbutils.widgets.dropdown("model_name", finetune_model_version_3, [pretrained_model_path, finetune_model_version_1,finetuned_model_version_2, finetune_model_version_3], label = "Model Name")
dbutils.widgets.dropdown("themes", "Extreme Ultraviolet (EUV) lithography", ["Extreme Ultraviolet (EUV) lithography","heat pumps and energy efficiency", "Optical components for data centers", "Glass used in pharma packaging", "specialty pet food retailers", "Large Language Models for finance"], label = "Themes")
dbutils.widgets.text("experiment_name",f"/Users/charbel@alpha10x.com/dev_theme_Company_Retrieval", label="MLflow experiment name")

# COMMAND ----------

deployment = dbutils.widgets.get("deployment")
model_name = dbutils.widgets.get("model_name")
theme = dbutils.widgets.get("themes")
experiment_name = dbutils.widgets.get("experiment_name")

# COMMAND ----------

model, tokenizer, device = load_model_and_tokenizer(model_name, num_labels= 1, use_fast = True)

# COMMAND ----------

items = get_theme_recommandations(theme)

# COMMAND ----------

ES_list_comp = extract_company_info(items)

# COMMAND ----------

new_items, permutations = get_permutations(items, model, tokenizer, device)

# COMMAND ----------

reranking_list_comp = extract_company_info(new_items[0])

# COMMAND ----------

List_companies = pd.concat([ES_list_comp.add_prefix('ES_'), reranking_list_comp.add_prefix('Rer_')], axis = 1)

# COMMAND ----------

result = (List_companies['ES_alpha_id'] == List_companies['Rer_alpha_id']).all()
print("All ES_alpha_id == Rer_alpha_id:", result)

# COMMAND ----------

model_path = extract_model_name(model_name)

# COMMAND ----------

save_dataframe_to_csv(List_companies, "../../reports", theme, model_path, file_name="List_companies.csv")

# COMMAND ----------

experiment = mlflow.get_experiment_by_name(experiment_name)

if experiment is None:
    mlflow.create_experiment(experiment_name)

# Set the active experiment
mlflow.set_experiment(experiment_name)
with mlflow.start_run(run_name="semantic_search_reranking") as run:
    mlflow.set_tag("model_name", "semantic_search_reranking")
    mlflow.pytorch.log_model(model, "semantic_search_reranking")
    mlflow.log_artifacts("../../reports", artifact_path="semantic_search_reranking/reports")
