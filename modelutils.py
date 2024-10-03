from dotenv import load_dotenv
import os
import pandas as pd
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()  

huggingface_token=os.getenv("HUGGINGFACE_TOKEN")
# snapshot_download(repo_id='llmunlearningsemeval2025organization/olmo-finetuned-semeval25-unlearning', token=huggingface_token, local_dir='semeval25-unlearning-model')
model = AutoModelForCausalLM.from_pretrained('semeval25-unlearning-model')
tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-7B-0724-Instruct-hf")
# snapshot_download(repo_id='llmunlearningsemeval2025organization/semeval25-unlearning-dataset-public', token=huggingface_token, local_dir='semeval25-unlearning-data', repo_type="dataset")
retain_train_df = pd.read_parquet('semeval25-unlearning-data/data/retain_train-00000-of-00001.parquet', engine='pyarrow') # Retain split: train set
retain_validation_df = pd.read_parquet('semeval25-unlearning-data/data/retain_validation-00000-of-00001.parquet', engine='pyarrow') # Retain split: validation set
forget_train_df = pd.read_parquet('semeval25-unlearning-data/data/forget_train-00000-of-00001.parquet', engine='pyarrow') # Forget split: train set
forget_validation_df = pd.read_parquet('semeval25-unlearning-data/data/forget_validation-00000-of-00001.parquet', engine='pyarrow') # Forget split: validation set