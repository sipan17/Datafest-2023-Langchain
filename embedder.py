import os

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings

from config import config

if config["model_type"] == "huggingface":
    embedding = HuggingFaceEmbeddings(
        model_name=config["model_name"],
        cache_folder=os.path.join(config["workdir"], "embedder_cache", config["model_name_modified"])
    )
elif config["model_type"] == "openai":
    embedding = OpenAIEmbeddings(model=config["model_name"])
else:
    raise ValueError(f"Model type {config['model_type']} is not supported. Yet)")
