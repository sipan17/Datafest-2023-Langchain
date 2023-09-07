import os

here = os.path.dirname(__file__)

config = {
    "json_dir": os.path.join(here, "data"),
    "workdir": here,
    "model_type": "huggingface",
    "model_name": "BAAI/bge-base-en",
    "retriever_kwargs": {
        "search_type": "mmr",
        "search_kwargs": {'k': 5, 'fetch_k': 15},
    },
    "llm_kwargs": {
        "temperature": 0,
    }
}

config["model_name_modified"] = config["model_name"].replace('/', '_')  # to evade creating nested paths,
# happens when model name contains "/"
