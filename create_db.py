import os
import glob

from langchain.vectorstores import Chroma
from langchain.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import config
from embedder import embedding


def metadata_func(record: dict, metadata: dict) -> dict:
    metadata["chapter_name"] = record.get("chapter_name")
    metadata["speaker"] = record.get("speaker")
    
    return metadata


def parse_source(path):
    filename = path.split("/")[-1].split(".json")[0]
    episode_number, name = filename.split("-", maxsplit=1)
    if not episode_number.isnumeric():
        episode_number = "None"
        name = filename
    name = name.replace("-", " ").title()
    return episode_number, name


def main():
    json_paths = glob.glob(f"{config['json_dir']}/*json")
    
    # Load jsons
    loaders = [
        JSONLoader(file_path=path, jq_schema=".chapters[].content[]", content_key="speech",
                   metadata_func=metadata_func, text_content=False) for path in json_paths
    ]
    
    docs = []
    for loader in loaders:
        speeches = loader.load()
        for speech in speeches:
            metadata = speech.metadata
            episode_number, episode_name = parse_source(metadata["source"])
            speech.page_content = (f"In the episode number `{episode_number}` named `{episode_name}` "
                                   f"{metadata['speaker']} said `{speech.page_content}`")  # enrich the content
            docs.append(speech)
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200
    )
    
    splits = text_splitter.split_documents(docs)
    print(len(splits))
    
    # persist_directory = os.path.join(config["workdir"], "chroma_dbs", config["model_name_modified"])
    #
    # vectordb = Chroma.from_documents(
    #     documents=splits,
    #     embedding=embedding,
    #     persist_directory=persist_directory
    # )
    # vectordb.persist()


if __name__ == "__main__":
    main()
