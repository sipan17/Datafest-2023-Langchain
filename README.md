# Datafest 2023: LangChain Chatbot
Feel free to modify and experiment with any step in the pipeline.

### Unzip transcription data
```bash
unzip data.zip
```

### Create a conda environment
```bash
conda create -n langchain_chatbot python=3.9 -y
conda activate langchain_chatbot
```

### Install the requirements
```bash
pip install -r requirements.txt
```

#### Add any keys required to the environment

For example, for using gpt-4 you need to create OPENAI_API_KEY and export it.
```bash
export OPENAI_API_KEY=...
```

### Embed the source documents and store
```bash
python create_db.py
```

### Run the chatbot
```bash
python chatbot.py
```