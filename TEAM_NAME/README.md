# Workflow

## Convert PDFs to Markdown

-   Procedures: `procedures-to-markdown.py`
-   FDA Docs:

## Create vector databases

-   `python TEAM_NAME\embedding_utils\populate_database_from_markdown_batch.py --model all-minilm --batch-size 5000 --persist-every 10`

-   `python TEAM_NAME\embedding_utils\populate_database_from_markdown_batch.py --model mxbai-embed-large --batch-size 5000 --persist-every 10`

-   `python TEAM_NAME\embedding_utils\populate_database_from_markdown_batch.py --model nomic-embed-text --batch-size 5000 --persist-every 10`

-   `python TEAM_NAME\embedding_utils\populate_database_from_markdown_batch.py --model bge-m3 --batch-size 5000 --persist-every 10`

Couldn't get these to work:

-   `python TEAM_NAME\embedding_utils\populate_database_from_markdown_batch.py --model snowflake-arctic-embed --batch-size 5000 --persist-every 10`

-   `python TEAM_NAME\embedding_utils\populate_database_from_markdown_batch.py --model bge-m3 --batch-size 5000 --persist-every 10`

-   `python TEAM_NAME\embedding_utils\populate_database_from_markdown_batch.py --model EmbeddingGemma --batch-size 5000 --persist-every 10`

## Query in Terminal

-   `python TEAM_NAME/embedding_utils/query_data_dynamic_model.py "explain the protocol for the confirmation of CarboxyTHC in meconium samples" --chroma-path TEAM_NAME/chroma/all-minilm --llm-model llama3.1`

## Query with QC App

-   `streamlit run TEAM_NAME/embedding_utils/QC-App.py`

# Working with UV

-   Install UV: `pip install uv`
-   cd into the Team directory: `cd TEAM_NAME`
-   activate the virtual env: `.venv\Scripts\activate` (optional: find the venv on your machine for interpreter if needed `dir .venv\Scripts\python.exe`)
-   install needed packages: `uv sync`
-   To add a new package: `uv add package_name` (optionally `uv add package_name --native-tls`)

# Working with Ollama

-   Download Ollama: https://ollama.com/
-   Install models if needed: `ollama pull model-name`
