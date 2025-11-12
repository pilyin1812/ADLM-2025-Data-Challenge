# Workflow

## Working with UV

-   Install UV: `pip install uv`
-   cd into the Team directory: `cd TEAM_NAME`
-   activate the virtual env: `.venv\Scripts\activate` (optional: find the venv on your machine for interpreter if needed `dir .venv\Scripts\python.exe`)
-   install needed packages: `uv sync`
-   To add a new package: `uv add package_name` (optionally `uv add package_name --native-tls`)
  
- Quick Start when reopening terminal  `cd TEAM_NAME; .\.venv\Scripts\activate`

## Working with Ollama

-   Download Ollama: https://ollama.com/
-   Install models in terminal: `ollama pull gpt-oss:20b` `ollama pull qwen3-embedding`
If using alternative models replace model-name with your intended model `ollama pull model-name`
- 
## Load LabDocs to IDE
Windows Command Line
`curl -L -C - -o LabDocs.zip \
    "https://zenodo.org/records/16328490/files/LabDocs.zip?download=1"
tar -xf LabDocs.zip`
## Convert PDFs to Markdown
Run each line in order in terminal:
-   Synthetic Procedures:`python embedding_utils/Synthetic_procedures-to-markdown.py`
-   FDA Docs:
  - `python embedding_utils/FDA-sorting.py`
  - `python embedding_utils/FDA_REVIEW-Docs-to-markdown.py`
  - `python embedding_utils/OCR_test.py`
## Create vector databases
-   `python TEAM_NAME\embedding_utils\populate_database_from_markdown_batch.py --model qwen3-embedding --batch-size 128 --persist-every 10`
Alternative models:
-   `python TEAM_NAME\embedding_utils\populate_database_from_markdown_batch.py --model all-minilm --batch-size 128 --persist-every 10`

-   `python TEAM_NAME\embedding_utils\populate_database_from_markdown_batch.py --model mxbai-embed-large --batch-size 128 --persist-every 10`

-   `python TEAM_NAME\embedding_utils\populate_database_from_markdown_batch.py --model nomic-embed-text --batch-size 128 --persist-every 10`

-   `python TEAM_NAME\embedding_utils\populate_database_from_markdown_batch.py --model bge-m3 --batch-size 128 --persist-every 10`

Couldn't get these to work:

-   `python TEAM_NAME\embedding_utils\populate_database_from_markdown_batch.py --model snowflake-arctic-embed --batch-size 128 --persist-every 10`

-   `python TEAM_NAME\embedding_utils\populate_database_from_markdown_batch.py --model bge-m3 --batch-size 128 --persist-every 10`

-   `python TEAM_NAME\embedding_utils\populate_database_from_markdown_batch.py --model EmbeddingGemma --batch-size 128 --persist-every 10`
## Query with QC App
-   `streamlit run embedding_utils/QC-App.py`
## Query in Terminal
Replace embeddingmodel and llmmodel with the names of your embedding model and llm model.
-   `python TEAM_NAME/embedding_utils/query_data_dynamic_model.py "explain the protocol for the confirmation of CarboxyTHC in meconium samples" --chroma-path chroma/embeddingmodel --llm-model llmmodel`