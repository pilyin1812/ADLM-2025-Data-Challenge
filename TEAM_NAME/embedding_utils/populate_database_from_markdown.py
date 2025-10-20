import argparse
import os
import shutil
# from langchain.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
# from langchain_text_splitters import TextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding_function import get_embedding_function
from langchain_chroma import Chroma

CHROMA_PATH = "TEAM_NAME/chroma"
DATA_PATH = "TEAM_NAME/Markdown-Output"

# --- Custom splitter for ## headers ---
# class MarkdownHeaderSplitter(TextSplitter):
#     """Splits markdown content at '## ' headers."""

#     def __init__(self, chunk_overlap: int = 0):
#         self.chunk_overlap = chunk_overlap

#     def split_text(self, text: str):
#         chunks = []
#         lines = text.splitlines()
#         current_chunk = []

#         for line in lines:
#             if line.startswith("## ") and current_chunk:
#                 # start a new chunk
#                 chunks.append("\n".join(current_chunk))
#                 # optionally include overlap
#                 current_chunk = current_chunk[-self.chunk_overlap:] if self.chunk_overlap else []
#             current_chunk.append(line)

#         if current_chunk:
#             chunks.append("\n".join(current_chunk))

#         return chunks

# -------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    parser.add_argument("--model", type=str, default="llama3.1", help="Embedding model to use")
    args = parser.parse_args()

    if args.reset:
        print("✨ Clearing Database")
        clear_database(args.model)

    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks, args.model)

def load_documents():
    # Load only markdown files
    loader = DirectoryLoader(
        DATA_PATH,
        glob="**/*.md",
        loader_cls=UnstructuredMarkdownLoader,
    )
    return loader.load()

# def split_documents(documents: list[Document]):
#     splitter = MarkdownHeaderSplitter(chunk_overlap=0)
#     all_chunks = []
#     for doc in documents:
#         text_chunks = splitter.split_text(doc.page_content)
#         for chunk_text in text_chunks:
#             all_chunks.append(Document(
#                 page_content=chunk_text,
#                 metadata=doc.metadata.copy()
#             ))
#     return all_chunks

def split_documents(documents: list[Document]):
  text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=1000,
      chunk_overlap = 150,
      length_function=len,
      is_separator_regex=False,
  )
  return text_splitter.split_documents(documents)

def add_to_chroma(chunks: list[Document], model_name: str):
    # Make DB folder specific to the embedding model
    model_db_path = os.path.join(CHROMA_PATH, model_name.replace(":", "_"))

    # Include model name in collection_name
    collection_name = f"ADLM_Embeddings_{model_name.replace(':', '_')}"

    db = Chroma(
        collection_name=collection_name,
        persist_directory=model_db_path,
        embedding_function=get_embedding_function(model_name)
    )

    chunks_with_ids = calculate_chunk_ids(chunks)
    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB ({model_name}): {len(existing_ids)}")

    new_chunks = [chunk for chunk in chunks_with_ids if chunk.metadata["id"] not in existing_ids]

    if len(new_chunks):
        print(f"👉 Adding new documents for {model_name}: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        # db.persist()
    else:
        print(f"✅ No new documents to add for {model_name}")


def calculate_chunk_ids(chunks):
    last_source = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        current_page_id = f"{source}"

        if current_page_id == last_source:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_source = current_page_id
        chunk.metadata["id"] = chunk_id

    return chunks


def clear_database(model_name: str):
    model_db_path = os.path.join(CHROMA_PATH, model_name.replace(":", "_"))
    if os.path.exists(model_db_path):
        shutil.rmtree(model_db_path)

if __name__ == "__main__":
    main()

# EXAMPLE USAGE
# python .\py\populate_database_from_markdown.py --model llama3.1
# python .\py\populate_database_from_markdown.py --model nomic-embed-text
