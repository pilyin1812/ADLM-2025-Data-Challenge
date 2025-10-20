import argparse
import os
import shutil
import subprocess
import sys
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding_function import get_embedding_function
from langchain_chroma import Chroma


CHROMA_PATH = "TEAM_NAME/chroma"
DATA_PATH = "TEAM_NAME/Markdown-Output"

# Reasonable defaults to avoid Chroma's hard limit (~5461)
DEFAULT_BATCH_SIZE = 5000
DEFAULT_PERSIST_EVERY = 10  # persist every N batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    parser.add_argument("--model", type=str, default="llama3.1", help="Embedding model to use")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Number of docs to upsert per call (<= ~5461). Default {DEFAULT_BATCH_SIZE}.",
    )
    parser.add_argument(
        "--persist-every",
        type=int,
        default=DEFAULT_PERSIST_EVERY,
        help=f"Call persist() every N batches to avoid data loss on long runs. Default {DEFAULT_PERSIST_EVERY}.",
    )

    args = parser.parse_args()

    if not check_ollama_model_downloaded(args.model):
        print("❌ Exiting because the Ollama model is not available locally.")
        sys.exit(1)

    if args.reset:
        print("✨ Clearing Database")
        clear_database(args.model)

    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks, args.model, batch_size=args.batch_size, persist_every=args.persist_every)

def check_ollama_model_downloaded(model_name: str) -> bool:
    """
    Checks if the specified Ollama model is downloaded locally.
    Prints success or failure message and returns True/False.
    """
    base_name = model_name.split(":")[0]  # ignore tag if present
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=True
        )
    except FileNotFoundError:
        print("❌ Ollama CLI not found. Install from https://ollama.com and ensure it's on your PATH.")
        return False
    except subprocess.CalledProcessError:
        print("❌ Failed to query Ollama. Is the Ollama daemon running?")
        return False

    # Parse available models
    lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
    available_models = {line.split()[0].split(":")[0] for line in lines[1:]}  # skip header

    if base_name in available_models:
        print(f"✅ Ollama model '{model_name}' is downloaded locally.")
        return True
    else:
        print(f"⚠️  Ollama model '{model_name}' is NOT downloaded.")
        print(f"💡 To download:  ollama pull {model_name}")
        return False

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
        chunk_overlap=150,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def add_to_chroma(chunks: list[Document], model_name: str, batch_size: int = DEFAULT_BATCH_SIZE, persist_every: int = DEFAULT_PERSIST_EVERY):
    # Make DB folder specific to the embedding model
    model_db_path = os.path.join(CHROMA_PATH, model_name.replace(":", "_"))

    # Include model name in collection_name
    collection_name = f"ADLM_Embeddings_{model_name.replace(':', '_')}"

    db = Chroma(
        collection_name=collection_name,
        persist_directory=model_db_path,
        embedding_function=get_embedding_function(model_name),
    )

    # Compute stable IDs for chunks
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Fetch existing IDs (empty include to avoid pulling embeddings/vectors)
    # limit=None should return all; wrapper supports it.
    existing_items = db.get(include=[], limit=None)
    existing_ids = set(existing_items.get("ids", []))
    print(f"🔍 Number of existing documents in DB ({model_name}): {len(existing_ids):,}")

    # Filter new chunks
    new_chunks = [chunk for chunk in chunks_with_ids if chunk.metadata["id"] not in existing_ids]

    if not new_chunks:
        print(f"✅ No new documents to add for {model_name}")
        return

    print(f"👉 Adding new documents for {model_name}: {len(new_chunks):,}")

    # Enforce a safe ceiling just in case
    MAX_SAFE = 5461
    if batch_size > MAX_SAFE:
        print(f"⚠️ Requested batch_size={batch_size} > {MAX_SAFE}. Using {MAX_SAFE}.")
        batch_size = MAX_SAFE
    new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
    total = len(new_chunks)
    batches = (total + batch_size - 1) // batch_size

    try:
        for b in range(batches):
            start = b * batch_size
            end = min(start + batch_size, total)
            batch_docs = new_chunks[start:end]
            batch_ids = new_chunk_ids[start:end]

            # Perform the upsert for this batch
            db.add_documents(batch_docs, ids=batch_ids)

            # Periodic persist for resiliency in long runs
            if persist_every > 0 and (b + 1) % persist_every == 0 and hasattr(db, "persist"):
                db.persist()

            print(f"✔ Upserted {end:,}/{total:,} docs "
                  f"(batch {b + 1}/{batches}, size {end - start})")
    finally:
        # Always persist at the end even if there was an exception
        if hasattr(db, "persist"):
            db.persist()
        print("💾 Persisted vector store to disk.")

    print("✅ Finished adding documents.")


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
# python .\py\populate_database_from_markdown_batch.py --model llama3.1
# python .\py\populate_database_from_markdown_batch.py --model nomic-embed-text

# You can also tune batch/persist:
# python TEAM_NAME\embedding_utils\populate_database_from_markdown_batch.py --model all-minilm --batch-size 5000 --persist-every 10
