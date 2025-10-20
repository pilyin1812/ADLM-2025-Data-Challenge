import argparse
import os
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from get_embedding_function import get_embedding_function
import csv

# {context} → placeholder for the relevant text chunks retrieved from embedding db.
# {question} → placeholder for the user’s query.

PROMPT_TEMPLATE = """
Answer the question using the context below if relevant. If the context does not contain the answer, respond 'Context not found.'

{context}

---

Answer the question based on the above context: {question}
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The question you want to ask.")
    parser.add_argument(
        "--chroma-path",
        type=str,
        required=True,
        help="Path to the Chroma DB folder (e.g., chroma/llama3.1).",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="gemma3",
        help="Ollama LLM model to use for answering queries.",
    )
    parser.add_argument(
        "--log-csv",
        type=str,
        default=None,
        help="Optional path to append results as CSV (e.g., logs/rag_results.csv).",
    )
    args = parser.parse_args()

    query_rag(
        query_text=args.query_text,
        chroma_path=args.chroma_path,
        llm_model=args.llm_model,
        log_csv=args.log_csv
    )



def log_to_csv(
    csv_path: str,
    query: str,
    embedding_model: str,
    chroma_path: str,
    collection_name: str,
    llm_model: str,
    llm_output: str,
    sources: list
):
    # Ensure directory exists
    os.makedirs(os.path.dirname(csv_path), exist_ok=True) if os.path.dirname(csv_path) else None

    # Prepare row
    row = {
        "query": query,
        "embedding_model": embedding_model,
        "chroma_path": chroma_path,
        "collection_name": collection_name,
        "llm_model": llm_model,
        "llm_output": llm_output,
        "sources": ";".join([str(s) for s in sources])  # flatten list
    }

    # Write header if file doesn't exist
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def query_rag(query_text: str, chroma_path: str, llm_model: str, log_csv: str | None = None):
    # --- Determine embedding model from folder name
    embedding_model = os.path.basename(chroma_path)
    safe_model = embedding_model.replace(":", "_")

    # --- Match the naming used in add_to_chroma
    collection_name = f"ADLM_Embeddings_{safe_model}"

    # --- Embedding function (ensure this accepts your folder-derived name)
    embedding_function = get_embedding_function(embedding_model)

    # --- Load Chroma DB
    db = Chroma(
        collection_name=collection_name,
        persist_directory=chroma_path,
        embedding_function=embedding_function
    )

    # After creating `db` in query_rag:
    # try:
    #     # Count docs
    #     all_ids = db.get(include=[])["ids"]
    #     print(f"Documents in collection: {len(all_ids)}")
    #     print("Sample IDs:", all_ids[:5])
    # except Exception as e:
    #     print("Error reading collection:", e)

    # --- Retrieve top-k chunks
    results = db.similarity_search_with_score(query_text, k=5)

    if not results:
        print("⚠️ No relevant documents found for your query!")
        return "No relevant context available."

    # --- Build context string
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    # --- Build chain: Prompt → Model
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    model = OllamaLLM(model=llm_model)
    chain = prompt_template | model  # LCEL chain

    # --- Stream response
    print("Response: ", end="", flush=True)
    response_chunks = []
    for chunk in chain.stream({"context": context_text, "question": query_text}):
        # `chunk` is a string piece from the LLM
        print(chunk, end="", flush=True)
        response_chunks.append(chunk)
    print()  # newline after the streamed output

    response_text = "".join(response_chunks)

    # --- Sources (IDs)
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    print(f"Sources: {sources}")

    # --- Log to CSV
    if log_csv:
        log_to_csv(
            csv_path=log_csv,
            query=query_text,
            embedding_model=embedding_model,
            chroma_path=chroma_path,
            collection_name=collection_name,
            llm_model=llm_model,
            llm_output=response_text,
            sources=sources
        )

    return response_text


if __name__ == "__main__":
    main()

# EXAMPLE USAGE
# python py/query_data_dynamic_model.py "Using the ADLM Embedding knowledge store, explain the protocol for the confirmation of CarboxyTHC in meconium samples" --chroma-path chroma/llama3.1 --llm-model llama3.1
