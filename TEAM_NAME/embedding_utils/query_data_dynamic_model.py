import argparse
import os
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from get_embedding_function import get_embedding_function
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.schema import Document
import csv

PROMPT_TEMPLATE = """
Answer the question using only the context provided below. If the context clearly does not contain any relevant information, respond with 'Context not found.'

Context:
{context}

---

Question:
{question}
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
    os.makedirs(os.path.dirname(csv_path), exist_ok=True) if os.path.dirname(csv_path) else None

    row = {
        "query": query,
        "embedding_model": embedding_model,
        "chroma_path": chroma_path,
        "collection_name": collection_name,
        "llm_model": llm_model,
        "llm_output": llm_output,
        "sources": ";".join([str(s) for s in sources])
    }

    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

#def transform_query(query: str) -> str:
   # prompt = PromptTemplate.from_template(
        #"Rewrite the following query to improve retrieval from clinical SOPs: {query}"
    )
    #model = OllamaLLM(model="llama3.1")
   # chain = LLMChain(prompt=prompt, llm=model)
   # return chain.run({"query": query})

def query_rag(query_text: str, chroma_path: str, llm_model: str, log_csv: str | None = None):
    query_text = transform_query(query_text)

    embedding_model = os.path.basename(chroma_path)
    safe_model = embedding_model.replace(":", "_")
    collection_name = f"ADLM_Embeddings_{safe_model}"
    embedding_function = get_embedding_function(embedding_model)

    db = Chroma(
        collection_name=collection_name,
        persist_directory=chroma_path,
        embedding_function=embedding_function
    )

    dense_retriever = db.as_retriever(search_kwargs={"k": 5})

    try:
        all_docs = db.get(include=["documents"])["documents"]
        all_docs = [Document(page_content=doc) for doc in all_docs]
        sparse_retriever = BM25Retriever.from_documents(all_docs)
        sparse_retriever.k = 5

        hybrid_retriever = EnsembleRetriever(
            retrievers=[dense_retriever, sparse_retriever],
            weights=[0.5, 0.5]
        )

        results = hybrid_retriever.get_relevant_documents(query_text)
        results = [(doc, None) for doc in results]

    except Exception as e:
        print("⚠️ Hybrid search fallback to dense only due to error:", e)
        results = db.similarity_search_with_score(query_text, k=5)

    if not results:
        print("⚠️ No relevant documents found for your query!")
        return "No relevant context available."

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    model = OllamaLLM(model=llm_model)
    chain = prompt_template | model

    print("Response: ", end="", flush=True)
    response_chunks = []
    for chunk in chain.stream({"context": context_text, "question": query_text}):
        print(chunk, end="", flush=True)
        response_chunks.append(chunk)
    print()

    response_text = "".join(response_chunks)
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    print(f"Sources: {sources}")

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
