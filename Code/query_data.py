import argparse
import os
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from get_embedding_function import get_embedding_function


# {context} → placeholder for the relevant text chunks retrieved from embedding db.
# {question} → placeholder for the user’s query.

PROMPT_TEMPLATE = """
Answer the question using the context below if relevant. If the context doesn’t contain the answer, respond 'Context not found.'

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
    args = parser.parse_args()

    query_rag(
        query_text=args.query_text,
        chroma_path=args.chroma_path,
        llm_model=args.llm_model
    )

# def query_rag(query_text: str, chroma_path: str, llm_model: str):
#     # Prepare the DB
#     embedding_model = os.path.basename(chroma_path)
#     embedding_function = get_embedding_function(embedding_model)
#     db = Chroma(persist_directory=chroma_path, embedding_function=embedding_function)

#     # Search the DB
#     results = db.similarity_search_with_score(query_text, k=5)

#     # BUILD THE PROMPT
#     # results → the top-k chunks retrieved from Chroma for the query.
#     # context_text → concatenates all these chunks into a single string, separated by \n\n---\n\n.
#     context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
#     prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
#     prompt = prompt_template.format(context=context_text, question=query_text)

#     # Query LLM
#     model = OllamaLLM(model=llm_model)
#     response_text = model.invoke(prompt) #sends the fully formatted prompt to the LLM.

#     # Collect sources
#     sources = [doc.metadata.get("id", None) for doc, _score in results]
#     formatted_response = f"Response: {response_text}\nSources: {sources}"
#     print(formatted_response)
#     return response_text

def query_rag(query_text: str, chroma_path: str, llm_model: str):
    # --- Determine embedding model from folder name
    embedding_model = os.path.basename(chroma_path)
    embedding_function = get_embedding_function(embedding_model)

    # --- Load Chroma DB
    db = Chroma(persist_directory=chroma_path, embedding_function=embedding_function)

    # --- Retrieve top-k chunks
    results = db.similarity_search_with_score(query_text, k=50)

    if not results:
        print("⚠️ No relevant documents found for your query!")
        return "No relevant context available."

    # --- Debug: print top retrieved chunks
    print("Top retrieved chunks:", flush=True)
    for doc, score in results:
        print(f"Score: {score:.4f}", flush=True)
        snippet = doc.page_content.replace("\n", " ")[:200]
        print(f"Text snippet: {snippet}", flush=True)
        print("---", flush=True)

    # --- Build context string
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    # --- Fill the prompt template
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # --- Query the LLM
    model = OllamaLLM(model=llm_model)
    response_text = model.invoke(prompt)

    # --- Collect sources
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text

if __name__ == "__main__":
    main()

# EXAMPLE USAGE
# python py/query_data_dynamic_model.py "Using the ADLM Embedding knowledge store, explain the protocol for the confirmation of CarboxyTHC in meconium samples" --chroma-path chroma/llama3.1 --llm-model llama3.1
