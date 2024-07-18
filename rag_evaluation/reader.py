from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.vectorstores import VectorStore
from langchain_core.language_models.llms import LLM
from prompts import RAG_PROMPT_TEMPLATE
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')


repo_id = "HuggingFaceH4/zephyr-7b-beta"


READER_LLM = HuggingFaceEndpoint(
    repo_id=repo_id,
    task="text-generation",
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
    max_new_tokens=512,
    top_k=30,
    temperature=0.1,
    repetition_penalty=1.03
)

def answer_with_rag(
    question: str,
    llm: LLM,
    knowledge_index: VectorStore,
    num_retrieved_docs: int = 30,
    num_docs_final: int = 7) -> tuple[str, list]:
    """Answer a question using RAG with the given knowledge index."""
    # Gather documents with retriever
    
    relevant_docs = knowledge_index.similarity_search(query=question, k=num_retrieved_docs)
    relevant_docs = [doc.page_content for doc in relevant_docs]  # keep only the text
    
    relevant_docs = relevant_docs[:num_docs_final]

    # Build the final prompt
    context = "\nExtracted documents:\n"
    context += "".join([f"Document {str(i)}:::\n" + doc for i, doc in enumerate(relevant_docs)])

    final_prompt = RAG_PROMPT_TEMPLATE.format(question=question, context=context)

    # Redact an answer
    answer = llm.invoke(final_prompt)

    return answer, relevant_docs

if __name__ == '__main__':

    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=HUGGINGFACEHUB_API_TOKEN,
        model_name="thenlper/gte-small")
    loaded_vectorstore = FAISS.load_local(folder_path="website_docs/test", embeddings=embeddings, allow_dangerous_deserialization=True)

    answer, relevant_docs = answer_with_rag(question='When can I cancel my policy',
                                            llm=READER_LLM,
                                            knowledge_index=loaded_vectorstore)
    print(f"{answer=}\n\n{relevant_docs=}")