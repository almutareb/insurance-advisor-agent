import datasets
from langchain_core.vectorstores import VectorStore
from typing import Optional
import json
from tqdm.auto import tqdm
from reader import answer_with_rag, answer_with_rerank_rag
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()


def run_rag_tests(
    eval_dataset: datasets.Dataset,
    llm,
    knowledge_index: VectorStore,
    output_file: str,
    # reranker: Optional[RAGPretrainedModel] = None,
    verbose: Optional[bool] = True,
    test_settings: Optional[str] = None,  # To document the test settings used
):
    """Runs RAG tests on the given dataset and saves the results to the given output file."""
    try:  # load previous generations if they exist
        with open(output_file, "r") as f:
            outputs = json.load(f)
    except:
        outputs = []

    for example in tqdm(eval_dataset):
        question = example["question"]
        if question in [output["question"] for output in outputs]:
            continue

        answer, relevant_docs = answer_with_rag(question=question, llm=llm, knowledge_index=knowledge_index)
        if verbose:
            print("=======================================================")
            print(f"Question: {question}")
            print(f"Answer: {answer}")
            print(f'True answer: {example["answer"]}')
        result = {
            "question": question,
            "true_answer": example["answer"],
            "source_doc": example["source_doc"],
            "generated_answer": answer,
            "retrieved_docs": [doc for doc in relevant_docs],
        }
        if test_settings:
            result["test_settings"] = test_settings
        outputs.append(result)

        with open(output_file, "w") as f:
            json.dump(outputs, f)


def run_rag_rerank_tests(
    eval_dataset: datasets.Dataset,
    llm,
    knowledge_index: VectorStore,
    output_file: str,
    hf_api_key:str,
    verbose: Optional[bool] = True,
    test_settings: Optional[str] = None,  # To document the test settings used
):
    """Runs RAG tests on the given dataset and saves the results to the given output file."""
    
    outputs = []

    for example in tqdm(eval_dataset):
        
        question = example["question"]
        if question in [output["question"] for output in outputs]:
            continue
        try:    
            answer, relevant_docs = answer_with_rerank_rag(question=question, 
                                                    llm=llm, 
                                                    knowledge_index=knowledge_index,
                                                    hf_api_key=hf_api_key)
        except Exception as e:
            print(f'Error in answer_with_rerank_rag() for question: {question}')
            print(f'Error details: {str(e)}')
            continue
        

        if verbose:
            print("=======================================================")
            print(f"Question: {question}")
            print(f"Answer: {answer}")
            print(f'True answer: {example["answer"]}')
        result = {
            "question": question,
            "true_answer": example["answer"],
            "source_doc": example["source_doc"],
            "generated_answer": answer,
            "retrieved_docs": [doc for doc in relevant_docs],
        }
        
        if test_settings:
            result["test_settings"] = test_settings

        outputs.append(result)
        with open(output_file, "w") as f:
            json.dump(outputs, f, indent=2)

        
        


if __name__ == "__main__":
    
    # Load the Q&A from file
    with open(file='critique_outputs.json', mode ='r', encoding='utf-8') as f:
        crit_outputs = json.load(f)

    generated_questions = pd.DataFrame.from_dict(crit_outputs)

    # Filter out the best questions and answers based on score
    generated_questions = generated_questions.loc[
    (generated_questions["groundedness_score"] >= 4)
    & (generated_questions["relevance_score"] >= 4)
    & (generated_questions["standalone_score"] >= 4)]

    # Put it in a HuggingFace dataset
    eval_dataset = datasets.Dataset.from_pandas(generated_questions, split="train", preserve_index=False)
    # # Get the vectorstore
    HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')
    embeddings = HuggingFaceInferenceAPIEmbeddings(api_key=HUGGINGFACEHUB_API_TOKEN,
                                                   model_name="thenlper/gte-small")
    loaded_vectorstore = FAISS.load_local(folder_path="vectorstore_db", 
                                        embeddings=embeddings, 
                                        allow_dangerous_deserialization=True)

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

    # run_rag_tests(eval_dataset=eval_dataset,
    #               llm=READER_LLM, 
    #               knowledge_index=loaded_vectorstore, 
    #               output_file='test_rag.json')
    
    
    # from rag_app.reranking import get_reraked_docs
    # re_ranked_docs = get_reraked_docs(query='Do you offer boat insurance?',
    #                                   vectorstore=loaded_vectorstore, hf_api_key=HUGGINGFACEHUB_API_TOKEN)
    # print(f"{re_ranked_docs=}")
    # print(f"{len(re_ranked_docs)=}")

    run_rag_rerank_tests(eval_dataset=eval_dataset,
                         llm=READER_LLM,
                         knowledge_index=loaded_vectorstore,
                         output_file='test_rerank_rag.json',
                         hf_api_key=HUGGINGFACEHUB_API_TOKEN,
                         verbose=True,
                         test_settings=False)

