from huggingface_hub import InferenceClient
import json
from dotenv import load_dotenv
import os
from prompts import QA_generation_prompt
from chunking import get_chunks, pdf_source
from langchain.docstore.document import Document
from tqdm.auto import tqdm

load_dotenv()

def call_llm(inference_client: InferenceClient, prompt: str):
    response = inference_client.post(
        json={
            "inputs": prompt,
            "parameters": {"max_new_tokens": 1000, "return_full_text":False},
            "task": "text-generation",
        },
    )
    return json.loads(response.decode())[0]["generated_text"]


def get_qa_couple_outputs(chunks:list[Document], inference_client: InferenceClient) -> None:
    """ Generates Questions & Answer outputs and saves it to a json locally

        Args:
            chunks (list): List of Document chunks
    """
    outputs = []
    for chunk in tqdm(chunks):
        output_QA_couple = call_llm(inference_client=inference_client,
                                    prompt=QA_generation_prompt.format(context=chunk.page_content))
        try:
            question = output_QA_couple.split("Factoid question: ")[-1].split("Answer: ")[0]
            answer = output_QA_couple.split("Answer: ")[-1]
            assert len(answer) < 300, "Answer is too long"
            outputs.append(
                {
                    "context": chunk.page_content,
                    "question": question,
                    "answer": answer,
                    "source_doc": chunk.metadata["source"],
                }
            )
        except:
            print('Something went wrong in get_qa_couple_outputs()')
            continue

    # Write to a JSON file
    with open(file='qa_couple_outputs.json', mode='w', encoding='utf-8') as f:
        json.dump(obj=outputs, fp=f, ensure_ascii=True, indent=4)


if __name__ == "__main__":
    
    # HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')
    # EMBEDDING_MODEL = "sentence-transformers/multi-qa-mpnet-base-dot-v1"

    # repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    # llm_client = InferenceClient(model=repo_id, timeout=120, token=HUGGINGFACEHUB_API_TOKEN)

    # chunks = get_chunks(website_docs='website_docs',sources=pdf_source)

    # get_qa_couple_outputs(chunks=chunks, inference_client=llm_client)

    with open(file='qa_couple_outputs.json', mode ='r', encoding='utf-8') as f:
        loaded_qa_couples = json.load(f)

    for qa in loaded_qa_couples:
        print(f"{qa['question'] =}")
        print(f"{qa['answer'] =}\n\n")
    