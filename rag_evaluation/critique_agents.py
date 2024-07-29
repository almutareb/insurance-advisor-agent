from question_generation import call_llm
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os
import json
from tqdm.auto import tqdm
from prompts import (question_groundedness_critique_prompt, 
                     question_relevance_critique_prompt, 
                     question_standalone_critique_prompt)

import pandas as pd
import datasets



load_dotenv()


def append_to_json(file_path:str ='critique_outputs.json', new_data=None):
    """ Append data to a json file 
    
    """
    if os.path.exists(file_path):
        with open(file=file_path, mode ='r+') as file:
            file_data = json.load(file)
            file_data.append(new_data)
            file.seek(0)
            json.dump(obj=file_data, fp=file, ensure_ascii=True,  indent=4)
    else:
        with open(file_path, 'w') as file:
            json.dump(obj=[new_data], fp=file, indent=4)




def get_critique_agent_scores(qa_couples:list, inference_client: InferenceClient) -> None:
    """ Calculates a score for groundness, relevance and standalone given for Q&A and context

        Args:
            qa_couples (list): List of dicts containing the context and the Q&A generated 
            inference_client (InferenceClient): The LLM model used by the critique agent
        
        Returns:
            None - writes the results to a json file      
    """
    
    for output in tqdm(qa_couples):
        
        evaluations = {
            "groundedness": call_llm(inference_client=inference_client,
                                        prompt=question_groundedness_critique_prompt.format(context=output["context"], 
                                                                                            question=output["question"])),
            "relevance": call_llm(inference_client=inference_client,
                                    prompt=question_relevance_critique_prompt.format(question=output["question"])),
            "standalone": call_llm(inference_client=inference_client,
                                    prompt=question_standalone_critique_prompt.format(question=output["question"])),
        }

        try:
            for criterion, evaluation in evaluations.items():
                score_text = evaluation.split("Total rating: ")[-1].strip()
                try:
                    score = int(float(score_text))
                except ValueError:
                    score = int(float(score_text.split("\n")[0]))

                eval = evaluation.split("Total rating: ")[-2].split("Evaluation: ")[1]  
                
                output.update(
                    {
                        f"{criterion}_score": score,
                        f"{criterion}_eval": eval,
                    }
                )
            
        except Exception as e:
            print(f'Something went wrong while added the criterion scores in get_critique_agent_scores()\n{e}')
            continue
        append_to_json(new_data=output)
    
if __name__ == "__main__":


    HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')

    repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    llm_client = InferenceClient(model=repo_id, timeout=120, token=HUGGINGFACEHUB_API_TOKEN)

    with open(file='qa_couple_outputs.json', mode ='r', encoding='utf-8') as f:
        loaded_qa_couples = json.load(f)

    get_critique_agent_scores(qa_couples=loaded_qa_couples[280:300], inference_client=llm_client)

# print(eval_dataset)
