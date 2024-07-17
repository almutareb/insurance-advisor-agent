from question_generation import call_llm
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os
import json
from tqdm.auto import tqdm
from prompts import (question_groundedness_critique_prompt, 
                     question_relevance_critique_prompt, 
                     question_standalone_critique_prompt)


if __name__ == "__main__":
    HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')

    repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    llm_client = InferenceClient(model=repo_id, timeout=120, token=HUGGINGFACEHUB_API_TOKEN)

    with open(file='qa_couple_outputs.json', mode ='r', encoding='utf-8') as f:
        loaded_qa_couples = json.load(f)

    output = loaded_qa_couples[0]
    # print(type(output))

    def get_critique_agent_scores(qa_couples:list, inference_client: InferenceClient) -> None:
        """ Calculates a score for groundness, relevance and standalone given a Q&A and context

            Args:
                qa_couples (list): List of dicts containing the context and the Q&A generated 
                inference_client (InferenceClient): The LLM model used by the critique agent
            
            Returns:
                None - writes the results to a json file      
        """

        for output in tqdm(loaded_qa_couples):
        
            evaluations = {
                "groundedness": call_llm(
                    llm_client,
                    question_groundedness_critique_prompt.format(context=output["context"], question=output["question"]),
                ),
                "relevance": call_llm(
                    llm_client,
                    question_relevance_critique_prompt.format(question=output["question"]),
                ),
                "standalone": call_llm(
                    llm_client,
                    question_standalone_critique_prompt.format(question=output["question"]),
                ),
            }

            print(f'{evaluations =}\n\n')

            try:
                for criterion, evaluation in evaluations.items():
                    score, eval = (
                        int(evaluation.split("Total rating: ")[-1].strip()),
                        evaluation.split("Total rating: ")[-2].split("Evaluation: ")[1],
                    )
                    output.update(
                        {
                            f"{criterion}_score": score,
                            f"{criterion}_eval": eval,
                        }
                    )
            except Exception as e:
                print('Something went wrong while added the criterion scores in get_critique_agent_scores()\ne')
