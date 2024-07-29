from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate)
from langchain.schema import SystemMessage
from prompts import EVALUATION_PROMPT
import json
from langchain_huggingface import HuggingFaceEndpoint
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from tqdm.auto import tqdm

load_dotenv()

def evaluate_answers(
    answer_path: str,
    eval_chat_model,
    evaluator_name: str,
    output_file:str) -> None:
    """Evaluates generated answers. Modifies the given answer file in place for better checkpointing."""
    answers = []
    if os.path.isfile(answer_path):  # load previous generations if they exist
        answers = json.load(open(answer_path, "r"))

    evaluation_prompt_template = ChatPromptTemplate.from_messages(
        [
        SystemMessage(content="You are a fair evaluator language model."),
        HumanMessagePromptTemplate.from_template(EVALUATION_PROMPT)
        ]
    )

    for experiment in tqdm(answers):
        if f"eval_score_{evaluator_name}" in experiment:
            continue

        eval_prompt = evaluation_prompt_template.format_messages(
            instruction=experiment["question"],
            response=experiment["generated_answer"],
            reference_answer=experiment["true_answer"],
        )
        eval_result = eval_chat_model.invoke(eval_prompt)
        if evaluator_name == 'gpt4':
            try:
                feedback, score = [item.strip() for item in eval_result.content.split("[RESULT]")]
            except:
                print("Couldn't get the result from the response")    
                      
        else:
            try:
                feedback, score = [item.strip() for item in eval_result.split("[RESULT]")]
            except:
                print("Couldn't get the result from the response")
            
        experiment[f"eval_score_{evaluator_name}"] = score
        experiment[f"eval_feedback_{evaluator_name}"] = feedback

        with open(output_file, "w") as f:
            json.dump(answers, f)


if __name__ == "__main__":

    # HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')

    # repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"

    # eval_model  = HuggingFaceEndpoint(repo_id=repo_id,
    #                                   task="text-generation",
    #                                   huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN)

    # evaluate_answers(answer_path='test_rag.json',
    #                  eval_chat_model=eval_model,
    #                  evaluator_name='Mixtral-8x7B-Instruct-v0.1',
    #                  output_file='eval_answers.json')
    
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    
    chatgpt = ChatOpenAI(
        model="gpt-4-turbo",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=OPENAI_API_KEY)
    
    evaluate_answers(answer_path='test_rerank_rag.json',
                     eval_chat_model=chatgpt,
                     evaluator_name='gpt4',
                     output_file='eval_answers_rerank_chatgpt.json')
