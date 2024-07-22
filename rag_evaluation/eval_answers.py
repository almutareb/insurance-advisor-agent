from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate)
from langchain.schema import SystemMessage
from prompts import EVALUATION_PROMPT
import json
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()

HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')



evaluation_prompt_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content="You are a fair evaluator language model."),
        HumanMessagePromptTemplate.from_template(EVALUATION_PROMPT),
    ]
)

answers = json.load(open(file='test_rag.json', mode="r"))

experiment = answers[0]

eval_prompt = evaluation_prompt_template.format_messages(
                instruction=experiment["question"],
                response=experiment["generated_answer"],
                reference_answer=experiment["true_answer"])

# repo_id = "meta-llama/Meta-Llama-3-8B-Instruct"
repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"

eval_chat_model  = HuggingFaceEndpoint(
    repo_id=repo_id,
    task="text-generation",
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN)

eval_result = eval_chat_model.invoke(eval_prompt)

print(eval_result)

# feedback, score = [item.strip() for item in eval_result.content.split("[RESULT]")]
#     experiment[f"eval_score_{evaluator_name}"] = score
#     experiment[f"eval_feedback_{evaluator_name}"] = feedback


if __name__ == "__main__":
    pass