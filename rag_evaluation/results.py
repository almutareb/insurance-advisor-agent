import glob
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


qa_couple_outputs = json.load(open('qa_couple_outputs.json', "r"))
print(f"{len(qa_couple_outputs) = }")
critique_outputs = json.load(open('critique_outputs.json', "r"))
print(f"{len(critique_outputs) = }")
test_rag = json.load(open('test_rag.json', "r"))
print(f"{len(test_rag) = }")
eval_answers = json.load(open('eval_answers.json', "r"))
print(f"{len(eval_answers) = }")


# result = json.load(open('eval_answers_chatgpt.json', "r"))
# print(f"gpt_eval_answers {len(result) = }")


# scores = []

# for res in result:
#     if isinstance(res, dict) and 'eval_score_gpt4' in res:
#         try:
#             score =int(res['eval_score_gpt4'])
#             scores.append(score)
#         except ValueError:
#             continue
# avg_score = np.mean(scores)

# # avg_score = np.mean([int(res['eval_score_Mixtral-8x7B-Instruct-v0.1']) for res in result if isinstance(res, dict) and 'eval_score_Mixtral-8x7B-Instruct-v0.1' in res])

# print(f'{(avg_score) =}')

# accuracy_percentage = ((avg_score) / 5) * 100

# print(f'{(accuracy_percentage) =}')

def get_accuracy_scores(eval_answers:json, evaluator_name:str) -> float:
    """ Returns the accuracy percentage from evaluation output"""

    result = json.load(open(file=eval_answers, mode="r"))
    
    scores = []

    for res in result:
        if isinstance(res, dict) and f'eval_score_{evaluator_name}' in res:
            try:
                score =int(res[f'eval_score_{evaluator_name}'])
                scores.append(score)
            except ValueError:
                continue
    avg_score = np.mean(scores)
    accuracy_percentage = ((avg_score) / 5) * 100
    return accuracy_percentage

baseline_rag = get_accuracy_scores(eval_answers='eval_answers_chatgpt.json', evaluator_name='gpt4')
print(baseline_rag)
rerank_score = get_accuracy_scores(eval_answers='eval_answers_rerank_chatgpt.json',evaluator_name='gpt4')
print(rerank_score)

# fig, ax = plt.subplots(figsize=(8, 6))
# ax.bar(['Baseline RAG', 'with Re-Ranking', 'Hybrid Search'], [baseline_rag, rerank_score, 0])
# ax.set_ylim(0, 100)
# ax.set_ylabel('Accuracy', fontsize=12)
# ax.set_xlabel('RAG settings', fontsize=12)
# ax.set_title('Accuracy of different RAG configurations', fontsize=16, fontweight='bold')
# plt.tight_layout()
# plt.show()


