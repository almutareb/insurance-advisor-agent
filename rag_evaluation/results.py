import glob
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

result = json.load(open('eval_answers.json', "r"))

avg_score = avg_score = np.mean([int(res['eval_score_Mixtral-8x7B-Instruct-v0.1']) for res in result if isinstance(res, dict) and 'eval_score_Mixtral-8x7B-Instruct-v0.1' in res])

print(f'{(avg_score) =}')

accuracy_percentage = ((avg_score) / 5) * 100

print(f'{(accuracy_percentage) =}')

fig, ax = plt.subplots(figsize=(8, 6))
ax.bar(['Baseline RAG', 'with Re-Ranking', 'Hybrid Search'], [accuracy_percentage, 0, 0])
ax.set_ylim(0, 100)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_xlabel('RAG settings', fontsize=12)
ax.set_title('Accuracy of different RAG configurations', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()