---
title: Insurance Advisor Agents PoC
emoji: 🤖
colorFrom: red
colorTo: indigo
sdk: docker
python: 3.11
app_file: app_gui.py
pinned: false
---


# Insurance Advisor Agent(s)

Setup a modular, multi-agent system to handle inqueries to an insurance company. The system utilizes different approachs to find reliable answers regarding the insurance products

1. Improve Vectorstore search
    Isayah:
    - multi index (chunks+summaries)
    - metadata filtering
    Karan:
    - re-ranking
    - hybrid search (bm25+vectorstore)

    
    - HyDE/HyQE
2. Use ReAct agent instead of chain [DONE]
    - add tools (Google search engine) [DONE]
    - query re-writing
    - ...
3. Use Corrective RAG
4. Workflow for the agents
    - determine funnel stage of customer
    - different prompts per step
    - follow a "script"

This project is licensed under the MIT License: LICENSE.
