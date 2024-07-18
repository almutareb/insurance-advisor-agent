---
title: Insurance Advisor Agents PoC
emoji: 🤖
colorFrom: red
colorTo: indigo
sdk: gradio
python: 3.11
app_file: app_gui.py
pinned: false
---


# Insurance Advisor Agent(s)

Welcome to the Insurance Advisor Agent(s) project! This project aims to set up a modular, multi-agent system to handle inquiries for an insurance company. The system utilizes various approaches to provide reliable answers regarding insurance products.

## Features

### Vectorstore Search
- **Multi-index (chunks + summaries)**
- **Metadata Filtering** 
- **Re-ranking** [DONE]
- **Hybrid Search (BM25 + Vectorstore)** [DONE]
- **HyDE/HyQE**

### ReAct Agent
- **Usage of ReAct Agent instead of Chain** [DONE]
- **Added Tools (Google Search Engine)** [DONE]
- **Query Re-writing**
- **Additional Enhancements**

### Corrective RAG
- Implementation of Corrective Retrieval-Augmented Generation

### Workflow for Agents
- Determination of the customer's funnel stage
- Custom prompts for each step
- Following a structured "script"

## Getting Started

To get started with this project, follow these steps:

1. **Clone the Repository**:
   ```sh
   git clone https://github.com/your-username/insurance-advisor-agent.git
   cd insurance-advisor-agent

2. **Install Dependencies**:
   ```sh
   pip install -r requirements.txt

3. **Run the Application**:
   ```sh
   python main.py

## Contributing

We welcome contributions! Please read our Contributing Guide to get started.
License

This project is licensed under the MIT License. See the LICENSE file for details.
Contact

If you have any questions or suggestions, feel free to open an issue or start a discussion.