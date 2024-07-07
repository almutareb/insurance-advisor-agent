from langchain.tools import BaseTool, StructuredTool, tool
from rag_app.agents.kb_retriever_agent import agent_worker

@tool
def web_research(query: str) -> str:
    """Verbessere die Ergebnisse durch eine Suche über die Webseite der Versicherung. Erstelle eine neue Suchanfrage, um die Erfolgschancen zu verbesseren."""
    
    result = agent_worker.invoke(
        {
            "input": query
        }
    )
    print(result)
    return result.__str__()