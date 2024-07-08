from langchain.tools import BaseTool, StructuredTool, tool
from langchain_community.tools import HumanInputRun
from rag_app.agents.kb_retriever_agent import agent_worker
from operator import itemgetter
from typing import Dict, List

@tool
def web_research(query: str) -> List[dict]:
    """Verbessere die Ergebnisse durch eine Suche über die Webseite der Versicherung. Erstelle eine neue Suchanfrage, um die Erfolgschancen zu verbesseren."""
    
    result = agent_worker.invoke(
        {
            "input": query
        }
    )
    #print(result)
    return result

@tool
def ask_user(query: str) -> str:
    """Frage den Benutzer direkt wenn du nicht sicher bist was er meint oder du eine Entscheidung brauchst."""
    
    result = HumanInputRun.invoke(query)
    return result

@tool
def get_email(query: str) -> str:
    """Frage den Benutzer nach seiner EMail Adresse, wenn du denkst du hast seine Anfrage beantwortet hast, damit wir ihm mehr Informationen im Anschluss zu senden kannst."""

    result = HumanInputRun.invoke(query)
    return result
