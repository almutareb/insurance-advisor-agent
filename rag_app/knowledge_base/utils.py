from langchain_core.documents import Document
from chains import generate_document_summary_prompt
from config import SECONDARY_LLM


def generate_document_summaries(
        docs: list[Document]
    ) -> list[Document]:
    """
    Generates summaries for a list of Document objects and updates their metadata with the summaries.

    Args:
        docs (List[Document]): A list of Document objects to generate summaries for.

    Returns:
        List[Document]: A new list of Document objects with updated metadata containing the summaries.

    Example:
        docs = [Document(metadata={"title": "Doc1"}), Document(metadata={"title": "Doc2"})]
        updated_docs = generate_document_summaries(docs)
        for doc in updated_docs:
            print(doc.metadata["summary"])

    """
    
    new_docs = docs.copy()
    
    for doc in new_docs:
        
        genrate_summary_chain = generate_document_summary_prompt | SECONDARY_LLM
        summary = genrate_summary_chain.invoke(
            {"document":str(doc.metadata)}
        )        
        
        doc.metadata.update(
            {"summary":summary}
        )
    
    return new_docs