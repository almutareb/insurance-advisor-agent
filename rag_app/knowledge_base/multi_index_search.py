from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.documents import Document
from typing import List, Dict



def multi_index_search(
    vector_Store_retrievers:List[VectorStoreRetriever],
    query:str,
        ) -> List[Dict[VectorStoreRetriever,List[str]]]:
    """
    
    ## Summary
    Search a set of vector stores and returns a list of `Documents` for each vector store 
    in a query
    
    ## Arguements
    vector_Store_retrievers (List[VectorStoreRetriever]): A set of VectorStoreRetriever
    query (str): the query
    
    ## Return 
    List[Dict[VectorStoreRetriever,List[str]]]
    """
    
    multi_indices_search_list = []
    for i in vector_Store_retrievers:
        results = i.invoke(query)
        data = {i : results }
        multi_indices_search_list.append(data)
    

    return multi_indices_search_list

if __name__ == "__main__":
    pass