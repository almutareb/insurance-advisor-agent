from rag_app.load_data_from_urls import load_docs_from_urls
from rag_app.create_embedding import create_embeddings
from rag_app.generate_summary import generate_description, generate_keywords
from rag_app.handle_vector_store import build_vector_store

docs = load_docs_from_urls(["https://www.wuerttembergische.de/"],5)

for doc in docs:
    keywords=generate_keywords(doc)
    description=generate_description(doc)
    doc.metadata['keywords']=keywords
    doc.metadata['description']=description

build_vector_store(docs, './vectorstore/faiss-insurance-agent-1500','sentence-transformers/multi-qa-mpnet-base-dot-v1',True,1500,150)


#print(create_embeddings(docs))
