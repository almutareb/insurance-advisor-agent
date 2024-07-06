from rag_app.loading_data.load_urls_recurisvely import load_docs_from_urls
from rag_app.knowledge_base.create_embedding import create_embeddings
from rag_app.utils.generate_summary import generate_description, generate_keywords
from rag_app.knowledge_base.build_vector_store import build_vector_store
from rag_app.loading_data.scrap_website import scrap_website
from rag_app.loading_data.load_S3_vector_stores import get_chroma_vs, get_faiss_vs

# 1. load the urls
# 2. build the vectorstore -> the function will create the chunking and embeddings
# 3. initialize the db retriever
# 4. 

# docs = load_docs_from_urls(["https://www.wuerttembergische.de/"],6)

# # for doc in docs:
# #     keywords=generate_keywords(doc)
# #     description=generate_description(doc)
# #     doc.metadata['keywords']=keywords
# #     doc.metadata['description']=description
# #     print(doc.metadata)

# build_vector_store(docs, './vectorstore/faiss-insurance-agent-1500','sentence-transformers/multi-qa-mpnet-base-dot-v1',True,1500,150)


# print(create_embeddings(docs))


#print(scrap_website(target_url='https://www.wuerttembergische.de/',depth=1))
get_faiss_vs()
#get_chroma_vs()