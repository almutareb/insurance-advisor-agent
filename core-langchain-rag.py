# Importing necessary libraries
import sys
import os
import time

# # Importing RecursiveUrlLoader for web scraping and BeautifulSoup for HTML parsing
# from langchain.document_loaders.recursive_url_loader import RecursiveUrlLoader
# from bs4 import BeautifulSoup as Soup
# import mimetypes

# # List of URLs to scrape
# urls = ["https://langchain-doc.readthedocs.io/en/latest"]

# # Initialize an empty list to store the documents
# docs = []

# # Looping through each URL in the list - this could take some time!
# stf = time.time()  # Start time for performance measurement
# for url in urls:
#     try:
#         st = time.time()  # Start time for performance measurement
#         # Create a RecursiveUrlLoader instance with a specified URL and depth
#         # The extractor function uses BeautifulSoup to parse the HTML content and extract text
#         loader = RecursiveUrlLoader(url=url, max_depth=5, extractor=lambda x: Soup(x, "html.parser").text)
        
#         # Load the documents from the URL and extend the docs list
#         docs.extend(loader.load())

#         et = time.time() - st  # Calculate time taken for splitting
#         print(f'Time taken for downloading documents from {url}: {et} seconds.')
#     except Exception as e:
#         # Print an error message if there is an issue with loading or parsing the URL
#         print(f"Failed to load or parse the URL {url}. Error: {e}", file=sys.stderr)
# etf = time.time() - stf  # Calculate time taken for splitting
# print(f'Total time taken for downloading {len(docs)} documents: {etf} seconds.')

# # Import necessary modules for text splitting and vectorization
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import time
# from langchain_community.vectorstores import FAISS
# from langchain.vectorstores.utils import filter_complex_metadata
# from langchain_community.embeddings import HuggingFaceEmbeddings

# # Configure the text splitter
# text_splitter = RecursiveCharacterTextSplitter(
#     separators=["\n\n", "\n", "(?<=\. )", " ", ""],  # Define the separators for splitting text
#     chunk_size=500,  # The size of each text chunk
#     chunk_overlap=50,  # Overlap between chunks to ensure continuity
#     length_function=len,  # Function to determine the length of each chunk
# )

# try:
#     # Stage one: Splitting the documents into chunks for vectorization
#     st = time.time()  # Start time for performance measurement
#     print('Loading documents and creating chunks ...')
#     # Split each document into chunks using the configured text splitter
#     chunks = text_splitter.create_documents([doc.page_content for doc in docs], metadatas=[doc.metadata for doc in docs])
#     et = time.time() - st  # Calculate time taken for splitting
#     print(f"created "+chunks+" chunks")
#     print(f'Time taken for document chunking: {et} seconds.')
# except Exception as e:
#     print(f"Error during document chunking: {e}", file=sys.stderr)

# # Path for saving the FAISS index
# FAISS_INDEX_PATH = "./vectorstore/lc-faiss-multi-mpnet-500"

# try:
#     # Stage two: Vectorization of the document chunks
#     model_name = "sentence-transformers/multi-qa-mpnet-base-dot-v1"  # Model used for embedding

#     # Initialize HuggingFace embeddings with the specified model
#     embeddings = HuggingFaceEmbeddings(model_name=model_name)

#     print(f'Loading chunks into vector store ...')
#     st = time.time()  # Start time for performance measurement
#     # Create a FAISS vector store from the document chunks and save it locally
#     db = FAISS.from_documents(filter_complex_metadata(chunks), embeddings)
#     db.save_local(FAISS_INDEX_PATH)
#     et = time.time() - st  # Calculate time taken for vectorization
#     print(f'Time taken for vectorization and saving: {et} seconds.')
# except Exception as e:
#     print(f"Error during vectorization or FAISS index saving: {e}", file=sys.stderr)

# alternatively download a preparaed vectorized index from S3 and load the index into vectorstore
# Import necessary libraries for AWS S3 interaction, file handling, and FAISS vector stores
import boto3
from botocore import UNSIGNED
from botocore.client import Config
import zipfile
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

# Load environment variables from a .env file
config = load_dotenv(".env")

# Retrieve the Hugging Face API token from environment variables
HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')
S3_LOCATION = os.getenv("S3_LOCATION")
S3_FILE_NAME = os.getenv("FAISS_VS_NAME")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH")

# try:
#     # Initialize an S3 client with unsigned configuration for public access
#     s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

#     # Define the FAISS index path and the destination for the downloaded file
#     #FAISS_INDEX_PATH = './vectorstore/lc-faiss-multi-mpnet-500-markdown'
#     VS_DESTINATION = FAISS_INDEX_PATH + ".zip"

#     # Download the pre-prepared vectorized index from the S3 bucket
#     print("Downloading the pre-prepared vectorized index from S3...")
#     s3.download_file(S3_LOCATION, S3_FILE_NAME, VS_DESTINATION)

#     # Extract the downloaded zip file
#     with zipfile.ZipFile(VS_DESTINATION, 'r') as zip_ref:
#         zip_ref.extractall('./vectorstore/')
#     print("Download and extraction completed.")
    
# except Exception as e:
#     print(f"Error during downloading or extracting from S3: {e}", file=sys.stderr)

# Define the model name for embeddings
model_name = os.getenv("EMBEDDING_MODEL")

try:
    # Initialize HuggingFace embeddings with the specified model
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    # Load the local FAISS index with the specified embeddings
    db = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    print("FAISS index loaded successfully.")
except Exception as e:
    print(f"Error during FAISS index loading: {e}", file=sys.stderr)

# Import necessary modules for environment variable management and HuggingFace integration
from langchain_huggingface import HuggingFaceEndpoint

# Initialize the vector store as a retriever for the RAG pipeline
retriever = db.as_retriever()#search_type="mmr", search_kwargs={'k': 3, 'lambda_mult': 0.25})

llm_model = os.getenv("LLM_MODEL")

try:
    # Load the model from the Hugging Face Hub
    model_id = HuggingFaceEndpoint(repo_id=llm_model, 
        temperature=0.1,         # Controls randomness in response generation (lower value means less random)
        max_new_tokens=1024,     # Maximum number of new tokens to generate in responses
        repetition_penalty=1.2,  # Penalty for repeating the same words (higher value increases penalty)
        return_full_text=False   # If False, only the newly generated text is returned; if True, the input is included as well
    )
    print("Model loaded successfully from Hugging Face Hub.")
except Exception as e:
    print(f"Error loading model from Hugging Face Hub: {e}", file=sys.stderr)



# Importing necessary modules for retrieval-based question answering and prompt handling
from langchain.chains import RetrievalQA
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_core.output_parsers import StrOutputParser

# Declare a global variable 'qa' for the retrieval-based question answering system
global qa

# Define a prompt template for guiding the model's responses
template = """
You are a friendly insurance product advisor, your task is to help customers find the best products from Württembergische GmbH.\
  You help the user find the answers to all his questions queries. Answer in short and simple terms and offer to explain the product and terms to the user.\
    Use the following context (delimited by <ctx></ctx>) and the chat history (delimited by <hs></hs>) to help find the best product for the user:
------
<ctx>
{context}
</ctx>
------
<hs>
{history}
</hs>
------
{question}
Answer:
"""

# Create a PromptTemplate object with specified input variables and the defined template
prompt = PromptTemplate.from_template(
    #input_variables=["history", "context", "question"],  # Variables to be included in the prompt
    template=template,  # The prompt template as defined above
)
prompt.format(context="context", history="history", question="question")
# Create a memory buffer to manage conversation history
memory = ConversationBufferMemory(
    memory_key="history",  # Key for storing the conversation history
    input_key="question"  # Key for the input question
)

# Initialize the RetrievalQA object with the specified model, retriever, and additional configurations
qa = RetrievalQA.from_chain_type(
    llm=model_id,  # Language model loaded from Hugging Face Hub
    retriever=retriever,  # The vector store retriever initialized earlier
    return_source_documents=True,  # Option to return source documents along with responses
    chain_type_kwargs={
        "verbose": True,  # Enables verbose output for debugging and analysis
        "memory": memory,  # Memory buffer for managing conversation history
        "prompt": prompt  # Prompt template for guiding the model's responses
    }
)

def generate_qa_retriever(history: dict, question: str, llm_model:HuggingFaceEndpoint = model_id) -> dict:
    """ Generare a response to queries using the retriever"""

    # Define a prompt template for guiding the model's responses
    template = """
    You are a friendly insurance product advisor, your task is to help customers find the best products from Württembergische GmbH.\
    You help the user find the answers to all his questions. Answer in short and simple terms and offer to explain the product and terms to the user.\
    Respond only using the provided context (delimited by <ctx></ctx>) and only in German or English, depending on the question's language.
    Use the chat history (delimited by <hs></hs>) to help find the best product for the user:
    ------
    <ctx>
    {context}
    </ctx>
    ------
    <hs>
    {history}
    </hs>
    ------
    {question}
    Answer:
    """

    # Create a PromptTemplate object with specified input variables and the defined template
    prompt = PromptTemplate.from_template(
        template=template,  # The prompt template as defined above
    )
    prompt.format(context="context", history="history", question="question")
    # Create a memory buffer to manage conversation history
    memory = ConversationBufferMemory(
        memory_key="history",  # Key for storing the conversation history
        input_key="question"  # Key for the input question
    )
    
    llm_chain = prompt | llm_model
    result = llm_chain.invoke({"context": retriever, "history": history, "question": question})
    print(result)
    return result

# Import Gradio for UI, along with other necessary libraries
import gradio as gr

# Function to add a new input to the chat history
def add_text(history, text):
    # Append the new text to the history with a placeholder for the response
    history = history + [(text, None)]
    return history, ""

# Function representing the bot's response mechanism
def bot(history):
    # Obtain the response from the 'infer' function using the latest input
    response = infer(history[-1][0], history)
    sources = [doc.metadata.get("source") for doc in response['source_documents']]
    src_list = '\n'.join(sources)
    print_this = response['result'] + "\n\n\n Sources: \n\n\n" + src_list


    #history[-1][1] = response #print_this #response['answer']
    history[-1][1] = print_this #response['answer']
    # Update the history with the bot's response
    #history[-1][1] = response['result']
    return history

# Function to infer the response using the RAG model
def infer(question, history):
    # Use the question and history to query the RAG model
    #result = generate_qa_retriever(history, question)
    result = qa({"query": question, "history": history, "question": question})
    print(*result)
    return result

# CSS styling for the Gradio interface
css = """
#col-container {max-width: 700px; margin-left: auto; margin-right: auto;}
"""

# HTML content for the Gradio interface title
title = """
<div style="text-align:left;">
    <p>Hello, I BotTina 2.0, your intelligent AI assistant. I can help you explore Wuerttembergische Versicherungs products.<br />
</div>
"""

# Building the Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    with gr.Column(elem_id="col-container"):
        gr.HTML(title)  # Add the HTML title to the interface
        chatbot = gr.Chatbot([], elem_id="chatbot",
                                    label="BotTina 2.0",
                                     bubble_full_width=False,
                                     avatar_images=(None, "https://dacodi-production.s3.amazonaws.com/store/87bc00b6727589462954f2e3ff6f531c.png"),
                                     height=680,)  # Initialize the chatbot component
        clear = gr.Button("Clear")  # Add a button to clear the chat

        # Create a row for the question input
        with gr.Row():
            question = gr.Textbox(label="Question", placeholder="Type your question and hit Enter ")

    # Define the action when the question is submitted
    question.submit(add_text, [chatbot, question], [chatbot, question], queue=False).then(
        bot, chatbot, chatbot
    )
    # Define the action for the clear button
    clear.click(lambda: None, None, chatbot, queue=False)

# Launch the Gradio demo interface
demo.launch(debug=True)