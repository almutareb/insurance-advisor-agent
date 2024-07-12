# Import Gradio for UI, along with other necessary libraries
import gradio as gr
from fastapi import FastAPI
from fastapi import FastAPI
from rag_app.agents.react_agent import agent_executor, llm
from rag_app.chains import user_response_sentiment_prompt
from typing import Dict
import re
from rag_app.utils.utils import extract_responses
from rag_app.loading_data.load_S3_vector_stores import get_chroma_vs
from rag_app.agents.react_agent import agent_executor
# need to import the qa!

app = FastAPI()
get_chroma_vs()

if __name__ == "__main__":

    # Function to add a new input to the chat history
    def add_text(history, text):
        # Append the new text to the history with a placeholder for the response
        history = history + [(text, None)]
        return history, ""
    # Function to add a new input to the chat history
    def add_text(history, text):
        # Append the new text to the history with a placeholder for the response
        history = history + [(text, None)]
        return history, ""

    # Function representing the bot's response mechanism
    def bot(history):
        # Obtain the response from the 'infer' function using the latest input
        response = infer(history[-1][0], history)
        #sources = [doc.metadata.get("source") for doc in response['source_documents']]
        #src_list = '\n'.join(sources)
        #print_this = response['result'] + "\n\n\n Sources: \n\n\n" + src_list


        #history[-1][1] = print_this #response['answer']
        # Update the history with the bot's response
        history[-1][1] = response['output']
        return history
    # Function representing the bot's response mechanism
    def bot(history):
        # Obtain the response from the 'infer' function using the latest input
        response = infer(history[-1][0], history)
        history[-1][1] = response['output']
        return history

    # Function to infer the response using the RAG model
    def infer(question, history):
        # Use the question and history to query the RAG model
        #result = qa({"query": question, "history": history, "question": question})
        try:
            result = agent_executor.invoke(
                {
                    "input": question,
                    "chat_history": history
                }
            )
            return result
        except Exception:
            raise gr.Error("Model is Overloaded, Please retry later!")
        
    def vote(data: gr.LikeData):
        if data.liked:
            print("You upvoted this response: " + data.value)
        else:
            print("You downvoted this response: " + data.value)
    # Function to infer the response using the RAG model
    def infer(question, history):
        # Use the question and history to query the RAG model
        #result = qa({"query": question, "history": history, "question": question})
        # try:
        #     data = user_sentiment_chain.invoke({"user_reponse":question})
        #     responses = extract_responses(data)
        #     if responses['AI'] == "1":
        #         pass
        #         # Do important stuff here plox
        #         # store into database
        # except Exception as e:
        #     raise e
        try:
            result = agent_executor.invoke(
                {
                    "input": question,
                    "chat_history": history
                }
            )
            return result
        except Exception as e:
            # raise gr.Error("Model is Overloaded, Please retry later!")
            raise e
        
    def vote(data: gr.LikeData):
        if data.liked:
            print("You upvoted this response: ")
        else:
            print("You downvoted this response: ")

    # CSS styling for the Gradio interface
    css = """
    #col-container {max-width: 700px; margin-left: auto; margin-right: auto;}
    """
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
            chatbot.like(vote, None, None)
            clear = gr.Button("Clear")  # Add a button to clear the chat
    # Building the Gradio interface
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        with gr.Column(elem_id="col-container"):
            gr.HTML(title)  # Add the HTML title to the interface
            chatbot = gr.Chatbot([], elem_id="chatbot",
                                        label="BotTina 2.0",
                                        bubble_full_width=False,
                                        avatar_images=(None, "https://dacodi-production.s3.amazonaws.com/store/87bc00b6727589462954f2e3ff6f531c.png"),
                                        height=680,)  # Initialize the chatbot component
            chatbot.like(vote, None, None)
            clear = gr.Button("Clear")  # Add a button to clear the chat

            # Create a row for the question input
            with gr.Row():
                question = gr.Textbox(label="Question", placeholder="Type your question and hit Enter ")
            # Create a row for the question input
            with gr.Row():
                question = gr.Textbox(label="Question", placeholder="Type your question and hit Enter ")

        # Define the action when the question is submitted
        question.submit(add_text, [chatbot, question], [chatbot, question], queue=False).then(
            bot, chatbot, chatbot
        )
        # Define the action for the clear button
        clear.click(lambda: None, None, chatbot, queue=False)
        # Define the action when the question is submitted
        question.submit(add_text, [chatbot, question], [chatbot, question], queue=False).then(
            bot, chatbot, chatbot
        )
        # Define the action for the clear button
        clear.click(lambda: None, None, chatbot, queue=False)

    # Launch the Gradio demo interface
    demo.queue().launch(share=False, debug=True)

    app = gr.mount_gradio_app(app, demo, path="/")
    # Launch the Gradio demo interface
    demo.queue().launch(share=False, debug=True)