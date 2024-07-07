# Import Gradio for UI, along with other necessary libraries
import gradio as gr
from rag_app.agents.react_agent import agent_executor
# need to import the qa!

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

# Function to infer the response using the RAG model
def infer(question, history):
    # Use the question and history to query the RAG model
    #result = qa({"query": question, "history": history, "question": question})
    result = agent_executor.invoke(
        {
            "input": question,
            "chat_history": history
        }
    )
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
demo.launch(share=False, debug=True)