# Import Gradio for UI, along with other necessary libraries
import gradio as gr
from rag_app.loading_data.load_S3_vector_stores import get_chroma_vs
from rag_app.agents.react_agent import agent_executor

get_chroma_vs()


if __name__ == "__main__":

    # Function to add a new input to the chat history
    def add_text(history, text):
        # Append the new text to the history with a placeholder for the response
        history = history + [(text, None)]
        return history, ""

    # Function representing the bot's response mechanism
    def bot(history):
        # Obtain the response from the 'infer' function using the latest input
        response = infer(history[-1][0], history)
        print(response)
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
            raise gr.Warning("Model is Overloaded, please try again in a few minutes!")
        
    def vote(data: gr.LikeData):
        if data.liked:
            print("You upvoted this response: ")
        else:
            print("You downvoted this response: ")

    def get_examples(input_text: str):
        tmp_history = [(input_text, None)]
        response = infer(input_text, tmp_history)
        return response['output']

    # CSS styling for the Gradio interface
    css = """
    #col-container {max-width: 1200px; margin-left: auto; margin-right: auto;}
    """

    # HTML content for the Gradio interface title
    title = """
    <div style="text-align:left;">
        <p>Hello, I BotTina 2.0, your intelligent AI assistant. I can help you explore Wuerttembergische Versicherungs products.<br />
    </div>
    """
    head_style = """
    <style>
    @media (min-width: 1536px)
    {
        .gradio-container {
            min-width: var(--size-full) !important;
        }
    }
    </style>
    """

    # Building the Gradio interface
    with gr.Blocks(theme=gr.themes.Soft(), title="InsurePal AI 🤵🏻‍♂️", head=head_style) as demo:
        with gr.Column(elem_id="col-container"):
            gr.HTML()  # Add the HTML title to the interface
            chatbot = gr.Chatbot([], elem_id="chatbot",
                                        label="InsurePal AI",
                                        bubble_full_width=False,
                                        avatar_images=(None, "https://dacodi-production.s3.amazonaws.com/store/87bc00b6727589462954f2e3ff6f531c.png"),
                                        height=680,)  # Initialize the chatbot component
            chatbot.like(vote, None, None)

            # Create a row for the question input
            with gr.Row():
                question = gr.Textbox(label="Question", show_label=False, placeholder="Type your question and hit Enter ", scale=4)
                send_btn = gr.Button(value="Send", variant="primary", scale=0)
            with gr.Accordion(label="Beispiele", open=False):
                #examples
                examples = gr.Examples([
                    "Welche Versicherungen brauche ich als Student?", 
                    "Wie melde ich einen Schaden?",
                    "Wie kann ich mich als Selbstständiger finanziell absichern?",
                    "Welche Versicherungen sollte ich für meine Vorsorge abschliessen?"
                    ], inputs=[question], label="") #, cache_examples="lazy", fn=get_examples, outputs=[chatbot]

            with gr.Row():
                clear = gr.Button("Clear")  # Add a button to clear the chat

        # Define the action when the question is submitted
        question.submit(add_text, [chatbot, question], [chatbot, question], queue=False).then(
            bot, chatbot, chatbot)
        send_btn.click(add_text, [chatbot, question], [chatbot, question], queue=False).then(
            bot, chatbot, chatbot)
        # Define the action for the clear button
        clear.click(lambda: None, None, chatbot, queue=False)

    # Launch the Gradio demo interface
    demo.queue().launch(share=False, debug=True)