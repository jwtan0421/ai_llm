import gradio as gr
from vector_db_utils import VectorDBConnector
from llm_utils import LLMUtils

# Load environment variables
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

# Initialize VectorDBConnector instance
vector_db = VectorDBConnector()
file_uploaded = False

def handle_file_upload(file, chat_history):
    """
    Handle the file upload and update the vector database.

    Args:
        file: The uploaded file.
        chat_history: The chat history.

    Returns:
        Updated chat history and empty text fragments.
    """
    if not file:
        chat_history = [("Assistant", "请先选择文件。")]
        return chat_history, ""
    
    # Add the file to the vector database
    vector_db.add_file(file.name)
    global file_uploaded 
    file_uploaded = True

    chat_history = [("Assistant", "文件已上传并处理成功。")]
    return chat_history, ""

def handle_query(query, chat_history, selected_llm):
    """
    Handle the user query by retrieving relevant documents and invoking the language model.

    Args:
        query: The user query.
        chat_history: The chat history.
        selected_llm: The selected language model.

    Returns:
        Updated chat history and relevant text fragments.
    """
    if file_uploaded == False:
        chat_history = [("Assistant", "请先上传文件。")]        
        return chat_history, ""
    
    if query.strip() == "":
        chat_history = [("Assistant", "请输入问题。")]        
        return chat_history, ""    
    
    # Get the retriever object
    retriever = vector_db.get_retriever()
    
    # Initialize the LLMUtils with the selected model
    llm = LLMUtils(selected_llm)
    
    # Invoke the language model with the query and retriever
    response, ref_texts = llm.invoke(query, retriever)

    chat_history.append(("User", query))
    chat_history.append(("Assistant", f"{selected_llm}: {response}\n"))    
    return chat_history, ref_texts

def format_chat(chat_history):
    """
    Format the chat history for display.

    Args:
        chat_history: The chat history.

    Returns:
        Formatted HTML string of the chat history.
    """
    formatted_chat = "<div><strong>对话历史</strong></div>"
    for speaker, text in chat_history:
        if speaker == "User":
            formatted_chat += f'<div style="text-align: right; margin: 10px;"><span style="background-color: #daf7a6; padding: 5px; border-radius: 5px;">{text}</span></div>'
        else:
            formatted_chat += f'<div style="text-align: left; margin: 10px;"><span style="background-color: #ffcccc; padding: 5px; border-radius: 5px;">{text}</span></div>'
    return formatted_chat

# Define the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("## ChatPDF by Langchain")

    with gr.Row():
        with gr.Column(scale=3):
            chat_display = gr.HTML(label="对话历史", elem_id="chat_display")

        with gr.Column(scale=1):
            chat_history = gr.State([])            
            upload = gr.File(label="上传PDF文件")
            upload_button = gr.Button("上传")
            query = gr.Textbox(label="输入问题", placeholder="请输入您的问题...")

            radio = gr.Radio(
                choices=["gpt", "ernie"],
                label="选择大模型",
                value="gpt"
            )
            
            with gr.Row():
                query_button = gr.Button("提交")
                clear_button = gr.Button("清除")                   
  
            ref_texts = gr.Textbox(label="相关文档片段", elem_id="ref_texts", interactive=False)
            
            # Define the click actions for the buttons
            upload_button.click(handle_file_upload, inputs=[upload, chat_history], outputs=[chat_history, ref_texts])
            query_button.click(handle_query, inputs=[query, chat_history, radio], outputs=[chat_history, ref_texts])
            clear_button.click(lambda: ([], ""), inputs=None, outputs=[chat_history, ref_texts])

    # Load and update the chat display
    demo.load(lambda: format_chat([]), inputs=None, outputs=chat_display)
    chat_history.change(fn=format_chat, inputs=chat_history, outputs=chat_display)

# Launch the Gradio app
demo.queue().launch(share=False, server_name='0.0.0.0', server_port=7860, inbrowser=True)
