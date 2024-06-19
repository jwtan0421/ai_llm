import gradio as gr
from utilities import *
from vector_db import MyVectorDBConnector
from rag_bot import RAG_Bot
from llm_api import get_completion, get_embeddings


# 创建一个向量数据库对象
vector_db = MyVectorDBConnector("demo_text_split", get_embeddings)

# 创建一个RAG机器人
bot = RAG_Bot(
    vector_db,
    llm_api=get_completion
)


def handle_file_upload(file, chat_history):       
    if not file:
        chat_history = [("Assistant", "请先选择文件。")]
        return chat_history, ""
    
    paragraphs = extract_text_from_pdf(file.name, min_line_length=10)
    
    chunks = split_text(paragraphs, 300, 100)

    # 向向量数据库中添加文档
    vector_db.add_documents(chunks)
    chat_history = [("Assistant", "文件已上传并处理成功。")]
    return chat_history, ""

def handle_query(query, chat_history):
    collection_count = vector_db.collection_size()
    
    if collection_count == 0:
        chat_history = [("Assistant", "请先上传文件。")]        
        return chat_history, ""
    
    search_results = vector_db.search(query, 2)
    ref_docs = ""

    for doc in search_results['documents'][0]:
        ref_docs += doc+"\n\n"

    response = bot.chat(query)
    chat_history.append(("User", query))
    chat_history.append(("Assistant", f"{response}\n"))    
    return chat_history, ref_docs

def format_chat(chat_history):
    formatted_chat = "<div><strong>对话历史</strong></div>"
    for speaker, text in chat_history:
        if speaker == "User":
            formatted_chat += f'<div style="text-align: right; margin: 10px;"><span style="background-color: #daf7a6; padding: 5px; border-radius: 5px;">{text}</span></div>'
        else:
            formatted_chat += f'<div style="text-align: left; margin: 10px;"><span style="background-color: #ffcccc; padding: 5px; border-radius: 5px;">{text}</span></div>'
    return formatted_chat


with gr.Blocks() as demo:
    gr.Markdown("## ChatPDF")

    with gr.Row():
        with gr.Column(scale=3):
            chat_display = gr.HTML(label="对话历史", elem_id="chat_display")

        with gr.Column(scale=1):
            chat_history = gr.State([])            
            upload = gr.File(label="上传PDF文件")
            upload_button = gr.Button("上传")
            query = gr.Textbox(label="输入问题", placeholder="请输入您的问题...")
            
            with gr.Row():
                query_button = gr.Button("提交")
                clear_button = gr.Button("清除")                   
  
            ref_docs = gr.Textbox(label="相关文档片段", elem_id="ref_docs", interactive=False)
            upload_button.click(handle_file_upload, inputs=[upload, chat_history], outputs=[chat_history, ref_docs])
            query_button.click(handle_query, inputs=[query, chat_history], outputs=[chat_history, ref_docs])
            clear_button.click(lambda: ([], ""), inputs=None, outputs=[chat_history, ref_docs])

    demo.load(lambda: format_chat([]), inputs=None, outputs=chat_display)
    chat_history.change(fn=format_chat, inputs=chat_history, outputs=chat_display)

demo.launch()
