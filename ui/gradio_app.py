import gradio as gr
import uuid
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.pdf_loader import load_pdf
from ingestion.chunker import chunk_documents
from ingestion.indexer import index_chunks
from rag.chain import ask_question
from voice.stt_service import transcribe
from voice.tts_service import synthesize
from langchain_core.messages import HumanMessage, AIMessage

# Global state to keep track of uploaded documents in this session
# In a real app, this would be user-session scoped.
current_session_docs = []

def process_pdfs(files):
    global current_session_docs
    if not files:
        return "No files uploaded."
    
    status_msg = []
    
    for file_obj in files:
        file_path = file_obj.name
        filename = os.path.basename(file_path)
        doc_id = str(uuid.uuid4())
        
        try:
            # 1. Load
            docs = load_pdf(file_path, doc_id)
            
            # 2. Chunk
            chunks = chunk_documents(docs)
            
            # 3. Index
            num_indexed = index_chunks(doc_id, filename, chunks)
            
            current_session_docs.append(doc_id)
            status_msg.append(f"✅ {filename}: Indexed {num_indexed} chunks.")
            
        except Exception as e:
            status_msg.append(f"❌ {filename}: Error - {str(e)}")
            
    return "\n".join(status_msg)

def chat_interface(message, history, audio_input):
    global current_session_docs
    
    # Check if voice input is provided
    if audio_input and not message:
        message = transcribe(audio_input)
        if not message:
            message = "Could not transcribe audio."
            
    if not message:
        return "", history, None
        
    # Convert Gradio history to LangChain messages
    chat_history = []
    for user_msg, ai_msg in history:
        chat_history.append(HumanMessage(content=user_msg))
        chat_history.append(AIMessage(content=ai_msg))
        
    # Ask question
    try:
        answer, docs = ask_question(message, chat_history, document_ids=current_session_docs)
    except Exception as e:
        answer = f"Error generating answer: {str(e)}"
        
    # Synthesize audio response
    audio_output = synthesize(answer)
    
    history.append((message, answer))
    
    return "", history, audio_output

def clear_session():
    global current_session_docs
    current_session_docs = []
    return "Session cleared.", [], None

def create_ui():
    with gr.Blocks(title="VOICEPDF - PDF Assistant") as demo:
        gr.Markdown("# VOICEPDF\n### Voice-Enabled Conversational PDF Assistant")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📄 Upload PDFs")
                file_input = gr.File(file_count="multiple", type="filepath")
                process_btn = gr.Button("Process / Index")
                status_output = gr.Textbox(label="Status", interactive=False)
                clear_btn = gr.Button("Clear Session Documents")
                
            with gr.Column(scale=2):
                gr.Markdown("### 💬 Chat")
                chatbot = gr.Chatbot(height=400)
                
                with gr.Row():
                    msg_input = gr.Textbox(label="Type your question here...", scale=4)
                    audio_input = gr.Audio(sources=["microphone"], type="filepath", label="🎙️ Voice Input", scale=1)
                
                submit_btn = gr.Button("Send")
                audio_output = gr.Audio(label="🔊 Answer Audio", autoplay=True, type="filepath")
                
        # Event wiring
        process_btn.click(fn=process_pdfs, inputs=[file_input], outputs=[status_output])
        
        # When submit button is clicked, we call chat_interface
        # Note: audio_input is reset to None after processing usually, but here we just pass it
        submit_btn.click(
            fn=chat_interface,
            inputs=[msg_input, chatbot, audio_input],
            outputs=[msg_input, chatbot, audio_output]
        )
        msg_input.submit(
            fn=chat_interface,
            inputs=[msg_input, chatbot, audio_input],
            outputs=[msg_input, chatbot, audio_output]
        )
        clear_btn.click(fn=clear_session, inputs=[], outputs=[status_output, chatbot, audio_output])
        
    return demo

if __name__ == "__main__":
    demo = create_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860)
